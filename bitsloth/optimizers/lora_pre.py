"""
LoRA-Pre Optimizer: 저랭크 옵티마이저 상태로 메모리 효율적 학습
기반 논문: "TAMING MOMENTUM: RETHINKING OPTIMIZER STATES THROUGH LOW-RANK APPROXIMATION" (ICLR 2026)

핵심 아이디어:
  EMA 모멘텀 업데이트 == 온라인 선형 회귀 (Theorem 3.1)
  → 모멘텀을 저랭크 행렬 mB @ mA 로 분해하여 메모리 절감

업데이트 규칙 (Newton's method):
  mB <- (1 - γ1) * mB + γ1 * g @ mA.T @ inv(mA @ mA.T)
  mA <- (1 - γ1) * mA + γ1 * inv(mB.T @ mB) @ mB.T @ g
  결합 조건: (1 - γ1)^2 = β1  →  γ1 = 1 - sqrt(β1)

2차 모멘텀: v = (vB @ vA)^◦2 로 재파라미터화하여 양수 보장
  결합 조건: (1 - γ2)^4 = β2  →  γ2 = 1 - β2^0.25
"""

import math
from typing import Optional, Iterable

import torch
from torch.optim import Optimizer


# ═══════════════════════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RANK_RATIO: float = 1 / 8  # 논문 권장: 1/8 rank로 comparable 성능
_TIKHONOV_LAMBDA: float = 1e-6  # 역행렬 수치 안정화 damping
_EPS: float = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# 저랭크 모멘텀 상태 컨테이너
# ═══════════════════════════════════════════════════════════════════════════════


class _LowRankMomentum:
    """
    단일 파라미터의 저랭크 모멘텀 상태 관리

    2D 이상: m ≈ B @ A  (B: [rows, r], A: [r, cols])
    1D/스칼라: 저랭크 분해 불필요, 풀벡터 유지
    """

    __slots__ = ("B", "A", "original_shape", "use_lowrank", "rows", "cols", "rank")

    def __init__(
        self,
        param: torch.Tensor,
        rank_ratio: float,
        init_std: float = 0.02,
    ) -> None:
        self.original_shape = param.shape
        shape = param.shape

        # 2D 이상이고 충분히 큰 파라미터에만 저랭크 적용
        if param.dim() >= 2 and param.numel() > 1:
            rows = shape[0]
            cols = param[0].numel()  # 나머지 차원을 평탄화
            rank = max(1, int(min(rows, cols) * rank_ratio))

            self.use_lowrank = True
            self.rows = rows
            self.cols = cols
            self.rank = rank
            # 논문 Algorithm 1: mA ~ N(0, 0.02), mB = 0
            self.B = torch.zeros(rows, rank, device=param.device, dtype=torch.float32)
            self.A = (
                torch.randn(rank, cols, device=param.device, dtype=torch.float32)
                * init_std
            )
        else:
            self.use_lowrank = False
            self.rows = self.cols = self.rank = 0
            self.B = torch.zeros_like(param, dtype=torch.float32)
            self.A = None

    def reconstruct(self) -> torch.Tensor:
        """m = B @ A 를 원래 shape로 복원"""
        if self.use_lowrank:
            return (self.B @ self.A).view(self.original_shape)
        return self.B.view(self.original_shape)

    def update(self, grad: torch.Tensor, gamma: float) -> None:
        """
        저랭크 모멘텀을 Newton step으로 업데이트

        Args:
            grad: 현재 gradient (original_shape)
            gamma: 업데이트 계수 (= 1 - sqrt(β))
        """
        if not self.use_lowrank:
            # 1D: 일반 EMA
            beta = (1.0 - gamma) ** 2
            self.B.mul_(beta).add_(grad.view_as(self.B).float(), alpha=1.0 - beta)
            return

        g = grad.view(self.rows, self.cols).float()
        lam = _TIKHONOV_LAMBDA

        # mB 업데이트: mB <- (1-γ) * mB + γ * g @ mA.T @ inv(mA @ mA.T + λI)
        AtAt = self.A @ self.A.t()  # [r, r]
        AtAt.diagonal().add_(lam)
        try:
            L = torch.linalg.cholesky(AtAt)
            gAt = g @ self.A.t()  # [rows, r]
            target_B = torch.cholesky_solve(gAt.t(), L).t()  # [rows, r]
        except torch.linalg.LinAlgError:
            target_B = g @ self.A.t() @ torch.linalg.pinv(AtAt)

        self.B.mul_(1.0 - gamma).add_(target_B, alpha=gamma)

        # mA 업데이트: mA <- (1-γ) * mA + γ * inv(mB.T @ mB + λI) @ mB.T @ g
        BtB = self.B.t() @ self.B  # [r, r]
        BtB.diagonal().add_(lam)
        try:
            L = torch.linalg.cholesky(BtB)
            Btg = self.B.t() @ g  # [r, cols]
            target_A = torch.cholesky_solve(Btg, L)  # [r, cols]
        except torch.linalg.LinAlgError:
            target_A = torch.linalg.pinv(BtB) @ self.B.t() @ g

        self.A.mul_(1.0 - gamma).add_(target_A, alpha=gamma)


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA-Pre Optimizer
# ═══════════════════════════════════════════════════════════════════════════════


class LoRAPreOptimizer(Optimizer):
    """
    LoRA-Pre: 저랭크 옵티마이저 상태로 메모리 효율적 학습

    Adam의 모멘텀을 저랭크 행렬로 분해하여 옵티마이저 메모리를 절감
    - 1차 모멘텀: m ≈ mB @ mA
    - 2차 모멘텀: v = (vB @ vA)^◦2 (양수 보장)
    - 파라미터 업데이트는 저랭크 재구성값으로 수행

    Args:
        params: 모델 파라미터
        lr: 학습률
        betas: (β1, β2) Adam 모멘텀 계수
        eps: 수치 안정화 epsilon
        weight_decay: L2 정규화
        rank_ratio: 저랭크 비율 (기본 1/8)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = _EPS,
        weight_decay: float = 0.0,
        rank_ratio: float = DEFAULT_RANK_RATIO,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr={lr}은 음수입니다.")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"betas[0]={betas[0]} 범위 오류")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas[1]={betas[1]} 범위 오류")
        if eps <= 0:
            raise ValueError(f"eps={eps}은 양수여야 합니다.")
        if not 0.0 < rank_ratio <= 1.0:
            raise ValueError(f"rank_ratio={rank_ratio} 범위 오류 (0, 1]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank_ratio=rank_ratio,
        )
        super().__init__(params, defaults)

    def _init_state(self, p: torch.Tensor, rank_ratio: float) -> None:
        state = self.state[p]
        state["step"] = 0
        state["m"] = _LowRankMomentum(p, rank_ratio)  # 1차 모멘텀
        state["v"] = _LowRankMomentum(p, rank_ratio)  # 2차 모멘텀 (|g| 기반)

    def add_new_params(self, new_params, **group_kwargs) -> None:
        """
        변환 후 새로 생긴 파라미터를 옵티마이저에 추가

        Args:
            new_params: 새 파라미터 iterable
            **group_kwargs: lr, weight_decay 등 오버라이드 가능
        """
        base = {k: v for k, v in self.defaults.items()}
        base.update(group_kwargs)
        params = [p for p in new_params if p not in self._params_set()]
        if params:
            base["params"] = params
            self.add_param_group(base)

    def _params_set(self) -> set:
        return {p for g in self.param_groups for p in g["params"]}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            rr = group["rank_ratio"]

            # 결합 계수 (논문 Appendix D, 수식 89)
            gamma1 = 1.0 - math.sqrt(beta1)  # 1차 모멘텀용
            gamma2 = 1.0 - beta2**0.25  # 2차 모멘텀용

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("LoRAPreOptimizer: sparse gradient 미지원")

                # 상태 초기화
                if len(self.state[p]) == 0:
                    self._init_state(p, rr)

                state = self.state[p]
                state["step"] += 1
                t = state["step"]

                grad = p.grad.float()

                # weight decay (AdamW 방식: gradient에 추가)
                if wd != 0.0:
                    grad = grad.add(p.float(), alpha=wd)

                # 1차 모멘텀 업데이트 (저랭크 Newton)
                state["m"].update(grad, gamma1)
                m_hat = state["m"].reconstruct() / (1.0 - beta1**t)

                # 2차 모멘텀 업데이트 (|g| 기반, 양수 보장)
                state["v"].update(grad.abs(), gamma2)
                v_recon = state["v"].reconstruct()
                # v = (vB @ vA)^◦2 → 재구성값을 제곱하여 양수 확보
                v_sq = v_recon.pow(2) / (1.0 - beta2**t)

                # 파라미터 업데이트
                denom = v_sq.sqrt().add_(eps)
                step_size = lr
                p.add_(m_hat.to(p.dtype) / denom.to(p.dtype), alpha=-step_size)

        return loss

    def set_weight_decay(self, wd: float) -> None:
        """모든 param_group의 weight_decay를 일괄 변경"""
        for group in self.param_groups:
            group["weight_decay"] = wd


# ═══════════════════════════════════════════════════════════════════════════════
# 팩토리 함수
# ═══════════════════════════════════════════════════════════════════════════════


def create_lora_pre_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = _EPS,
    weight_decay: float = 0.0,
    rank_ratio: float = DEFAULT_RANK_RATIO,
) -> LoRAPreOptimizer:
    """
    모델로부터 LoRAPreOptimizer 생성

    Args:
        model: PyTorch 모델
        lr: 학습률
        betas: Adam 모멘텀 계수
        eps: 수치 안정화
        weight_decay: L2 정규화
        rank_ratio: 저랭크 비율 (기본 1/8)

    Returns:
        LoRAPreOptimizer 인스턴스
    """
    return LoRAPreOptimizer(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        rank_ratio=rank_ratio,
    )
