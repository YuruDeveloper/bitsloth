"""
BitNet b1.58 레이어 모듈
- STE (Straight-Through Estimator) 기반 ternary 양자화
- bitsloth 4bit 모델 위에 래핑하여 BitNet 학습 가능

설계 기반: bitsloth_BITNET_PLAN.md
참고: reference/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.md
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Ternary 양자화 함수들
# ═══════════════════════════════════════════════════════════════════════════════


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Per-tensor 양자화 → {-1, 0, 1} (1.58bit)

    W_ternary = α * clamp(round(W / α), -1, 1)
    α = w.abs().mean()
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    u = (w * scale).round().clamp(-1, 1) / scale
    return u


def weight_quant_with_scale(w: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    외부 α (학습 가능한 scale factor) 를 사용한 ternary 양자화

    W_ternary = α * clamp(round(W_latent / α), -1, 1)
    """
    return alpha * (w / alpha).round().clamp(-1, 1)


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token 양자화 → INT8
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# BitNet Linear 레이어
# ═══════════════════════════════════════════════════════════════════════════════


class BitNetLinear(nn.Module):
    """
    STE 기반 BitNet Linear 레이어

    핵심 동작:
    - 원본 weight는 latent로 유지 (gradient 전달됨)
    - forward 시 ternary 양자화 적용 (STE로 backward 연결)
    - α (scale factor) 는 레이어당 학습 가능한 스칼라

    기존 nn.Linear를 감싸거나 새로 생성 가능
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        original_linear: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if original_linear is not None:
            # 기존 Linear 레이어에서 weight 가져오기
            self.weight = original_linear.weight
            if original_linear.bias is not None:
                self.bias = original_linear.bias
            else:
                self.bias = None
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.bias = None
            self._reset_parameters()

        # 학습 가능한 scale factor (alpha)
        # 초기값: weight의 평균 절대값
        with torch.no_grad():
            init_alpha = self.weight.data.abs().mean().clamp(min=1e-5)
        self.alpha = nn.Parameter(init_alpha.clone())

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        STE ternary 양자화 적용 forward

        w_quant = w + (weight_quant_with_scale(w, α) - w).detach()
        → forward는 ternary로 계산
        → backward는 latent weight와 α로 gradient 전달
        """
        # STE: ternary 양자화 + detach trick
        w_ternary = weight_quant_with_scale(self.weight, self.alpha)
        w_ste = self.weight + (w_ternary - self.weight).detach()

        return F.linear(x, w_ste, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    @torch.no_grad()
    def get_ternary_weight(self) -> torch.Tensor:
        """추론용 ternary weight 반환"""
        return weight_quant_with_scale(self.weight, self.alpha)


class BitNetLinear4Bit(nn.Module):
    """
    bitsloth 4bit 모델 위에 래핑하는 BitNet Linear

    설계 (bitsloth_BITNET_PLAN.md):
    - 원본 4bit weight: 고정 (gradient 차단)
    - 추가 학습 파라미터: α (scale factor) 만
    - forward: 4bit → 역양자화 → STE ternary → 출력
    """

    def __init__(
        self,
        original_linear: nn.Module,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.original = original_linear
        self.in_features = in_features
        self.out_features = out_features

        # 원본 weight는 gradient 차단
        for param in self.original.parameters():
            param.requires_grad_(False)

        # 학습 가능한 scale factor
        # 초기값: 원본 weight의 추정 평균 절대값
        with torch.no_grad():
            w = self._get_full_precision_weight()
            init_alpha = w.abs().mean().clamp(min=1e-5)
        self.alpha = nn.Parameter(init_alpha.clone())

    def _get_full_precision_weight(self) -> torch.Tensor:
        """4bit weight를 full precision으로 역양자화"""
        if hasattr(self.original, "weight"):
            w = self.original.weight
            # bitsandbytes 4bit인 경우 dequantize 시도
            if hasattr(w, "CB") and hasattr(w, "SCB"):
                # bitsandbytes 4bit quantized weight
                from bitsandbytes.functional import dequantize_4bit

                return dequantize_4bit(w, w.quant_state)
            elif hasattr(w, "dequantize"):
                return w.dequantize()
            else:
                return w.float()
        raise ValueError("Cannot extract weight from original linear layer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """역양자화 → STE ternary → forward"""
        w = self._get_full_precision_weight()

        # STE ternary 양자화
        w_ternary = weight_quant_with_scale(w, self.alpha)
        w_ste = w + (w_ternary - w).detach()

        # bias 처리
        bias = None
        if hasattr(self.original, "bias") and self.original.bias is not None:
            bias = self.original.bias

        return F.linear(x, w_ste, bias)

    @torch.no_grad()
    def get_ternary_weight(self) -> torch.Tensor:
        """추론용 ternary weight 반환"""
        w = self._get_full_precision_weight()
        return weight_quant_with_scale(w, self.alpha)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, mode=4bit+STE_ternary"


# ═══════════════════════════════════════════════════════════════════════════════
# 점진적 변환 유틸리티
# ═══════════════════════════════════════════════════════════════════════════════

# MoE 변환 순서 (bitsloth_BITNET_PLAN.md 기반)
CONVERSION_ORDER = ["ffn", "attention", "router"]


def _is_ffn_module(name: str, module: nn.Module) -> bool:
    """FFN (Expert) 모듈인지 판별"""
    ffn_keywords = [
        "mlp",
        "expert",
        "ffn",
        "feed_forward",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    name_lower = name.lower()
    if any(kw in name_lower for kw in ffn_keywords):
        return True
    # Qwen3MoE의 experts 리스트
    if hasattr(module, "__class__") and "Expert" in module.__class__.__name__:
        return True
    return False


def _is_attention_module(name: str, module: nn.Module) -> bool:
    """Attention 모듈인지 판별"""
    attn_keywords = [
        "attn",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "out_proj",
    ]
    name_lower = name.lower()
    return any(kw in name_lower for kw in attn_keywords)


def _is_router_module(name: str, module: nn.Module) -> bool:
    """Router 모듈인지 판별"""
    router_keywords = ["gate", "router", "gate_proj"]
    name_lower = name.lower()
    if any(kw in name_lower for kw in router_keywords):
        # FFN의 gate_proj는 제외
        if "expert" in name_lower or "moe" in name_lower.split("."):
            return False
        return True
    return False


def _is_convertible_linear(name: str, module: nn.Module) -> bool:
    """변환 가능한 nn.Linear 모듈인지 판별"""
    if not isinstance(module, nn.Linear):
        return False
    # 이미 BitNetLinear이면 스킵
    if isinstance(module, (BitNetLinear, BitNetLinear4Bit)):
        return False
    # 너무 작은 레이어는 스킵 (embedding 등)
    if module.weight.numel() < 100:
        return False
    return True


def convert_linear_to_bitnet(
    module: nn.Linear,
    use_4bit_wrapper: bool = False,
) -> nn.Module:
    """
    nn.Linear → BitNetLinear 변환

    Args:
        module: 변환할 nn.Linear 모듈
        use_4bit_wrapper: True이면 BitNetLinear4Bit (bitsloth 4bit 모델용)
    """
    if use_4bit_wrapper:
        return BitNetLinear4Bit(
            original_linear=module,
            in_features=module.in_features,
            out_features=module.out_features,
        )
    else:
        return BitNetLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            original_linear=module,
        )


def apply_bitnet_conversion(
    model: nn.Module,
    target: str = "ffn",
    use_4bit_wrapper: bool = False,
    verbose: bool = True,
) -> dict:
    """
    모델의 특정 레이어를 BitNet으로 점진적 변환

    Args:
        model: 변환할 모델
        target: 변환 대상 ("ffn", "attention", "router", "all")
        use_4bit_wrapper: bitsloth 4bit 모델용 래퍼 사용 여부
        verbose: 변환 로그 출력 여부

    Returns:
        변환 통계 딕셔너리
    """
    stats = {
        "converted": 0,
        "skipped": 0,
        "total_params_before": 0,
        "total_params_after": 0,
    }

    for name, param in model.named_parameters():
        stats["total_params_before"] += param.numel()

    target_modules = []
    for name, module in model.named_modules():
        if not _is_convertible_linear(name, module):
            continue

        should_convert = False
        if target == "all":
            should_convert = True
        elif target == "ffn" and _is_ffn_module(name, module):
            should_convert = True
        elif target == "attention" and _is_attention_module(name, module):
            should_convert = True
        elif target == "router" and _is_router_module(name, module):
            should_convert = True

        if should_convert:
            target_modules.append((name, module))

    # 변환 수행
    for name, module in target_modules:
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]

        parent = model
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)

        bitnet_module = convert_linear_to_bitnet(module, use_4bit_wrapper)
        setattr(parent, child_name, bitnet_module)
        stats["converted"] += 1

        if verbose:
            print(
                f"  [BitNet] {name} → BitNetLinear ({module.in_features}→{module.out_features})"
            )

    # 변환 후 파라미터 수
    for name, param in model.named_parameters():
        stats["total_params_after"] += param.numel()

    # 학습 가능/고정 파라미터 분류
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    stats["trainable_params"] = trainable
    stats["frozen_params"] = frozen

    if verbose:
        print(f"\n  [BitNet] 변환 완료: {stats['converted']}개 레이어")
        print(f"  [BitNet] 학습 가능: {trainable:,} 파라미터 (α scale factors)")
        print(f"  [BitNet] 고정: {frozen:,} 파라미터 (원본 weight)")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MoE 점진적 변환 헬퍼
# ═══════════════════════════════════════════════════════════════════════════════


def convert_moe_progressive(
    model: nn.Module,
    use_4bit_wrapper: bool = False,
    verbose: bool = True,
) -> list:
    """
    MoE 모델 점진적 변환 (FFN → Attention → Router 순서)

    Args:
        model: MoE 모델
        use_4bit_wrapper: bitsloth 4bit 모델용 래퍼 사용 여부
        verbose: 변환 로그 출력 여부

    Returns:
        각 단계별 변환 통계 리스트
    """
    all_stats = []

    for step in CONVERSION_ORDER:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  [BitNet MoE] 단계: {step.upper()} 변환 시작")
            print(f"{'=' * 60}")

        stats = apply_bitnet_conversion(
            model,
            target=step,
            use_4bit_wrapper=use_4bit_wrapper,
            verbose=verbose,
        )
        all_stats.append({"step": step, **stats})

    return all_stats
