"""
다중 GPU 유틸리티

convert.py의 GPU 관련 유틸리티를 bitsloth에 통합.
- device_map="auto" 환경에서 GPU 메모리 분배
- 다중 GPU 모델 병렬 지원
"""

import os
from typing import Optional

import torch
import torch.nn as nn


def get_gpu_count() -> int:
    """사용 가능한 GPU 수를 반환합니다."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def build_max_memory(reserve_gb: float = 2.0) -> Optional[dict]:
    """
    각 GPU의 가용 메모리를 기반으로 max_memory 딕셔너리를 생성합니다.
    accelerate의 device_map="auto"가 레이어를 균등 분배하도록 힌트를 제공합니다.

    Args:
        reserve_gb: 각 GPU마다 OS/드라이버용으로 예약할 GB
    Returns:
        {"0": "78GiB", "1": "78GiB", ..., "cpu": "48GiB"} 형태 딕셔너리
        GPU가 없으면 None 반환
    """
    n = get_gpu_count()
    if n == 0:
        return None

    max_memory: dict = {}
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        total_gb = prop.total_memory / (1024**3)
        usable = max(total_gb - reserve_gb, 1.0)
        max_memory[i] = f"{usable:.0f}GiB"

    # GPU가 부족할 경우 CPU 오프로드 허용 (48GB 기본값)
    cpu_offload_gb = int(os.environ.get("BITSLOTH_CPU_OFFLOAD_GB", "48"))
    max_memory["cpu"] = f"{cpu_offload_gb}GiB"

    return max_memory


def get_first_device(model: nn.Module) -> torch.device:
    """
    device_map="auto"로 분산된 모델에서 첫 번째 파라미터의 device를 반환합니다.
    입력 배치를 이 device로 올려야 합니다.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_gpu_summary() -> dict:
    """
    전체 GPU 정보를 출력하고 요약 딕셔너리를 반환합니다.

    Returns:
        {"num_gpus": int, "total_vram_gb": float, "max_memory": dict} 형태
    """
    n = get_gpu_count()
    if n == 0:
        print("[GPU] CUDA 없음 — CPU 모드로 실행")
        return {"num_gpus": 0, "total_vram_gb": 0.0, "max_memory": None}

    print(f"[GPU] 총 {n}개 감지")
    total_vram = 0.0
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        gb = prop.total_memory / (1024**3)
        total_vram += gb
        bf16 = torch.cuda.is_bf16_supported()
        print(f"  GPU[{i}] {prop.name}  {gb:.1f} GB  BF16: {bf16}")
    print(f"  합계 VRAM: {total_vram:.1f} GB")

    # 성능 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    max_memory = build_max_memory()
    return {
        "num_gpus": n,
        "total_vram_gb": total_vram,
        "max_memory": max_memory,
    }


def max_memory_to_str(max_memory: dict) -> str:
    """
    max_memory 딕셔너리를 환경변수용 문자열로 직렬화합니다.
    예: {0: "22GiB", 1: "22GiB", "cpu": "48GiB"} -> "0:22GiB|1:22GiB|cpu:48GiB"
    """
    return "|".join(f"{k}:{v}" for k, v in max_memory.items())


def max_memory_from_str(s: str) -> dict:
    """
    환경변수 문자열을 max_memory 딕셔너리로 역직렬화합니다.
    예: "0:22GiB|1:22GiB|cpu:48GiB" -> {0: "22GiB", 1: "22GiB", "cpu": "48GiB"}
    """
    result = {}
    for part in s.split("|"):
        k, v = part.split(":")
        key = int(k) if k.isdigit() else k
        result[key] = v
    return result
