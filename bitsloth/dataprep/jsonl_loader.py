"""
JSONL 데이터 로딩 유틸리티

convert.py의 데이터 로딩 관련 함수를 bitsloth에 통합.
- JSONL 파일 로드
- 멀티모달 (이미지) 지원
- Collate 함수 생성
"""

import json
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


def _load_image_safe(path: str):
    """
    이미지를 안전하게 로드합니다. 실패 시 None 반환.

    Args:
        path: 이미지 파일 경로
    Returns:
        PIL.Image 또는 None
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            return None
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _process_item(item: dict) -> dict:
    """
    JSONL 한 줄 아이템을 처리합니다.
    이미지는 경로(str)만 보존하고 실제 로드/전처리는 collate 시점으로 지연합니다.
    content가 문자열인 경우(assistant 턴 등) 자동으로 리스트로 변환합니다.
    """
    processed_messages = []
    for msg in item["messages"]:
        content = msg["content"]

        # content가 문자열이면 리스트로 정규화
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        content_out = []
        for c in content:
            if c["type"] == "text":
                content_out.append({"type": "text", "text": c["text"]})
            elif c["type"] == "image":
                # 이미지 경로만 저장 — 전처리는 collate 시점에 수행
                content_out.append({"type": "image", "image": c["image"]})
        processed_messages.append({"role": msg["role"], "content": content_out})
    return {"messages": processed_messages}


def load_train_data(jsonl_path: str) -> list[dict]:
    """
    JSONL 파일을 읽어 처리된 데이터 리스트를 반환합니다.

    Args:
        jsonl_path: JSONL 파일 경로
    Returns:
        처리된 데이터 리스트
    """
    data: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
                data.append(_process_item(item))
            except Exception as exc:
                print(f"  [경고] line {line_no} 처리 실패: {exc}")
    return data


def load_train_data_multi(jsonl_paths: list[str]) -> list[dict]:
    """
    여러 JSONL 파일을 읽어 합친 데이터 리스트를 반환합니다.

    Args:
        jsonl_paths: JSONL 파일 경로 리스트
    Returns:
        합쳐진 데이터 리스트
    """
    all_data: list[dict] = []
    for path in jsonl_paths:
        if not os.path.exists(path):
            print(f"  [경고] 파일 없음: {path}")
            continue
        chunk = load_train_data(path)
        print(f"  {path}: {len(chunk)}건")
        all_data.extend(chunk)
    print(f"  총 {len(all_data)}건")
    return all_data


class MultimodalDataset(Dataset):
    """멀티모달 (텍스트 + 이미지) 데이터셋"""

    def __init__(self, data: list[dict]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def make_collate_fn(tokenizer, image_processor=None, max_length: int = 2048):
    """
    DataLoader에 전달할 collate 함수를 반환합니다.

    Args:
        tokenizer: 토크나이저
        image_processor: 이미지 프로세서 (선택)
        max_length: 최대 시퀀스 길이
    Returns:
        collate 함수
    """

    def collate(batch_items: list[dict]) -> dict:
        texts: list[str] = []
        images: list = []

        for item in batch_items:
            msgs_for_template = []
            item_images = []

            for msg in item["messages"]:
                content_parts = []
                for c in msg["content"]:
                    if c["type"] == "text":
                        content_parts.append({"type": "text", "text": c["text"]})
                    elif c["type"] == "image" and image_processor is not None:
                        # 배치 단위로 이미지 로드 + 전처리
                        img = _load_image_safe(c["image"])
                        if img is not None:
                            item_images.append(image_processor(img))
                            content_parts.append(
                                {"type": "text", "text": "<|image_pad|>"}
                            )
                        else:
                            content_parts.append(
                                {"type": "text", "text": "[IMAGE_NOT_FOUND]"}
                            )
                msgs_for_template.append(
                    {"role": msg["role"], "content": content_parts}
                )

            try:
                text = tokenizer.apply_chat_template(
                    msgs_for_template,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = "\n".join(
                    f"{m['role']}: "
                    + " ".join(c["text"] for c in m["content"] if c["type"] == "text")
                    for m in msgs_for_template
                )

            texts.append(text)
            images.extend(item_images)

        enc_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        try:
            if images:
                encoded = tokenizer(texts, images=images, **enc_kwargs)
            else:
                encoded = tokenizer(texts, **enc_kwargs)
        except TypeError:
            encoded = tokenizer(texts, **enc_kwargs)

        encoded["labels"] = encoded["input_ids"].clone()
        return encoded

    return collate


def create_dataloader(
    data: list[dict],
    tokenizer,
    image_processor=None,
    batch_size: int = 1,
    max_length: int = 2048,
    num_gpus: int = 0,
) -> DataLoader:
    """
    JSONL 데이터로부터 DataLoader를 생성합니다.

    Args:
        data: 처리된 데이터 리스트
        tokenizer: 토크나이저
        image_processor: 이미지 프로세서 (선택)
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
        num_gpus: GPU 수 (num_workers 계산용)
    Returns:
        DataLoader 인스턴스
    """
    num_workers = min(max(num_gpus * 2, 2), 8)

    dataset = MultimodalDataset(data)
    collate_fn = make_collate_fn(tokenizer, image_processor, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_gpus > 0),
        persistent_workers=(num_workers > 0),
    )
    print(
        f"[DataLoader] num_workers={num_workers}, "
        f"pin_memory={num_gpus > 0}, "
        f"persistent_workers={num_workers > 0}"
    )
    return dataloader
