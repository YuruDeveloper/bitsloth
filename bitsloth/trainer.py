# Copyright 2023-present Daniel Han-Chen & the Bitsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import psutil
import warnings
from dataclasses import dataclass, field
from typing import Optional, List
from functools import wraps

import torch
import trl
import inspect
from trl import SFTTrainer
from . import is_bfloat16_supported
from bitsloth.utils import (
    configure_padding_free,
    configure_sample_packing,
    enable_padding_free_metadata,
    enable_sample_packing,
    get_gpu_count,
)
from bitsloth_zoo.training_utils import (
    bitsloth_train as _bitsloth_train,
)
from bitsloth_zoo.vision_utils import (
    BitslothVisionDataCollator,
)
from bitsloth_zoo.hf_utils import get_transformers_model_type
from bitsloth_zoo.utils import Version
import dataclasses

__all__ = [
    "BitslothTrainingArguments",
    "BitslothTrainer",
    "bitsloth_train",
    "_patch_trl_trainer",
    "BitslothVisionDataCollator",
    "QGaloreConfig",
    "BitNetConfig",
    "TwoStageBitNetScheduler",
    "save_model_with_meta",
]

logger = logging.getLogger(__name__)

_AUTO_PADDING_FREE_ENV_DISABLED = os.environ.get(
    "BITSLOTH_DISABLE_AUTO_PADDING_FREE", ""
).strip().lower() in {"1", "true", "yes", "on"}

PADDING_FREE_BLOCKLIST = {
    "gemma2",  # - gemma2:  Uses slow_attention_softcapping which has torch.compile issues
    "gpt_oss",  # - gpt_oss: Uses Flex Attention which doesn't handle padding_free correctly
}


def _should_pack(config) -> bool:
    if config is None or not getattr(config, "packing", False):
        return False
    return not getattr(config, "_bitsloth_disable_auto_packing", False)


def _should_auto_padding_free(config) -> bool:
    if (
        config is None
        or _AUTO_PADDING_FREE_ENV_DISABLED
        or getattr(config, "packing", False)
    ):
        return False
    return getattr(config, "padding_free", None) is None


def _disable_sample_packing(config):
    if config is None:
        return
    for attr, value in (("packing", False), ("padding_free", False)):
        if hasattr(config, attr):
            setattr(config, attr, value)
    if hasattr(config, "remove_unused_columns"):
        setattr(config, "remove_unused_columns", True)
    setattr(config, "_bitsloth_disable_auto_packing", True)


_AUTO_PACK_SKIP_MESSAGES = (
    "packing is not supported",
    "padding-free training",
    "passing a custom data collator",
)


def _should_skip_auto_packing_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(msg in message for msg in _AUTO_PACK_SKIP_MESSAGES)


# Bitsloth gradient accumulation fix:
from transformers import __version__ as transformers_version, ProcessorMixin

if Version(transformers_version) > Version("4.45.2"):

    def bitsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)

else:

    def bitsloth_train(trainer, *args, **kwargs):
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "Bitsloth: Our custom gradient accumulation fixed trainer does not support other arguments.\n"
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"
                "`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`"
            )
        print(
            "Bitsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"
            "`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`"
        )
        return _bitsloth_train(trainer)


try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments


@dataclass
class BitNetConfig:
    """Configuration for BitNet b1.58 + LoRA-Pre training.

    Pass an instance of this class to ``BitslothTrainingArguments`` (via
    ``bitnet_config``) to enable BitNet progressive conversion and training.
    """

    # 점진적 변환 대상
    target: str = "ffn"  # "ffn", "attention", "router", "all"
    # unsloth 4bit 모델용 래퍼 사용
    use_4bit_wrapper: bool = True
    # LoRA-Pre 옵티마이저 저랭크 비율
    rank_ratio: float = 1 / 8
    # 2단계 LR/WD 전략 (BitNet 논문 권장)
    two_stage_lr: bool = True
    peak_lr: float = 1.2e-3
    end_lr: float = 8e-4
    warmup_steps: int = 375
    weight_decay_stage1: float = 0.1
    # 점진적 변환 시 epoch 수 (변환 그룹당)
    epochs_per_group: int = 1
    # gradient clipping
    grad_clip: float = 1.0


class TwoStageBitNetScheduler:
    """
    BitNet b1.58 논문의 2단계 학습 전략 구현.

    Stage 1 (0 ~ midpoint):  높은 peak LR, weight_decay=0.1
    Stage 2 (midpoint ~ end): LR 선형 감소, weight_decay=0.0
    """

    def __init__(
        self,
        optimizer,
        peak_lr: float,
        end_lr: float,
        total_steps: int,
        warmup_steps: int = 375,
        wd_stage1: float = 0.1,
    ) -> None:
        self.opt = optimizer
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.mid_steps = total_steps // 2
        self.wd_stage1 = wd_stage1
        self._step = 0

    def step(self) -> None:
        self._step += 1
        t = self._step
        ws = self.warmup_steps
        ms = self.mid_steps
        ts = self.total_steps

        # LR 계산
        if t <= ws:
            # 웜업: 선형 증가
            lr = self.peak_lr * (t / max(ws, 1))
        elif t <= ms:
            # Stage 1: peak LR 유지
            lr = self.peak_lr
        else:
            # Stage 2: peak → end_lr 선형 감소
            ratio = (t - ms) / max(ts - ms, 1)
            lr = self.peak_lr + ratio * (self.end_lr - self.peak_lr)
        lr = max(lr, self.end_lr)

        # Weight Decay
        wd = self.wd_stage1 if t <= ms else 0.0

        # 옵티마이저 적용
        for group in self.opt.param_groups:
            group["lr"] = lr
        if hasattr(self.opt, "set_weight_decay"):
            self.opt.set_weight_decay(wd)

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, d: dict) -> None:
        self._step = d["step"]


@dataclass
class QGaloreConfig:
    """Configuration for Q-GaLore optimizer integration.

    Pass an instance of this class to ``BitslothTrainingArguments`` (via
    ``q_galore_config``) to enable Q-GaLore training.
    """

    rank: int = 256
    update_proj_gap: int = 200
    scale: float = 0.25
    proj_quant: bool = True
    proj_quant_group_size: int = -1
    proj_quant_n_bit: int = 4
    weight_quant: bool = False
    stochastic_round: bool = True
    weight_group_size: int = 128
    cos_threshold: float = 0.4
    gamma_proj: float = 2.0
    queue_size: int = 5
    target_modules: Optional[List[str]] = None


class BitslothTrainingArguments(TrainingArguments):
    def __init__(
        self,
        embedding_learning_rate: float = None,
        q_galore_config: Optional[QGaloreConfig] = None,
        bitnet_config: Optional[BitNetConfig] = None,
        *args,
        **kwargs,
    ):
        self.q_galore_config = q_galore_config
        self.bitnet_config = bitnet_config
        self.embedding_learning_rate = embedding_learning_rate
        super().__init__(*args, **kwargs)
        self.embedding_learning_rate = embedding_learning_rate


def _create_bitsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr=5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = {
        "non_embeddings": {},
        "embeddings": {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[: -len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".") + 1 :]
            print(
                f"Bitsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}."
            )
            param_groups["embeddings"][name] = param
        else:
            param_groups["non_embeddings"][name] = param

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["non_embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


class BitslothTrainer(SFTTrainer):
    def create_optimizer(self):
        # --- BitNet + LoRA-Pre optimizer ---
        bitnet_config = getattr(self.args, "bitnet_config", None)
        if bitnet_config is not None and self.optimizer is None:
            return self._create_bitnet_optimizer(bitnet_config)

        # --- Q-GaLore optimizer ---
        q_galore_config = getattr(self.args, "q_galore_config", None)
        if q_galore_config is not None and self.optimizer is None:
            embedding_lr = getattr(self.args, "embedding_learning_rate", None)
            return self._create_q_galore_optimizer(q_galore_config, embedding_lr)

        # --- Embedding-LR optimizer ---
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = _create_bitsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        return self.optimizer

    def _create_bitnet_optimizer(self, config: "BitNetConfig"):
        """Build the LoRA-Pre optimizer for BitNet training."""
        from bitsloth.optimizers.lora_pre import create_lora_pre_optimizer

        lr = config.peak_lr
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        eps = self.args.adam_epsilon

        self.optimizer = create_lora_pre_optimizer(
            self.model,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=config.weight_decay_stage1,
            rank_ratio=config.rank_ratio,
        )

        print(
            f"🦥 Bitsloth: BitNet + LoRA-Pre enabled — "
            f"rank_ratio=1/{int(1 / config.rank_ratio)}, "
            f"peak_lr={config.peak_lr:.2e} → {config.end_lr:.2e}"
        )

        return self.optimizer

    def convert_to_bitnet(self, config: Optional[BitNetConfig] = None):
        """
        모델의 Linear 레이어를 BitNet으로 점진적 변환.

        Args:
            config: BitNet 설정. None이면 self.args.bitnet_config 사용
        """
        if config is None:
            config = getattr(self.args, "bitnet_config", None)
        if config is None:
            raise ValueError("BitNetConfig가 설정되지 않았습니다.")

        from bitsloth.models.bitnet import (
            apply_bitnet_conversion,
            BitNetLinear,
            BitNetLinear4Bit,
        )

        if config.target == "all":
            stats = apply_bitnet_conversion(
                self.model,
                target="all",
                use_4bit_wrapper=config.use_4bit_wrapper,
            )
            return [stats]
        else:
            return self._progressive_convert_and_train(config)

    def _register_bitnet_params(self, already_converted: set):
        """변환된 BitNet 파라미터를 옵티마이저에 등록"""
        from bitsloth.models.bitnet import BitNetLinear, BitNetLinear4Bit

        if self.optimizer is None:
            return

        new_params = []
        for name, mod in self.model.named_modules():
            if name in already_converted:
                continue
            if isinstance(mod, (BitNetLinear, BitNetLinear4Bit)):
                if hasattr(mod, "alpha") and isinstance(mod.alpha, torch.nn.Parameter):
                    new_params.append(mod.alpha)
                if hasattr(mod, "weight") and isinstance(
                    mod.weight, torch.nn.Parameter
                ):
                    new_params.append(mod.weight)

        if new_params:
            self.optimizer.add_new_params(new_params)
            print(f"  [BitNet] 옵티마이저에 {len(new_params)}개 파라미터 등록")

    def _progressive_convert_and_train(self, config: BitNetConfig):
        """
        MoE 점진적 변환 + 학습 (FFN → Attention → Router)

        각 단계마다:
        1. 레이어 변환 (nn.Linear → BitNetLinear)
        2. 옵티마이저에 새 파라미터 등록
        3. 해당 레이어만 학습
        4. gradient checkpointing 적용
        """
        from bitsloth.models.bitnet import (
            CONVERSION_ORDER,
            apply_bitnet_conversion,
            BitNetLinear,
            BitNetLinear4Bit,
        )

        all_stats = []
        already_converted = set()

        for step in CONVERSION_ORDER:
            print(f"\n{'=' * 60}")
            print(f"  [BitNet] 단계: {step.upper()} 변환 시작")
            print(f"{'=' * 60}")

            # 1) 변환
            stats = apply_bitnet_conversion(
                self.model,
                target=step,
                use_4bit_wrapper=config.use_4bit_wrapper,
            )
            all_stats.append({"step": step, **stats})

            if stats["converted"] == 0:
                print(f"  [skip] {step}: 대상 레이어 없음")
                continue

            # 2) 옵티마이저 생성 (처음 한 번만)
            if self.optimizer is None:
                from bitsloth.optimizers.lora_pre import create_lora_pre_optimizer

                self.optimizer = create_lora_pre_optimizer(
                    self.model,
                    lr=config.peak_lr,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=config.weight_decay_stage1,
                    rank_ratio=config.rank_ratio,
                )

            # 3) 새 파라미터 등록
            self._register_bitnet_params(already_converted)
            already_converted |= set(
                name
                for name, mod in self.model.named_modules()
                if isinstance(mod, (BitNetLinear, BitNetLinear4Bit))
            )

            # 4) gradient checkpointing
            if hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    print("  [✓] gradient checkpointing 활성화")
                except Exception:
                    pass

        print("\n=== 점진적 변환 완료 ===")
        return all_stats

    def _create_q_galore_optimizer(self, config: "QGaloreConfig", embedding_lr=None):
        """Build the Q-GaLore optimizer from a QGaloreConfig."""
        from bitsloth.optimizers.q_galore_adamw import (
            QGaLoreAdamW8bit,
            make_q_galore_param_groups,
            install_weight_quant_hooks,
        )

        lr = self.args.learning_rate
        weight_decay = self.args.weight_decay

        param_groups = make_q_galore_param_groups(
            self.model,
            lr=lr,
            weight_decay=weight_decay,
            rank=config.rank,
            update_proj_gap=config.update_proj_gap,
            scale=config.scale,
            proj_quant=config.proj_quant,
            proj_quant_group_size=config.proj_quant_group_size,
            proj_quant_n_bit=config.proj_quant_n_bit,
            weight_quant=config.weight_quant,
            stochastic_round=config.stochastic_round,
            weight_group_size=config.weight_group_size,
            cos_threshold=config.cos_threshold,
            gamma_proj=config.gamma_proj,
            queue_size=config.queue_size,
            target_modules=config.target_modules,
        )

        # --- Split embedding params with custom LR (Fix #2) ---
        if embedding_lr is not None:
            # Build a fast param->name lookup (O(N) instead of O(N*M))
            param_to_name = {id(p): name for name, p in self.model.named_parameters()}

            new_groups = []
            for group in param_groups:
                if "rank" in group:
                    # GaLore group — keep as-is (embeddings are never in here)
                    new_groups.append(group)
                    continue
                # Non-GaLore group: split out embedding params
                embed_params = []
                other_params = []
                for p in group["params"]:
                    # Check if this param belongs to a modules_to_save embedding
                    name = param_to_name.get(id(p))
                    if name and name.endswith("modules_to_save.default.weight"):
                        partial_name = name[: -len(".modules_to_save.default.weight")]
                        partial_name = partial_name[partial_name.rfind(".") + 1 :]
                        print(
                            f"Bitsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}."
                        )
                        embed_params.append(p)
                    else:
                        other_params.append(p)
                if other_params:
                    other_group = dict(group)
                    other_group["params"] = other_params
                    new_groups.append(other_group)
                if embed_params:
                    embed_group = dict(group)
                    embed_group["params"] = embed_params
                    embed_group["lr"] = embedding_lr
                    new_groups.append(embed_group)
            param_groups = new_groups

        # --- Forward optimizer hyperparameters (Fix #3) ---
        self.optimizer = QGaLoreAdamW8bit(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        # Initialize INT8 weight quantization if enabled
        if config.weight_quant:
            QGaLoreAdamW8bit.init_weight_quantization(
                self.model,
                param_groups,
                group_size=config.weight_group_size,
                stochastic=config.stochastic_round,
            )
            # Forward pre-hooks dequantize INT8 weights to float before each
            # forward pass, allowing the optimizer to free float weight memory
            # between steps.
            install_weight_quant_hooks(self.model)

        n_galore = sum(len(g["params"]) for g in param_groups if "rank" in g)
        n_other = sum(len(g["params"]) for g in param_groups if "rank" not in g)
        print(
            f"🦥 Bitsloth: Q-GaLore enabled — "
            f"{n_galore} GaLore params (rank={config.rank}), "
            f"{n_other} standard params."
        )

        return self.optimizer


# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth
def _resolve_trainer_params(trainer_class, init_fn):
    """Resolve the real named parameters for a trainer __init__.

    Some TRL trainers (e.g., ORPOTrainer in TRL 0.27.1) are thin wrappers
    with only ``def __init__(self, *args, **kwargs)``.  For those, walk the
    MRO and return the first parent class that has real named parameters.
    """
    params = inspect.signature(init_fn).parameters
    named = {
        k
        for k, v in params.items()
        if v.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and k != "self"
    }
    if named:
        return set(params.keys())

    # Thin wrapper detected - walk MRO for real signature
    for cls in trainer_class.__mro__[1:]:
        if cls is object:
            continue
        parent_init = cls.__dict__.get("__init__")
        if parent_init is None:
            continue
        try:
            parent_params = inspect.signature(parent_init).parameters
            parent_named = {
                k
                for k, v in parent_params.items()
                if v.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and k != "self"
            }
            if parent_named:
                return set(parent_params.keys())
        except (ValueError, TypeError):
            continue
    return set(params.keys())


def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # All Trainer tokenizer are now called processing_class
        trainer_params = _resolve_trainer_params(trainer_class, original_init)

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if ("args" in kwargs) and (Version(trl) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove("self")
            trainer_params.remove("args")

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field
                for field in dataclasses.fields(config_class)
                if field.init
            }

            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
            from transformers import TrainingArguments

            moved_params = set(inspect.signature(config_class).parameters.keys()) - set(
                inspect.signature(TrainingArguments).parameters.keys()
            )

            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params:
                    trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            # Reinitialising config class with parameters (that were none initially but populated on first init)
            # causes the 2nd init to fail as there are mutual exclusive checks on pairs of parameters.
            # Refer: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L499-L502 for example
            # So we only create config class if the previous init was not TrainingArguments
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                config = training_args

            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        original_init(self, *args, **kwargs)

    return new_init


def _patch_sft_trainer_auto_packing(trl_module):
    sft_trainer = getattr(trl_module, "SFTTrainer", None)
    if sft_trainer is None:
        return
    if getattr(sft_trainer, "_bitsloth_auto_packing_wrapped", False):
        return

    original_init = sft_trainer.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        config_arg = None
        if len(args) >= 2:
            config_arg = args[1]
        else:
            config_arg = kwargs.get("args")

        # Check if model type is unsupported for padding_free
        model = kwargs.get("model")
        is_unsupported_model = False
        is_vlm = False
        if model is not None:
            model_config = getattr(model, "config", None)
            if model_config is not None:
                model_types = get_transformers_model_type(model_config)
                # Blocklist: models that don't work correctly with padding_free
                is_unsupported_model = any(
                    x in PADDING_FREE_BLOCKLIST for x in model_types
                )

                # Check if VLM
                architectures = getattr(model_config, "architectures", None)
                if architectures is None:
                    architectures = []
                is_vlm = any(
                    x.endswith("ForConditionalGeneration") for x in architectures
                )
                is_vlm = is_vlm or hasattr(model_config, "vision_config")

        processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")
        data_collator = kwargs.get("data_collator")

        # We also disable vision language models for padding free collators
        blocked = (
            (data_collator is not None)
            or isinstance(processing_class, ProcessorMixin)
            or is_vlm
            or is_unsupported_model
            or (
                os.environ.get("BITSLOTH_RETURN_LOGITS", "0") == "1"
            )  # Disable padding free on forced logits
        )
        requested_pack = bool(getattr(config_arg, "packing", False))
        if blocked:
            if hasattr(config_arg, "packing"):
                setattr(config_arg, "packing", False)
            if hasattr(config_arg, "padding_free"):
                setattr(config_arg, "padding_free", False)

        if blocked and requested_pack:
            reason = "custom data collator"
            if data_collator is None and isinstance(processing_class, ProcessorMixin):
                reason = "processor-based model"
            elif is_vlm:
                reason = "vision-language model"
            elif is_unsupported_model:
                reason = f"unsupported model type(s): {', '.join(model_types)}"
            message = f"Bitsloth: Sample packing skipped ({reason} detected)."
            print(message)

        packing_active = False
        if _should_pack(config_arg) and not blocked:
            configure_sample_packing(config_arg)
            packing_active = True
            logger.info("Bitsloth: Sample packing enabled for SFTTrainer instance.")

        # Resolve padding_free: None (default) = auto-enable unless env-disabled or packing
        auto_padding_free_active = False
        padding_free_requested = getattr(config_arg, "padding_free", None) is True
        if not blocked:
            if padding_free_requested:
                configure_padding_free(config_arg)
            elif _should_auto_padding_free(config_arg):
                configure_padding_free(config_arg)
                auto_padding_free_active = True
                logger.info(
                    "Bitsloth: Padding-free batching auto-enabled for SFTTrainer instance."
                )

        try:
            original_init(self, *args, **kwargs)
        except ValueError as exc:
            if packing_active and _should_skip_auto_packing_error(exc):
                logger.info(
                    "Bitsloth: Auto sample packing failed because trainer reported an incompatible setup (%s).",
                    exc,
                )
                _disable_sample_packing(config_arg)
                packing_active = False
                original_init(self, *args, **kwargs)
            else:
                raise

        trainer_args = getattr(self, "args", None)
        trainer_packing = bool(trainer_args and getattr(trainer_args, "packing", False))
        trainer_padding_free = bool(
            trainer_args and getattr(trainer_args, "padding_free", False)
        )

        if blocked and trainer_args is not None:
            # Mirror the block on the trainer args to avoid re-enabling later
            setattr(trainer_args, "packing", False)
            setattr(trainer_args, "padding_free", False)

        if (
            not blocked
            and trainer_packing
            and (packing_active or _should_pack(trainer_args))
        ):
            enable_sample_packing(self.model, self)
            print(
                "🦥 Bitsloth: Packing enabled - training is >2x faster and uses less VRAM!"
            )
        elif not blocked and trainer_padding_free:
            enable_padding_free_metadata(self.model, self)
            message = (
                "🦥 Bitsloth: Padding-free auto-enabled, enabling faster training."
                if auto_padding_free_active
                else "🦥 Bitsloth: Padding-free enabled, enabling faster training."
            )
            print(message)

    sft_trainer.__init__ = new_init
    sft_trainer._bitsloth_auto_packing_wrapped = True


def _patch_trl_trainer():
    import trl

    if hasattr(trl, "__BITSLOTH_BACKWARDS_COMPATIBLE__"):
        return
    if Version(trl) <= Version("0.11.0"):
        return

    import trl.trainer

    trl_classes = dir(trl.trainer)
    trl_trainers = set(
        x[: -len("Trainer")] for x in trl_classes if x.endswith("Trainer")
    )
    trl_configs = set(x[: -len("Config")] for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:
            exec(
                f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)",
                globals(),
            )
        except:
            continue

    _patch_sft_trainer_auto_packing(trl)

    trl.__BITSLOTH_BACKWARDS_COMPATIBLE__ = True


# ═══════════════════════════════════════════════════════════════════════════════
# BitNet 모델 저장 유틸리티
# ═══════════════════════════════════════════════════════════════════════════════


def save_model_with_meta(
    model,
    tokenizer,
    optimizer=None,
    save_dir: str = "./output",
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    private: bool = True,
    token: Optional[str] = None,
) -> None:
    """
    모델, 토크나이저, 옵티마이저 상태를 저장하고 convert_info.json 메타데이터를 생성합니다.

    Args:
        model: 저장할 모델
        tokenizer: 토크나이저
        optimizer: 옵티마이저 (선택)
        save_dir: 저장 디렉토리
        push_to_hub: Hub 업로드 여부
        hub_model_id: Hub 레포 ID
        private: 비공개 레포 여부
        token: HuggingFace 토큰
    """
    import json as json_module

    os.makedirs(save_dir, exist_ok=True)

    # 모델 & 토크나이저 저장
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 옵티마이저 상태 저장
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    # BitNet 레이어 정보 수집
    from bitsloth.models.bitnet import BitNetLinear, BitNetLinear4Bit

    bitnet_layers = []
    alpha_values = []
    for name, mod in model.named_modules():
        if isinstance(mod, (BitNetLinear, BitNetLinear4Bit)):
            bitnet_layers.append(name)
            if hasattr(mod, "alpha"):
                val = (
                    mod.alpha.item() if hasattr(mod.alpha, "item") else float(mod.alpha)
                )
                alpha_values.append(val)

    # 메타데이터 생성
    n_gpu = get_gpu_count()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    meta = {
        "base_model": getattr(getattr(model, "config", {}), "_name_or_path", "unknown"),
        "conversion_type": "BitNet b1.58 + LoRA-Pre",
        "quantization": "ternary_STE",
        "optimizer": "LoRA-Pre (ICLR 2026)",
        "multi_gpu": n_gpu > 1,
        "num_gpus": n_gpu,
        "device_strategy": "device_map=auto (model parallel)",
        "bitnet_layers": len(bitnet_layers),
        "trainable_params": trainable,
        "frozen_params": frozen,
        "alpha_stats": {
            "count": len(alpha_values),
            "min": min(alpha_values) if alpha_values else 0,
            "max": max(alpha_values) if alpha_values else 0,
            "mean": sum(alpha_values) / len(alpha_values) if alpha_values else 0,
        },
        "paper_references": [
            "The Era of 1-bit LLMs: Training Tips, Code and FAQ",
            "Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation (ICLR 2026)",
        ],
    }
    with open(os.path.join(save_dir, "convert_info.json"), "w") as f:
        json_module.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[저장 완료] {save_dir}")
    print(f"  BitNet 레이어: {len(bitnet_layers)}개")
    print(f"  학습 가능: {trainable:,} 파라미터")
    print(f"  고정: {frozen:,} 파라미터")

    # Hub 업로드
    if push_to_hub and hub_model_id:
        try:
            from huggingface_hub import HfApi, create_repo

            api = HfApi(token=token)
            create_repo(
                repo_id=hub_model_id,
                token=token,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=save_dir,
                repo_id=hub_model_id,
                repo_type="model",
                commit_message="BitNet b1.58 + LoRA-Pre model",
                ignore_patterns=["optimizer.pt"],
            )
            print(f"[Hub 업로드 완료] https://huggingface.co/{hub_model_id}")
        except Exception as exc:
            print(f"[Hub 업로드 실패] {exc}")
