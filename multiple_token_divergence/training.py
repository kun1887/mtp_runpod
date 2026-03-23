import sys
import time

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn
import os

import torch

from omegaconf import DictConfig, ListConfig, OmegaConf
from small_models_evaluation.pfa_evaluation import pfa_training_evaluation
from small_models_evaluation.creativity_evaluation import creativity_evaluation
from modules.self_prediction import initialize_mtp_layer_with_last_layer_weights

from torch import nn
from torch.distributed import destroy_process_group, init_process_group

from utils.checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)

from utils.meters import MultiMeter

from torch.backends.cuda import sdp_kernel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training
from torchtune import utils as torchtune_utils
from torchtune.data import padded_collate_sft, padded_collate_packed
from dataset_classes.utils import dummy_collate
from dataset_classes.packing_on_the_fly import PackedOnTheFlyDataset
from torchtune.datasets import ConcatDataset
try:
    from torchtune.modules.loss import SFTLoss, LinearCrossEntropyLoss
except ImportError:
    from torchtune.modules.loss import CEWithChunkedOutputLoss as LinearCrossEntropyLoss
    class SFTLoss:
        """Stub for torchtune versions that don't have SFTLoss (< 0.7)."""
        pass
from torchtune.datasets._text_completion import TextCompletionDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.activations import apply_selective_activation_checkpointing
try:
    from torchtune.modules.moe import utils as moe_utils
except ImportError:
    import types
    moe_utils = types.SimpleNamespace(use_grouped_mm=True)
try:
    from torchtune.modules.embedding_utils import resize_token_embeddings
except ImportError:
    def resize_token_embeddings(model: "nn.Module", new_vocab_size: int) -> None:
        """Resize token embedding and output projection layers to new_vocab_size."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings != new_vocab_size:
                old_weight = module.weight.data
                new_embed = nn.Embedding(new_vocab_size, module.embedding_dim, device=old_weight.device, dtype=old_weight.dtype)
                new_embed.weight.data[:min(old_weight.size(0), new_vocab_size)] = old_weight[:min(old_weight.size(0), new_vocab_size)]
                parent = model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_embed)
from torch.distributed.tensor.parallel import parallelize_module
from torchtune.training.lr_schedulers import get_lr
try:
    from torchtune.training.quantization import (
        convert_to_float8_training,
        is_fp8_tensorwise_scaling,
    )
except ImportError:
    def convert_to_float8_training(model, recipe_name=None):
        """Stub: FP8 training not available in this torchtune version."""
        return model
    def is_fp8_tensorwise_scaling(recipe_name=None) -> bool:
        return False

try:
    from torchtune.training import (
        DummyProfiler,
        PROFILER_KEY,
        VALID_BACKENDS_FOR_MEMORY_STATS,
    )
except ImportError:
    from torchtune.training import DummyProfiler, PROFILER_KEY
    VALID_BACKENDS_FOR_MEMORY_STATS = {"cuda"}

from huggingface_hub import snapshot_download
from torch.distributed._tensor import DTensor, Shard

from tqdm import tqdm

# ── Compatibility shims for torchtune < 0.7 ──────────────────────────────────
import contextlib as _contextlib

if not hasattr(training, "ParallelDims"):
    class _ParallelDims:
        """Stub for torchtune.training.ParallelDims (added in 0.7).
        Supports simple dp_shard (FSDP) only; TP/CP/dp_replicate raise NotImplementedError.
        """
        def __init__(self, dp_replicate, dp_shard, tp, cp, world_size):
            self.tp = tp
            self.cp = cp
            self.dp_replicate = dp_replicate
            self.world_size = world_size
            non_dp = tp * cp * max(dp_replicate, 1)
            self.dp_shard = (world_size // non_dp) if dp_shard == -1 else dp_shard

        @property
        def tp_enabled(self): return self.tp > 1
        @property
        def cp_enabled(self): return self.cp > 1
        @property
        def dp_shard_enabled(self): return self.dp_shard > 1
        @property
        def dp_replicate_enabled(self): return self.dp_replicate > 1
        @property
        def dp_enabled(self): return self.dp_shard_enabled or self.dp_replicate_enabled
        @property
        def non_data_parallel_size(self): return max(self.tp * self.cp, 1)

        def build_mesh(self, device_type):
            if self.tp_enabled or self.cp_enabled or self.dp_replicate_enabled:
                raise NotImplementedError(
                    "ParallelDims stub only supports dp_shard (FSDP). "
                    "TP / CP / dp_replicate require torchtune >= 0.7."
                )
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh(device_type, (self.dp_shard,), mesh_dim_names=("dp_shard_cp",))

            class _MeshAlias:
                _ALIAS = {"dp": "dp_shard_cp"}
                def __init__(self, m): self._m = m
                def __getitem__(self, key):
                    if isinstance(key, str):
                        key = self._ALIAS.get(key, key)
                    elif isinstance(key, (tuple, list)):
                        key = type(key)(self._ALIAS.get(k, k) for k in key)
                    return self._m[key]
                def __getattr__(self, name): return getattr(self._m, name)

            return _MeshAlias(mesh)

    training.ParallelDims = _ParallelDims

if not hasattr(training, "get_context_parallel_manager"):
    def _get_context_parallel_manager(enabled, rotate_method, world_mesh, model):
        """Stub: no-op context manager (CP not supported in torchtune 0.6.1)."""
        @_contextlib.contextmanager
        def _no_op(batch_values):
            yield batch_values
        return _no_op
    training.get_context_parallel_manager = _get_context_parallel_manager

if not hasattr(training, "get_train_context"):
    def _get_train_context(enable_loss_parallel):
        """Stub: no-op training context (loss parallel not supported in torchtune 0.6.1)."""
        @_contextlib.contextmanager
        def _ctx(cp_cm):
            with cp_cm:
                yield
        return _ctx
    training.get_train_context = _get_train_context
# ─────────────────────────────────────────────────────────────────────────────

log = torchtune_utils.get_logger("DEBUG")


class SelfPredictionTrainingRecipeDistributed(FTRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device
        self._device = torchtune_utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        if (
            cfg.get("fsdp_cpu_offload", False)
            and cfg.optimizer.get("fused", False)
            and not torchtune_utils.torch_version_ge("2.4.0")
        ):
            raise RuntimeError(
                "Using fused optimizer on CPU is only supported in PyTorch nightly."
            )

        if training.is_distributed():
            self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
            self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
            self.distributed_backend = training.get_distributed_backend(
                device_type,
                offload_ops_to_cpu=self.fsdp_cpu_offload
                                   or self._enable_async_checkpointing,
            )
            init_process_group(self.distributed_backend)
        else:
            cfg.nproc_per_node = 1
            self.fsdp_cpu_offload = False

        # Initialize distributed variables
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._world_size, self.rank = training.get_world_size_and_rank()
        else:
            self._world_size, self.rank = 1, 0
        self._is_rank_zero = self.rank == 0
        self.tp_plan = cfg.get("tensor_parallel_plan", None)
        self.tp_degree = cfg.get("tensor_parallel_dim", 1)
        if self.tp_degree > 1 and self.tp_plan is None:
            raise ValueError(
                "Tensor Parallel plan needs to be provided when tensor parallel is enabled."
            )
        if self.tp_degree > 1:
            # DTensor does not support grouped_mm yet
            moe_utils.use_grouped_mm = False
        self.cp_degree = cfg.get("context_parallel_dim", 1)
        self.context_parallel_rotate_method = cfg.get(
            "context_parallel_rotate_method", "allgather"
        )
        data_shard = cfg.get("data_parallel_shard_dim", -1)  # -1 means to infer
        data_replicate = cfg.get("data_parallel_replicate_dim", 1)

        # Set up n-d device mesh
        self.parallel_dims = training.ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=self.tp_degree,
            cp=self.cp_degree,
            world_size=self._world_size,
        )
        if torch.distributed.is_initialized():
            self.world_mesh = self.parallel_dims.build_mesh(device_type=device_type)
        else:
            self.world_mesh = None

        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree, self.dp_rank = (
                dp_mesh.size(),
                dp_mesh.get_local_rank(),
            )
        else:
            self.dp_degree, self.dp_rank = 1, 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = torchtune_utils.get_logger(cfg.log_level)
        if (
                self._log_peak_memory_stats
                and self._device.type not in VALID_BACKENDS_FOR_MEMORY_STATS
        ):
            self._logger.info(
                f"log_peak_memory_stats was set to True; however, training device is not in {VALID_BACKENDS_FOR_MEMORY_STATS}."
                "Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._train_from_scratch = cfg.get("train_from_scratch", False)
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        if self._resume_from_checkpoint and self._train_from_scratch:
            raise ValueError(
                "Both train_from_scratch and resume_from_checkpoint cannot be set to True."
            )
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.get("optimizer_in_bwd", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._checkpoint_client = CheckpointClient(cfg)
        self._enable_fp8_training = cfg.get("enable_fp8_training", False)
        self._fp8_recipe_name = cfg.get("fp8_recipe_name", None)
        self._checkpoint_every_n_steps = cfg.get("checkpoint_every_n_steps", 1000)
        self._overwrite_checkpoints = cfg.get("overwrite_checkpoints", True)

        self._evaluate_every_n_steps = cfg.get("evaluate_every_n_steps", 1000)

        # Early stopping configuration
        self.early_stopping_enabled = cfg.get("early_stopping", False)
        if self.early_stopping_enabled:
            self.patience = cfg.get("patience", 5000)
            self.best_eval_loss = float('inf')
            self.patience_counter = 0
            torchtune_utils.log_rank_zero(
                self._logger,
                f"Early stopping enabled with patience of {self.patience} small_models_evaluation steps."
            )

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if self._optimizer_in_bwd:
            if self._clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd."
                    "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if self._gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        self._activation_offloading_use_streams = cfg.get(
            "activation_offloading_use_streams", True
        )
        if (
                self._enable_activation_offloading
                and self._activation_offloading_use_streams
                and self.parallel_dims.tp_enabled
        ):
            warn(
                message=(
                    "Using activation offloading with streams is not advised in tensor parallel, and may "
                    "cause unstable training. It is advised to set activation_offloading_use_streams: False"
                )
            )
        if self._enable_activation_offloading:
            if device_type != "cuda" and device_type != "xpu":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA or XPU"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
                self._enable_activation_checkpointing
                and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            torchtune_utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        self._save_optimizer_state = cfg.get("save_optimizer_state", True)


        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_total_steps = cfg.max_total_steps
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._train_whole_model = cfg.get("train_whole_model", True)
        self._ignore_main_training_loss = cfg.get("ignore_main_training_loss", False)
        self._ic_generalization_eval = cfg.get("ic_generalization_eval", False)
        self.cfg = cfg

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self.global_step = ckpt_dict[training.STEPS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()


        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            # log config with parameter override
            self._metric_logger.log_config(cfg)

        if self._is_rank_zero:
            # Check if the output_dir in the checkpointer config needs to be resolved
            output_dir = str(cfg.checkpointer.output_dir)
            if "$WANDB_RUN_ID" in output_dir:
                try:
                    # Ensure the logger is a wandb logger and a run has started
                    if hasattr(self._metric_logger, "_wandb") and self._metric_logger._wandb.run:
                        run_id = self._metric_logger._wandb.run.id
                        resolved_output_dir = output_dir.replace("$WANDB_RUN_ID", run_id)

                        # Update the configuration object in place
                        OmegaConf.update(cfg.checkpointer, "output_dir", resolved_output_dir, merge=False)

                        # Also update the model checkpoint dir if it's the same
                        if cfg.checkpointer.checkpoint_dir == output_dir:
                            OmegaConf.update(cfg.checkpointer, "checkpoint_dir", resolved_output_dir, merge=False)

                        log.info(f"Resolved checkpoint output directory to: {cfg.checkpointer.output_dir}")
                    else:
                        warn(
                            "'$WANDB_RUN_ID' placeholder found in output_dir, but WandB logger is not "
                            "initialized or no active run was found. Path remains unresolved."
                        )
                except Exception as e:
                    warn(f"Failed to resolve '$WANDB_RUN_ID': {e}")

        if cfg.train_from_scratch:
            state_dict = None
        else:
            # check if checkpoint exists, else download it
            if not os.path.exists(cfg.checkpointer["checkpoint_dir"]):
                print("Model not found locally. Downloading from Hugging Face...")
                snapshot_download(
                    repo_id=cfg.checkpointer["checkpoint_dir"].replace("checkpoints/", "mistralai/"),
                    local_dir=cfg.checkpointer["checkpoint_dir"],
                    local_dir_use_symlinks=False,
                )
            state_dict = self._checkpoint_client.load_base_checkpoint()

        compile = cfg.get("compile")
        compile_bool = bool(compile)
        self._compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

        self._compile_model = compile_bool
        self._compile_loss = compile_bool
        self._compile_optimizer_step = compile_bool
        self._compile_scale_grads = compile_bool
        if isinstance(compile, DictConfig):
            self._compile_model = compile.get("model", True)
            self._compile_loss = compile.get("loss", True)
            self._compile_optimizer_step = compile.get("optimizer_step", False)
            self._compile_scale_grads = compile.get("scale_grads", True)
        if self._compile_model:
            # Capture scalar outputs is required to compile MoE
            torch._dynamo.config.capture_scalar_outputs = True

        # This indirection is needed to apply torch.compile to scale_grads step.
        self._grad_scaler = training.scale_grads
        if self._compile_scale_grads:
            self._grad_scaler = torch.compile(
                self._grad_scaler, backend=self._compile_backend
            )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        self.use_loss_parallel_ctx_manager = self.parallel_dims.tp_enabled and getattr(
            self._loss_fn,
            "tp_requires_loss_parallel_ctx_manager",
            False,
        )

        self._compile = cfg.get("compile", False)
        if self._world_size == 1:
            self._model = self._setup_model_single_device(
                cfg_model=cfg.model,
                enable_activation_checkpointing=cfg.enable_activation_checkpointing,
                enable_activation_offloading=self._enable_activation_offloading,
                compile_model=self._compile,
                model_state_dict=state_dict[training.MODEL_KEY]
                if state_dict is not None
                else None,
                initialize_mtp_with_last_layer_weights=cfg.get("initialize_mtp_with_last_layer_weights", False),
            )
        else:
            self._model = self._setup_model_distributed(
                cfg_model=cfg.model,
                enable_activation_checkpointing=self._enable_activation_checkpointing,
                enable_activation_offloading=self._enable_activation_offloading,
                activation_offloading_use_streams=self._activation_offloading_use_streams,
                custom_sharded_layers=cfg.get("custom_sharded_layers", None),
                fsdp_cpu_offload=self.fsdp_cpu_offload,
                reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
                model_state_dict=state_dict[training.MODEL_KEY]
                if state_dict is not None
                else None,
                ac_mode=cfg.get("ac_mode", None),
                ac_option=cfg.get("ac_option", None),
                initialize_mtp_with_last_layer_weights=cfg.get("initialize_mtp_with_last_layer_weights", False),
            )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        if hasattr(self._model, "pad_token_id"):
            self._model.pad_token_id = self._tokenizer.pad_id

        if cfg.get("resize_token_embeddings", False):
            resize_token_embeddings(self._model, self._tokenizer.vocab_size)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=self._optimizer_in_bwd,
            opt_state_dict=(
                state_dict[training.OPT_KEY] if state_dict and training.OPT_KEY in state_dict else None
            ),
        )

        if self._sampler is None:
            self._steps_per_epoch = self._estimate_steps_per_epoch()
        else:
            self._steps_per_epoch = (
                len(self._dataloader) // self._gradient_accumulation_steps
            )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        # Setup lr scheduler
        if self._optimizer is not None:
            self._lr_scheduler = self._setup_lr_scheduler(
                cfg_lr_scheduler=cfg.get("lr_scheduler", None),
                num_training_steps=self.max_total_steps,
                last_epoch=self.global_step - 1,
            )
        else:
            self._lr_scheduler = None

        if self._compile_optimizer_step:
            if self._optimizer_in_bwd:
                raise ValueError(
                    "optimizer_in_bwd not supported with compiling the optimizer step"
                )
            self._optimizer.step = torch.compile(
                self._optimizer.step,
                backend=self._compile_backend,
            )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                try:
                    state_dict = self._checkpoint_client.load_distributed_checkpoint(
                        self._model,
                        (
                            self._optim_ckpt_wrapper
                            if self._optimizer_in_bwd
                            else self._optimizer
                        ),
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(state_dict)

        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile_loss:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        torchtune_utils.log_rank_zero(self._logger, "Loss is initialized.")

        self.global_step = self.epochs_run * self._steps_per_epoch

        if self._checkpoint_every_n_steps is None:
            self._checkpoint_every_n_steps = self._steps_per_epoch
            self.checkpoint_dir_prefix = "epoch"
        else:
            self.checkpoint_dir_prefix = "step"

        if (
            self._resume_from_checkpoint
            and self.global_step % self._steps_per_epoch == 0
        ):
            list(self._dataloader)



        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        torchtune_utils.log_rank_zero(
            self._logger, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model_single_device(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        model_state_dict: Dict[str, Any],
        initialize_mtp_with_last_layer_weights: bool = False
    ) -> nn.Module:
        """
        Set up the model including enabling activation checkpointing.
        """
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        self._untrained_parameter_names: List[str] = []
        full_state_dict = model.state_dict()

        # Update the random weights with the loaded checkpoint weights.
        # This keeps the random weights for the new layer but uses pre-trained for the base.
        if model_state_dict:
            full_model_keys = set(full_state_dict.keys())
            pretrained_keys = set(model_state_dict.keys())
            self._untrained_parameter_names = list(full_model_keys - pretrained_keys)

        if not self._train_whole_model:
            log.info(
                f"Found {len(self._untrained_parameter_names)} parameter names "
                "not in the pretrained checkpoint."
            )
            log_param_list = self._untrained_parameter_names[:5]  # Log first 5 for preview
            log.info(
                f"Examples of untrained parameters: {log_param_list}"
                f"{'...' if len(self._untrained_parameter_names) > 5 else ''}"
            )

        if model_state_dict:
            full_state_dict.update(model_state_dict)

        if initialize_mtp_with_last_layer_weights and cfg_model.use_mtp:
            full_state_dict = initialize_mtp_layer_with_last_layer_weights(model_state_dict, full_state_dict)

        model.load_state_dict(full_state_dict, strict=True)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )

        # Enable activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        # remaining context managers for fwd/bwd
        self.train_context = training.get_train_context(
            enable_loss_parallel=self.use_loss_parallel_ctx_manager,
        )

        self.context_parallel_manager = training.get_context_parallel_manager(
            enabled=self.cp_degree > 1,
            rotate_method=self.context_parallel_rotate_method,
            world_mesh=self.world_mesh,
            model=model,
        )

        self._logger.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type == "cuda":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return model

    def _setup_model_distributed(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        activation_offloading_use_streams: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
        initialize_mtp_with_last_layer_weights: bool = False,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        torchtune_utils.log_rank_zero(self._logger, "Instantiating model on meta device...")

        init_start = time.perf_counter()

        #if model_state_dict is None or not self._train_whole_model:
        #    with training.set_default_dtype(self._dtype):
        #        model = config.instantiate(cfg_model)
        #else:
        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        self._untrained_parameter_names: List[str] = []
        if self._is_rank_zero:
            print("Merging pre-trained and random weights on CPU...")
            # Create a temporary model on CPU to get a full dict with random weights
            with training.set_default_dtype(self._dtype):
                random_init_model = config.instantiate(self.cfg.model)

            full_state_dict = random_init_model.state_dict()

            # Update the random weights with the loaded checkpoint weights.
            # This keeps the random weights for the new layer but uses pre-trained for the base.
            if model_state_dict:
                full_model_keys = set(full_state_dict.keys())
                pretrained_keys = set(model_state_dict.keys())
                self._untrained_parameter_names = list(full_model_keys - pretrained_keys)

            for key, value in full_state_dict.items():
                if key not in model.state_dict():
                    untrained_parameters.append(key)

            if not self._train_whole_model:
                log.info(
                    f"Found {len(self._untrained_parameter_names)} parameter names "
                    "not in the pretrained checkpoint."
                )
                log_param_list = self._untrained_parameter_names[:5]  # Log first 5 for preview
                log.info(
                    f"Examples of untrained parameters: {log_param_list}"
                    f"{'...' if len(self._untrained_parameter_names) > 5 else ''}"
                )

            if model_state_dict:
                full_state_dict.update(model_state_dict)

            if initialize_mtp_with_last_layer_weights and cfg_model.use_mtp:
                full_state_dict = initialize_mtp_layer_with_last_layer_weights(model_state_dict, full_state_dict)
        else:
            full_state_dict = None

        # Broadcast the full state dict to all processes
        full_state_dict_list = [full_state_dict]
        torch.distributed.broadcast_object_list(full_state_dict_list, src=0)
        full_state_dict = full_state_dict_list[0]

        # Broadcast the list of untrained parameter names to all processes
        untrained_params_list_to_broadcast = [self._untrained_parameter_names]
        torch.distributed.broadcast_object_list(untrained_params_list_to_broadcast, src=0)
        self._untrained_parameter_names = untrained_params_list_to_broadcast[0]

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        if self._enable_fp8_training:
            # Requires https://github.com/pytorch/pytorch/pull/148922
            if torch.__version__ < "2.8.0.dev20250318":
                raise RuntimeError(
                    "Float8 fine-tuning requires PyTorch 2.8.0.dev20250318 or later."
                )
            if self.cp_degree > 1:
                raise ValueError(
                    "Context Parallel for fp8 training is not currently supported"
                )
            model = convert_to_float8_training(model, self._fp8_recipe_name)

        # Apply tensor parallelism to the model
        if self.parallel_dims.tp_enabled:
            if not self.parallel_dims.dp_enabled and self.fsdp_cpu_offload:
                raise ValueError(
                    "Tensor parallelism is not supported with FSDP CPU offloading when data parallelism is disabled."
                )
            # Use the local number (num_heads, num_kv_heads, embed_dim) to account for tensor parallel
            model = training.prepare_mha_for_tp(model, self.world_mesh["tp"])
            if self.tp_plan is not None:
                self.tp_plan = config.instantiate(
                    self.tp_plan,
                    model=model,
                    enable_fp8_training=self._enable_fp8_training,
                )
                if isinstance(self._loss_fn, SFTLoss):
                    self._loss_fn.tp_enabled = True
                    self.tp_plan = self._loss_fn.patch_tp_plan(self.tp_plan)

            parallelize_module(
                model,
                self.world_mesh["tp"],
                parallelize_plan=self.tp_plan,
            )

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # Apply Fully Sharded Data Parallelism to the model
        if self.parallel_dims.dp_shard_enabled or self.parallel_dims.cp_enabled:
            fsdp_shard_conditions = [
                partial(
                    training.get_shard_conditions,
                    names_to_match=custom_sharded_layers,
                )
            ]

            if self.parallel_dims.dp_replicate_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_dim_names = ("dp_shard_cp",)

            training.shard_model(
                model=model,
                shard_conditions=fsdp_shard_conditions,
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=reshard_after_forward,
                dp_mesh=self.world_mesh[dp_mesh_dim_names],
            )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            full_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading, activation_offloading_use_streams
        )
        # context parallel
        self.context_parallel_manager = training.get_context_parallel_manager(
            enabled=self.cp_degree > 1,
            rotate_method=self.context_parallel_rotate_method,
            world_mesh=self.world_mesh,
            model=model,
        )
        # remaining context managers for fwd/bwd
        self.train_context = training.get_train_context(
            enable_loss_parallel=self.use_loss_parallel_ctx_manager,
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        torchtune_utils.log_rank_zero(
            self._logger,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier(device_ids=[self._device.index])

        return model

    def _setup_optimizer(
            self,
            cfg_optimizer: DictConfig,
            optimizer_in_bwd: bool = False,
            opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if self._train_whole_model:
            torchtune_utils.log_rank_zero(self._logger, "Setting up optimizer for all model parameters.")
            params_to_train = self._model.parameters()
        else:
            torchtune_utils.log_rank_zero(
                self._logger,
                "Freezing base model and setting up optimizer for untrained parameters only."
            )
            params_to_train = []
            untrained_names_set = set(self._untrained_parameter_names)

            for name, param in self._model.named_parameters():
                if name in untrained_names_set:
                    param.requires_grad = True
                    params_to_train.append(param)
                else:
                    param.requires_grad = False

            if not params_to_train:
                warn(
                    "train_whole_model is False, but no untrained parameters were found. "
                    "This means no parameters will be optimized."
                )
                return
            else:
                num_params_to_train = sum(p.numel() for p in params_to_train)
                torchtune_utils.log_rank_zero(
                    self._logger,
                    f"Optimizing {len(params_to_train)} parameter groups "
                    f"({num_params_to_train / 1e6:.2f}M parameters)."
                )

        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                param: config.instantiate(cfg_optimizer, [param])
                for param in params_to_train
            }

            # Register optimizer step hooks on the model to run optimizer in backward.
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            # Load optimizer states for each param. If optimizer states are being restored in an optimizer in
            # backward run, these need to have been saved with the same setting. Cannot restore from runs that
            # did not use optimizer in backward.
            if opt_state_dict is not None:
                for param in opt_state_dict.keys():
                    try:
                        training.load_from_full_optimizer_state_dict(
                            self._model,
                            self._optim_ckpt_wrapper.optim_map[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
            torchtune_utils.log_rank_zero(self._logger, "In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, params_to_train)
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            torchtune_utils.log_rank_zero(self._logger, "Optimizer is initialized.")
            return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        """
        Set up the learning rate scheduler based on the provided configuration.
        It supports both standard optimization and optimizer-in-backward cases.

        Args:
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                self._logger.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        if self._optimizer_in_bwd:
            # Use the first optimizer from the wrapper to represent the learning rate
            optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            # Standard case: use the single optimizer
            optimizer = self._optimizer

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        if self._optimizer_in_bwd:
            # Modify the scheduler for optimizer_in_bwd case
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

        if self._is_rank_zero:
            self._logger.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All dataset_classes related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable dataset_classes and streaming dataset_classes are not supported.
        """
        world_size, rank = training.get_world_size_and_rank()

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            packed = cfg_dataset.get("packed", False)

            packed_on_the_fly = cfg_dataset.pop("packed_on_the_fly", False)
            packed_sequence_length = cfg_dataset.pop("packed_sequence_length", 2048)
            split_across_pack = cfg_dataset.pop("split_across_pack", False)
            num_workers = cfg_dataset.pop("num_workers", 8)

            if packed_on_the_fly and "packed" in cfg_dataset:
                cfg_dataset["packed"] = False
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            if packed_on_the_fly:
                ds = PackedOnTheFlyDataset(
                    ds,
                    max_seq_len=packed_sequence_length,
                    padding_idx=self._tokenizer.pad_id,
                    world_size=world_size,
                    rank=rank,
                    permute_indices=shuffle,
                    split_across_pack=split_across_pack,
                )
                packed = True

        if packed_on_the_fly:
            dataloader = DataLoader(
                dataset=ds,
                batch_size=batch_size,
                num_workers=num_workers,
                worker_init_fn=ds._worker_init_fn,
                # multiprocessing_context="spawn",
                collate_fn=partial(
                    padded_collate_sft,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else partial(
                    dummy_collate,
                    #padded_collate_packed,
                ),
            )
            log.info("On the fly packing & tokenization Dataset is initialized.")
            return None, dataloader
        else:
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
            )
            dataloader = DataLoader(
                dataset=ds,
                batch_size=batch_size,
                sampler=sampler,
                # dropping last avoids shape issues with compile + flex attention
                drop_last=cfg_dataset.get("drop_last", True),
                collate_fn=partial(
                    padded_collate_sft,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else partial(
                    dummy_collate,
                    #padded_collate_packed,
                ),
            )

            if self._is_rank_zero:
                log.info("Dataset and Sampler are initialized.")

            return sampler, dataloader

    def save_checkpoint(
        self,
        epoch: int,
        full_tensors: bool,
        intermediate_checkpoint: bool = True,
    ) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the model weights and recipe state in
        different checkpoint files. To correctly resume training from an intermediate checkpoint,
        the model weights and recipe state must be provided.
        """
        # final dict passed onto the checkpointer

        #if self._checkpointer is None:
        #    if self._is_rank_zero:
        #        log.info("Checkpointer is not initialized. Skipping checkpointing.")
        #    return

        dir_prefix = "step" if not self._overwrite_checkpoints else "epoch"

        self._checkpoint_client.save_checkpoint(
            model=self._model,
            optimizer=(
                self._optimizer
                if not self._optimizer_in_bwd
                else self._optim_ckpt_wrapper
            ),
            training_progress=TrainingProgress(
                seed=self.seed,
                epochs_run=epoch,
                total_epochs=self.total_epochs,
                max_steps_per_epoch=self.max_steps_per_epoch,
                steps_run=self.global_step,
                total_training_steps=self.max_total_steps,
                dataloader_state_dict={},
                val_dataloader_state_dict={},
            ),
            epoch=self.global_step if not self._overwrite_checkpoints else epoch,
            single_device=(self._world_size == 1),
            full_tensors=full_tensors,
            dir_prefix=dir_prefix,
            train_whole_model=self._train_whole_model,
            untrained_parameter_names=self._untrained_parameter_names,
            intermediate_checkpoint=intermediate_checkpoint
        )

        # save also the cfg as a yaml file using OmegaConf
        yaml_path = f"{self.cfg.checkpointer.output_dir}config.yaml"
        with open(yaml_path, "w") as f:
            OmegaConf.save(self.cfg, f.name)

        return


    def _estimate_steps_per_epoch(self, num_batches=5) -> int:
        dataset_length = len(self._dataloader.dataset.ds)
        num_examples = 0
        for i, batch in enumerate(self._dataloader):
            batch = padded_collate_packed(batch)  # this line is only necessary if we use the dummy_collate function in the dataloader
            if i >= num_batches:
                break
            # count the number of times the position sequences go back to 0 (and it's not padding)
            num_zeros = torch.sum(
                torch.logical_and(
                    batch["input_pos"] == 0, batch["tokens"] != self._tokenizer.pad_id
                )
            ).item()

            num_examples += num_zeros
        examples_per_batch = num_examples / num_batches
        steps_per_epoch = int(
            dataset_length / (examples_per_batch * self._gradient_accumulation_steps)
        )
        return steps_per_epoch

    def _loss_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")

        with self.activations_handling_ctx:
            outputs = self._model(**batch)

        # if model has a mtp_layer, it returns a tuple of logits and shortcut_logits
        #if hasattr(self._model, "mtp_layer"):
        #    outputs, shortcut_outputs = outputs
        #else:
        #    shortcut_outputs = None

        # post process for third party loss functions
        if not isinstance(self._loss_fn, SFTLoss):
            labels = labels.reshape(-1)
            outputs = outputs.reshape(-1, outputs.size(-1))
            if isinstance(outputs, DTensor):
                outputs = outputs.full_tensor()

        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we don't need to slice the logits. We just add an ignore index to labels.
        #labels = torch.hstack(
        #    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        #)

        loss = self._loss_fn(outputs, labels)
        if isinstance(loss, tuple):
            loss, additional_logging_losses = loss
        else:
            additional_logging_losses = {}

        assert loss.numel() == 1, f"Loss should be a scalar. Use a loss function that aggregates the loss!"
        # free logits otherwise it peaks backward memory
        del outputs, labels

        return loss, additional_logging_losses

        additional_logging_losses = {}
        next_token_prediction_loss = loss

        if shortcut_logits is not None:
            shortcut_loss = self._loss_fn(shortcut_logits, labels)
            assert shortcut_loss.numel() == 1, f"Shortcut loss should be a scalar. Use a loss function that aggregates the loss!"
            del shortcut_logits

            additional_logging_losses["shortcut_loss"] = shortcut_loss.detach()
            additional_logging_losses["next_token_prediction_loss"] = next_token_prediction_loss.detach()
            shortcut_loss_factor = self._model.mtp_loss_factor
            loss = next_token_prediction_loss + shortcut_loss_factor * shortcut_loss
        else:
            shortcut_loss = None

        if callable(getattr(self._model, "get_additional_losses", None)):
            if self._ignore_main_training_loss:
                loss = 0.0

            additional_training_losses, additional_logging_losses = self._model.get_additional_losses()
            for loss_name, loss_val in additional_training_losses.items():
                loss += loss_val
            additional_logging_losses[
                "next token prediction losses"
            ] = next_token_prediction_loss.detach()

        return loss, additional_logging_losses

    def train(self, save_at_the_end=True) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        world_size, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0
        cumulative_tokens = 0
        meter = MultiMeter()

        stop_training_flag = False

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            inner_step_count = self.global_step % self._steps_per_epoch
            pbar = tqdm(
                initial=inner_step_count,
                total=self._steps_per_epoch,
                desc=f"{self.epochs_run}|{self.global_step}",
            )

            # Get iterator for the dataloader
            if self._sampler is not None:
                self._sampler.set_epoch(curr_epoch)
            dataloader_iter = iter(self._dataloader)
            batch_count = 0

            # Update the sampler to ensure dataset_classes is correctly shuffled across epochs
            # in case shuffle is True
            #if self._sampler is not None:
            #    self._sampler.set_epoch(curr_epoch)

            while inner_step_count < self._steps_per_epoch:
                # Try to get the next batch, break if we've reached the end of the dataset
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break

                batch = padded_collate_packed(batch)

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and batch_count
                        == self.profiler_wait_steps + self.profiler_warmup_steps
                        and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                # modified loss step
                torchtune_utils.batch_to_device(batch, self._device)

                total_tokens_in_batch = batch["tokens"].numel()
                cumulative_tokens += (total_tokens_in_batch * world_size)

                with self.train_context(
                        self.context_parallel_manager(list(batch.values()))
                ):
                    current_num_tokens = (
                            batch["labels"] != self._loss_fn.ignore_index
                    ).sum()
                    num_tokens += current_num_tokens

                    loss, sub_losses_dict = self._loss_step(batch)
                    current_loss = loss * current_num_tokens
                    running_loss += current_loss
                    # For optimizer in backward, we need to normalize before calling backward
                    # This case and gradient accumulation are mutually exclusive
                    if self._optimizer_in_bwd:
                        torch.distributed.all_reduce(num_tokens)
                        torch.distributed.all_reduce(running_loss)
                        current_loss = current_loss * (self.dp_degree / num_tokens)
                    current_loss.backward()

                sub_losses_dict = {k: v.item() for k, v in sub_losses_dict.items()}

                # join the loss with the sub_losses_dict
                meter.update(sub_losses_dict)

                # Step with optimizer
                if (batch_count + 1) % self._gradient_accumulation_steps == 0:
                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(num_tokens)
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_loss)

                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        self._grad_scaler(
                            list(self._model.parameters()),
                            self._world_size / num_tokens,
                            False if self.parallel_dims.tp_enabled else None,
                        )

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                            # If sharded, collect the DTensor here
                            if isinstance(grad_norm, DTensor):
                                grad_norm = grad_norm.full_tensor()
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1
                    inner_step_count += 1

                    # If float8 training is enabled, perform a single all-reduce to compute the
                    # scale for all float8 parameters efficiently instead of doing many small
                    # all-reduces for each parameter
                    if (
                            self._enable_fp8_training
                            and is_fp8_tensorwise_scaling(self._fp8_recipe_name)
                            and self.dp_degree > 1
                    ):
                        precompute_float8_dynamic_scale_for_fsdp(self._model)

                    loss_to_log = running_loss.detach().item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens /
                                                         (self.parallel_dims.non_data_parallel_size * time_per_step),
                            "tokens": cumulative_tokens,
                            "epoch": curr_epoch,
                        }
                        for key in meter.meters.keys():
                            log_dict[key] = meter.meters[key].avg
                        meter.reset()

                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    if self.global_step % self._evaluate_every_n_steps == 0:
                        print("Evaluating")
                        if 'learning_levels_pfa' in self.cfg.dataset._component_:
                            plotly_figure_dict, eval_values_dict = pfa_training_evaluation(
                                self,
                                num_datapoints=500,
                                ic_generalization_evaluation=self._ic_generalization_eval
                            )
                        elif 'discovery' in self.cfg.dataset._component_ or 'construction' in self.cfg.dataset._component_:
                            plotly_figure_dict, eval_values_dict = creativity_evaluation(self)
                        else:
                            raise ValueError("Unknown small_models_evaluation dataset component for self-prediction training recipe.")

                        log_dict = plotly_figure_dict
                        for key, value in eval_values_dict.items():
                            log_dict[key] = value

                        if self._is_rank_zero:
                            self._metric_logger.log_dict(
                                log_dict,
                                step=self.global_step,
                            )

                        # Early Stopping Logic
                        current_eval_loss = eval_values_dict.get('eval_loss', None)
                        if current_eval_loss is None:
                            current_eval_loss = eval_values_dict.get('eval_ce_loss', None)
                        if self.early_stopping_enabled and current_eval_loss is not None:
                            if current_eval_loss < self.best_eval_loss:
                                self.best_eval_loss = current_eval_loss
                                self.patience_counter = 0
                                torchtune_utils.log_rank_zero(
                                    self._logger,
                                    f"New best validation loss: {self.best_eval_loss:.4f}. Saving checkpoint."
                                )
                                # Save the best model
                                self.save_checkpoint(
                                    epoch=self.global_step
                                        if not self._overwrite_checkpoints
                                        else None,
                                    full_tensors=False,
                                    intermediate_checkpoint=self._save_optimizer_state,
                                )
                            else:
                                self.patience_counter += 1
                                torchtune_utils.log_rank_zero(
                                    self._logger,
                                    f"Validation loss did not improve. Best: {self.best_eval_loss:.4f}. "
                                    f"Patience: {self.patience_counter}/{self.patience}"
                                )

                            if self.patience_counter >= self.patience:
                                torchtune_utils.log_rank_zero(
                                    self._logger,
                                    f"Early stopping triggered after {self.patience} steps without improvement."
                                )
                                stop_training_flag = True

                    # Only do periodic checkpointing if early stopping is not enabled.
                    # If it is enabled, checkpoints are saved only on validation loss improvement.
                    if not self.early_stopping_enabled and self.global_step % self._checkpoint_every_n_steps == 0:
                        self.save_checkpoint(
                            epoch=self.global_step
                            if not self._overwrite_checkpoints
                            else None,
                            full_tensors=False,
                            intermediate_checkpoint=self._save_optimizer_state,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                    if stop_training_flag:
                        break

                    if self.global_step >= self.max_total_steps:
                        break

            if self.epochs_run == 0:
                self._steps_per_epoch = self.global_step

            self.epochs_run += 1

            if self.global_step >= self.max_total_steps:
                break

        self._profiler.stop()

        if save_at_the_end and self.early_stopping_enabled:
            self.save_checkpoint(
                epoch=self.global_step if not self._overwrite_checkpoints else None,
                intermediate_checkpoint=False,
                full_tensors=False,
            )

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        if self._world_size > 1:
            destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    print("start recipe_main")

    config.log_config(recipe_name="SelfPredictionTraining", cfg=cfg)
    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

