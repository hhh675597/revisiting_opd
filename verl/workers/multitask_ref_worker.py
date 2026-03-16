"""Multi-task reference policy worker that routes to task-specific ref models."""

import os
from typing import Dict

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict

from verl.protocol import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device, is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    offload_fsdp_model_to_cpu,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import update_model_config
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_dtypes import PrecisionType
from verl.utils import hf_tokenizer, hf_processor
from verl.workers.fsdp_workers import FSDPUlyssesShardingManager
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy

class MultiTaskRefWorker(Worker):
    """
    Multi-task reference policy worker that maintains separate ref models for each task
    and routes compute_ref_log_prob calls based on task_type in the batch.
    
    This worker is designed for sequential batching where each batch contains samples
    from only one task type, making routing straightforward.
    """

    def __init__(self, config: DictConfig, role: str = "ref"):
        super().__init__()
        self.config = config
        self.role = role
        
        import torch.distributed
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl",
                rank=rank,
                world_size=world_size
            )

        # Build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(
            world_size=self.world_size,
            fsdp_size=self.config.actor.fsdp_config.fsdp_size
        )

        # Build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        if self.ulysses_sequence_parallel_size > 1:
            from torch.distributed.device_mesh import init_device_mesh
            device_name = "cuda" if is_cuda_available else "cpu"
            dp = self.world_size // self.ulysses_sequence_parallel_size
            self.ulysses_device_mesh = init_device_mesh(
                device_name,
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # Initialize tokenizer and processor (needed for model config)
        # Use the base model path for tokenizer
        base_model_path = self.config.model.path
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(base_model_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(base_model_path, trust_remote_code=trust_remote_code)
        
        # Storage for task-specific ref models
        self.task_ref_models: Dict[str, any] = {}
        self.task_configs: Dict[str, DictConfig] = {}
        
        # Extract task configurations
        if OmegaConf.select(config, 'multitask.enable') and config.multitask.get('enable', False):
            tasks = config.multitask.get('tasks', {})
            for task_key, task_cfg in tasks.items():
                task_name = task_cfg.get('name')
                if task_name:
                    self.task_configs[task_name] = task_cfg
            
            print(f"[MultiTaskRefWorker] Initialized for tasks: {list(self.task_configs.keys())}")
        else:
            raise ValueError("MultiTaskRefWorker requires multitask.enable=true and task configurations")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize all task-specific reference models."""
        from verl.workers.actor import DataParallelPPOActor
        
        print(f"[MultiTaskRefWorker] Loading {len(self.task_configs)} task-specific ref models...")
        
        for task_name, task_cfg in self.task_configs.items():
            # Get ref model path for this task
            ref_model_path = task_cfg.get('ref_model_path')
            if not ref_model_path:
                # Fall back to global ref model path if not specified
                ref_model_path = self.config.ref.get("model", {}).get("path", None) or self.config.model.path
                print(f"[MultiTaskRefWorker] Warning: No ref_model_path specified for task '{task_name}', using global path: {ref_model_path}")
            else:
                print(f"[MultiTaskRefWorker] Loading ref model for task '{task_name}' from {ref_model_path}")
            
            if not ref_model_path:
                raise ValueError(
                    f"No ref_model_path found for task '{task_name}'. "
                    f"Please set multitask.tasks.{task_name}.ref_model_path or actor_rollout_ref.ref.model.path"
                )
            
            # Copy model to local
            use_shm = self.config.model.get("use_shm", False)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            
            # Get trust_remote_code setting
            ref_trust_remote_code = self.config.ref.get("model", {}).get(
                "trust_remote_code",
                self.config.model.get("trust_remote_code", False)
            )
            
            # Build the ref model for this task
            ref_module_fsdp = self._build_ref_model(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                trust_remote_code=ref_trust_remote_code,
            )
            
            # Wrap in DataParallelPPOActor
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = self.config.model.get("use_remove_padding", True)
                self.config.ref.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
            
            ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=ref_module_fsdp
            )
            
            self.task_ref_models[task_name] = ref_policy
            
            print(f"[MultiTaskRefWorker] Successfully loaded ref model for task '{task_name}'")
        
        print(f"[MultiTaskRefWorker] All {len(self.task_ref_models)} ref models loaded successfully")

    def _build_ref_model(self, model_path, fsdp_config, trust_remote_code):
        """Build a single reference model with FSDP, following fsdp_workers.py logic."""
        from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, MixedPrecision
        from transformers import AutoConfig, AutoModelForCausalLM
        from verl.utils.model import print_model_size
        
        # Import external libraries if needed
        external_lib = self.config.model.get("external_lib", None)
        if external_lib is not None:
            import_external_libs(external_lib)
        
        # Determine torch dtype - ref models typically use bfloat16
        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16  # Ref models use bf16 by default
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        # Load model config
        actor_model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2"
        )
        
        # Override model config with tokenizer settings
        override_model_config = self.config.model.get("override_config", {})
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        
        print(f"[MultiTaskRefWorker] Model config after override: {actor_model_config}")
        
        # Load model
        actor_module = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=actor_model_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        
        print_model_size(actor_module)
        
        # FSDP wrapping
        param_dtype = torch_dtype
        reduce_dtype = torch_dtype
        buffer_dtype = torch_dtype
        
        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype
        )
        
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=False
        )
        
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        
        # Force CPUOffload for ref models to save memory
        cpu_offload = CPUOffload(offload_params=True)
        
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
            from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict
            
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True
            )
            cpu_offload_policy = CPUOffloadPolicy(pin_memory=True)
            
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload_policy)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"Strategy {fsdp_strategy} not implemented")
        
        return actor_module_fsdp

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        """
        Compute reference log probabilities by routing to the appropriate task-specific ref model.
        
        Assumes sequential batching where all samples in the batch are from the same task.
        """
        # Extract task_type from batch
        if 'task_type' not in data.non_tensor_batch:
            raise ValueError("task_type not found in batch. Ensure MultiTaskRLHFDataset is used.")
        
        task_types = data.non_tensor_batch['task_type']
        
        # Verify all samples are from the same task (sequential batching guarantee)
        unique_tasks = set(task_types)
        if len(unique_tasks) != 1:
            raise ValueError(
                f"Mixed task batch detected: {unique_tasks}. "
                f"MultiTaskRefWorker requires sequential batching (one task per batch)."
            )
        
        task_name = list(unique_tasks)[0]
        
        if task_name not in self.task_ref_models:
            raise ValueError(
                f"No ref model found for task '{task_name}'. "
                f"Available tasks: {list(self.task_ref_models.keys())}"
            )
        
        # Route to the appropriate ref model
        ref_policy = self.task_ref_models[task_name]
        
        # Move data to device
        data = data.to(get_torch_device().current_device())
        
        # Configure computation parameters
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        
        # Compute ref log prob using task-specific model
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, _ = ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        
        output = output.to("cpu")
        
        # Handle FSDP resharding
        from verl.utils.fsdp_utils import fsdp_version
        if self.world_size > 1 and fsdp_version(ref_policy.actor_module) == 1:
            ref_policy.actor_module._handle.reshard(True)
        
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob_with_topk(self, data: DataProto):
        """
        Compute reference log probabilities along with top-k token indices and their log probs.
        
        Used for OPD with true KL divergence calculation.
        topk_k should be passed via data.meta_info["topk_k"]
        """
        # Get topk_k from meta_info
        topk_k = data.meta_info.get("topk_k", 50)
        
        # Extract task_type from batch
        if 'task_type' not in data.non_tensor_batch:
            raise ValueError("task_type not found in batch. Ensure MultiTaskRLHFDataset is used.")
        
        task_types = data.non_tensor_batch['task_type']
        
        # Verify all samples are from the same task
        unique_tasks = set(task_types)
        if len(unique_tasks) != 1:
            raise ValueError(
                f"Mixed task batch detected: {unique_tasks}. "
                f"MultiTaskRefWorker requires sequential batching (one task per batch)."
            )
        
        task_name = list(unique_tasks)[0]
        
        if task_name not in self.task_ref_models:
            raise ValueError(
                f"No ref model found for task '{task_name}'. "
                f"Available tasks: {list(self.task_ref_models.keys())}"
            )
        
        # Route to the appropriate ref model
        ref_policy = self.task_ref_models[task_name]
        
        # Move data to device
        data = data.to(get_torch_device().current_device())
        
        # Configure computation parameters
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        
        # Compute ref log prob with top-k using task-specific model
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            result = ref_policy.compute_log_prob_with_topk(data=data, topk_k=topk_k)
            output = DataProto.from_dict(tensors={
                "ref_log_prob": result["log_probs"],
                "teacher_topk_indices": result["topk_indices"],
                "teacher_topk_log_probs": result["topk_log_probs"],
            })
            output = self.ulysses_sharding_manager.postprocess_data(output)
        
        output = output.to("cpu")
        
        # Handle FSDP resharding
        from verl.utils.fsdp_utils import fsdp_version
        if self.world_size > 1 and fsdp_version(ref_policy.actor_module) == 1:
            ref_policy.actor_module._handle.reshard(True)
        
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob_with_logits(self, data: DataProto):
        """
        Compute reference log probabilities and extract logits for memory-efficient KL.
        
        Returns ref_log_prob, ref_logits_k, ref_logsumexp, and ref_topk_indices
        following the ema-pg pattern for compute_memory_efficient_kl.
        
        kl_topk_k should be passed via data.meta_info["kl_topk_k"]
        """
        # Get kl_topk_k from meta_info: -1 for full logits, >0 for top-k
        kl_topk_k = data.meta_info.get("kl_topk_k", 50)
        
        # Extract task_type from batch
        if 'task_type' not in data.non_tensor_batch:
            raise ValueError("task_type not found in batch. Ensure MultiTaskRLHFDataset is used.")
        
        task_types = data.non_tensor_batch['task_type']
        
        # Verify all samples are from the same task
        unique_tasks = set(task_types)
        if len(unique_tasks) != 1:
            raise ValueError(
                f"Mixed task batch detected: {unique_tasks}. "
                f"MultiTaskRefWorker requires sequential batching (one task per batch)."
            )
        
        task_name = list(unique_tasks)[0]
        
        if task_name not in self.task_ref_models:
            raise ValueError(
                f"No ref model found for task '{task_name}'. "
                f"Available tasks: {list(self.task_ref_models.keys())}"
            )
        
        # Route to the appropriate ref model
        ref_policy = self.task_ref_models[task_name]
        
        # Move data to device
        data = data.to(get_torch_device().current_device())
        
        # Configure computation parameters
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        
        # Compute ref log prob with logits using task-specific model
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            ref_log_prob, _, ref_kl_inputs = ref_policy.compute_log_prob_with_logits(
                data=data, kl_topk_k=kl_topk_k
            )
            output_tensors = {
                "ref_log_prob": ref_log_prob,
                "ref_logits_k": ref_kl_inputs["logits_k"],
                "ref_logsumexp": ref_kl_inputs["logsumexp"],
            }
            if ref_kl_inputs["topk_indices"] is not None:
                output_tensors["ref_topk_indices"] = ref_kl_inputs["topk_indices"]
            output = DataProto.from_dict(tensors=output_tensors)
            output = self.ulysses_sharding_manager.postprocess_data(output)
        
        output = output.to("cpu")
        
        # Handle FSDP resharding
        from verl.utils.fsdp_utils import fsdp_version
        if self.world_size > 1 and fsdp_version(ref_policy.actor_module) == 1:
            ref_policy.actor_module._handle.reshard(True)
        
        return output
