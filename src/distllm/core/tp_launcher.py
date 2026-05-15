"""Tensor parallelism worker and launcher for multi-GPU inference within a single node."""

import os
import torch
import torch.distributed as dist
from loguru import logger
from typing import Optional


def _tp_worker_entry(
    rank: int,
    world_size: int,
    model_name: str,
    dtype: str = "float16",
    trust_remote_code: bool = False,
    port: int = 29500,
    master_addr: str = "localhost",
):
    """Entry point for a single tensor parallel worker process."""
    # Initialize process group
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    logger.info(f"TP worker rank {rank}/{world_size} on {device}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from distllm.models.partitioner import DTYPE_MAP, _get_base_prefix

        trust = trust_remote_code
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        total_layers = config.num_hidden_layers
        torch_dtype = DTYPE_MAP.get(dtype, torch.float16)

        # Simple layer sharding: each GPU gets a contiguous chunk of layers
        layers_per_gpu = total_layers // world_size
        remainder = total_layers % world_size
        start_layer = rank * layers_per_gpu + min(rank, remainder)
        end_layer = start_layer + layers_per_gpu - 1
        if rank < remainder:
            end_layer += 1

        logger.info(f"Rank {rank}: loading layers {start_layer}-{end_layer}")

        model_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust,
            "low_cpu_mem_usage": True,
        }

        # For TP v1: use device_map to place different layers on different GPUs
        device_map = {}
        temp_model = AutoModelForCausalLM.from_pretrained(
            model_name, **{**model_kwargs, "device_map": "meta"}
        )
        base_prefix = _get_base_prefix(temp_model)
        temp_base = getattr(temp_model, base_prefix, None) if base_prefix else temp_model

        if rank == 0:
            for attr in ['embed_tokens', 'wte', 'word_embeddings']:
                if hasattr(temp_base, attr):
                    device_map[f"{base_prefix}.{attr}"] = device
                    break

        for attr in ['layers', 'block', 'h']:
            if hasattr(temp_base, attr):
                layers_name = attr
                break

        for i in range(total_layers):
            layer_device = device if start_layer <= i <= end_layer else "cpu"
            device_map[f"{base_prefix}.{layers_name}.{i}"] = layer_device

        if rank == world_size - 1:
            for attr in ['norm', 'final_layer_norm', 'ln_f']:
                if hasattr(temp_base, attr):
                    device_map[f"{base_prefix}.{attr}"] = device
                    break
            device_map["lm_head"] = device

        model = AutoModelForCausalLM.from_pretrained(model_name, **{**model_kwargs, "device_map": device_map})
        model.eval()

        logger.info(f"Rank {rank} loaded model with layers {start_layer}-{end_layer} on {device}")

        # Keep the process alive and ready for inference requests
        # In a full implementation, this would connect to a coordinator
        # For now, we just log and wait
        import time
        while True:
            time.sleep(1)

    except Exception as e:
        logger.error(f"TP worker rank {rank} failed: {e}")
        raise
    finally:
        dist.destroy_process_group()


def launch_tp_workers(
    model_name: str,
    num_gpus: int = 2,
    dtype: str = "float16",
    trust_remote_code: bool = False,
    port: int = 29500,
):
    """Spawn tensor parallel workers using torch.multiprocessing."""
    import torch.multiprocessing as mp

    logger.info(f"Launching {num_gpus} tensor parallel workers for {model_name}")

    mp.spawn(
        _tp_worker_entry,
        args=(num_gpus, model_name, dtype, trust_remote_code, port),
        nprocs=num_gpus,
        join=True,
    )


def main():
    """CLI entry point for tensor parallel inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Tensor parallel inference launcher")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--port", type=int, default=29500, help="Master port for NCCL")

    args = parser.parse_args()

    launch_tp_workers(
        model_name=args.model,
        num_gpus=args.num_gpus,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        port=args.port,
    )


if __name__ == "__main__":
    main()
