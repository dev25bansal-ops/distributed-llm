"""Tensor parallelism worker and launcher for multi-GPU inference within a single node.

Uses PyTorch's tensor.parallel (distributed._tensor) for actual tensor slicing:
- Attention heads are split across GPUs
- MLP intermediate dims are split across GPUs
- All-reduce aggregates partial results
- No layer-serial bottleneck (vs pipeline parallelism)
"""

import os

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed._tensor import DeviceMesh
try:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        PairwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
except ImportError:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    PairwiseParallel = None


def _start_tp_server(rank: int, world_size: int, model, tokenizer, device: str, port: int) -> None:
    """Start gRPC server for this TP worker with NCCL all-reduce coordination.

    Each TP worker handles a subset of attention heads and MLP neurons.
    During inference, all-reduce aggregates outputs across workers.
    """
    from concurrent import futures

    import grpc

    from distllm.communication.node_pb2 import ForwardPassResponse
    from distllm.communication.node_pb2_grpc import (
        NodeServiceServicer,
        add_NodeServiceServicer_to_server,
    )
    from distllm.communication.serializers import (
        proto_to_tensor,
        tensor_to_proto,
    )

    class TPServicer(NodeServiceServicer):
        def ForwardPass(self, request, context):
            try:
                if request.input_ids:
                    input_ids = torch.tensor([list(request.input_ids)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        output = model(input_ids, use_cache=request.use_cache)
                        logits = output.logits
                        # All-reduce across TP group to aggregate sharded outputs
                        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
                elif request.HasField('hidden_states'):
                    hidden = proto_to_tensor(request.hidden_states, device)
                    with torch.no_grad():
                        output = model(inputs_embeds=hidden, use_cache=request.use_cache)
                        logits = output.logits
                        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    return ForwardPassResponse(success=False)

                response = ForwardPassResponse(success=True)
                response.output.CopyFrom(tensor_to_proto(logits))
                return response
            except Exception as e:
                logger.error(f"TP rank {rank} ForwardPass failed: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return ForwardPassResponse(success=False)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    add_NodeServiceServicer_to_server(TPServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"TP worker rank {rank} serving on port {port}")
    server.wait_for_termination()


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
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    logger.info(f"TP worker rank {rank}/{world_size} on {device}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        trust = trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        # Load model on meta device first for fast initialization
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust,
                low_cpu_mem_usage=True,
            )

        # Create a device mesh for tensor parallelism
        mesh = DeviceMesh("cuda", list(range(world_size)))

        # Parallelize the transformer layers using PyTorch's tensor.parallel
        # This splits attention heads and MLP dimensions across GPUs
        if PairwiseParallel is None:
            raise ImportError(
                "PairwiseParallel is not available. This version of PyTorch does not support "
                "torch.distributed.tensor.parallel.PairwiseParallel. "
                "Install PyTorch >= 2.1.0 or use a different parallelism strategy."
            )
        model = parallelize_module(
            model,
            mesh,
            {
                "model.layers": PairwiseParallel(),
                "model.embed_tokens": RowwiseParallel(),
                "model.norm": RowwiseParallel(),
                "lm_head": ColwiseParallel(),
            },
        )

        model.to(device)
        model.eval()
        logger.info(f"Rank {rank}: TP model loaded and parallelized")

        # Start gRPC server for inference requests with NCCL all-reduce coordination
        _start_tp_server(rank, world_size, model, tokenizer, device, port + rank + 1)

    except Exception as e:
        logger.error(f"TP worker rank {rank} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
