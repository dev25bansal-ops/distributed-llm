"""Tensor parallelism worker and launcher for multi-GPU inference within a single node.

Uses PyTorch's tensor.parallel (distributed._tensor) for actual tensor slicing:
- Attention heads are split across GPUs
- MLP intermediate dims are split across GPUs
- All-reduce aggregates partial results
- No layer-serial bottleneck (vs pipeline parallelism)
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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


@dataclass
class TPWorkerHandle:
    """Runtime handle for local tensor-parallel worker servers."""
    process_context: object
    ports: list[int]
    world_size: int


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

        def ForwardPassAsync(self, request, context):
            """Async version with comm-compute overlap using AsyncTensorParallel."""
            try:
                from distllm.core.async_tp import AsyncTensorParallel

                async_tp = AsyncTensorParallel(tp_group=dist.group.WORLD, async_op=True)

                if request.input_ids:
                    input_ids = torch.tensor([list(request.input_ids)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        # Run model with async overlap
                        prev_output = None
                        for layer_module in self._get_layer_modules():
                            hidden = layer_module(input_ids if prev_output is None else prev_output)
                            prev_output = async_tp.forward_overlap(layer_module, input_ids if prev_output is None else prev_output, prev_output)
                        async_tp.synchronize()
                        logits = prev_output
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
                logger.error(f"TP rank {rank} ForwardPassAsync failed: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return ForwardPassResponse(success=False)

        def _get_layer_modules(self):
            """Extract individual layer modules from the model for async overlap."""
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return model.model.layers
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                return model.transformer.h
            return []

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

    context = mp.spawn(
        _tp_worker_entry,
        args=(num_gpus, model_name, dtype, trust_remote_code, port),
        nprocs=num_gpus,
        join=False,
    )
    return TPWorkerHandle(
        process_context=context,
        ports=[port + rank + 1 for rank in range(num_gpus)],
        world_size=num_gpus,
    )


def tp_forward(input_tensor: torch.Tensor, tp_handles: list[TPWorkerHandle], timeout: float = 30.0) -> torch.Tensor:
    """Fan out one forward request to all local TP ranks and return reduced logits."""
    if not tp_handles:
        raise RuntimeError("No tensor-parallel workers are running")

    from distllm.communication.grpc_client import NodeClient
    from distllm.communication.node_pb2 import ForwardPassRequest
    from distllm.communication.serializers import proto_to_tensor, tensor_to_proto

    handle = tp_handles[0]

    def _call_rank(port: int):
        request = ForwardPassRequest(request_id=f"tp_{port}", use_cache=False)
        if input_tensor.dtype in (torch.int32, torch.int64, torch.long):
            request.input_ids.extend([int(x) for x in input_tensor.flatten().tolist()])
            request.batch_size = int(input_tensor.shape[0]) if input_tensor.dim() > 1 else 1
            request.seq_len = int(input_tensor.shape[-1]) if input_tensor.dim() > 1 else int(input_tensor.numel())
        else:
            request.hidden_states.CopyFrom(tensor_to_proto(input_tensor))
        with NodeClient("127.0.0.1", port, use_tls=False) as client:
            return client.forward(request, timeout=timeout)

    with ThreadPoolExecutor(max_workers=handle.world_size) as executor:
        responses = list(executor.map(_call_rank, handle.ports))

    first = responses[0]
    if not first.success:
        raise RuntimeError(first.error_message or "tensor-parallel worker failed")
    return proto_to_tensor(first.output)


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
