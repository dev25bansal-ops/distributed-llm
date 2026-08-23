"""Tensor parallelism worker and launcher for multi-GPU inference within a single node."""

from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import torch.distributed as dist
from loguru import logger

from distllm.security import hf_revision
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
    process_context: object
    ports: list[int]
    world_size: int

    def wait_until_ready(self, timeout_s: float = 60.0, interval_s: float = 1.0) -> bool:
        import socket
        import time

        if not self.ports:
            return True

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            all_ready = True
            for port in self.ports:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=2):
                        pass
                except (ConnectionRefusedError, OSError):
                    all_ready = False
                    break
            if all_ready:
                return True
            time.sleep(interval_s)
        return False


def _start_tp_server(rank: int, world_size: int, model, tokenizer, device: str, port: int) -> None:
    from concurrent import futures

    import grpc

    from distllm.dist import node_pb2
    try:
        from distllm.communication.node_pb2_grpc import (
            NodeServiceServicer,
            add_NodeServiceServicer_to_server,
        )
    except ImportError:
        NodeServiceServicer = object
        add_NodeServiceServicer_to_server = lambda servicer, server: None

    def _tp_tensor_to_proto(tensor):
        if tensor is None:
            return node_pb2.TensorProto(shape=[], dtype="none", raw_data=b"")
        t = tensor.detach()
        if t.is_cuda:
            t = t.to('cpu', non_blocking=True)
            torch.cuda.current_stream().synchronize()
        dtype_str = str(t.dtype)
        if t.dim() == 0:
            t = t.reshape(1)
        raw = bytes(memoryview(t.contiguous().view(torch.uint8).numpy(force=True)))
        return node_pb2.TensorProto(raw_data=raw, shape=list(tensor.shape), dtype=dtype_str)

    def _tp_proto_to_tensor(proto, device="cpu"):
        if not proto.shape:
            return torch.empty(0, device=device)
        import numpy as np
        dtype_map = {"torch.float32": torch.float32, "torch.float16": torch.float16,
                     "torch.bfloat16": torch.bfloat16, "torch.int64": torch.int64,
                     "torch.int32": torch.int32, "torch.uint8": torch.uint8,
                     "torch.bool": torch.bool, "float32": torch.float32,
                     "float16": torch.float16, "bfloat16": torch.bfloat16,
                     "int64": torch.int64, "int32": torch.int32, "bool": torch.bool}
        tdtype = dtype_map.get(proto.dtype, torch.float32)
        if proto.raw_data:
            arr = np.frombuffer(proto.raw_data, dtype=np.uint8)
            tensor = torch.from_numpy(arr).view(tdtype).reshape(list(proto.shape)).clone()
        else:
            tensor = torch.tensor(list(proto.data), dtype=tdtype).reshape(list(proto.shape))
        return tensor.to(device)

    class TPServicer(NodeServiceServicer):
        def ForwardPass(self, request, context):
            try:
                if request.input_ids:
                    input_ids = torch.tensor([list(request.input_ids)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        output = model(input_ids, use_cache=request.use_cache)
                        logits = output.logits
                        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
                elif request.hidden_states is not None:
                    hidden = _tp_proto_to_tensor(request.hidden_states, device)
                    with torch.no_grad():
                        output = model(inputs_embeds=hidden, use_cache=request.use_cache)
                        logits = output.logits
                        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    return node_pb2.ForwardPassResponse(success=False)

                response = node_pb2.ForwardPassResponse(success=True)
                response.output.CopyFrom(_tp_tensor_to_proto(logits))
                return response
            except Exception as e:
                logger.error(f"TP rank {rank} ForwardPass failed: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return node_pb2.ForwardPassResponse(success=False)

        def ForwardPassAsync(self, request, context):
            try:
                from distllm.core.async_tp import AsyncTensorParallel

                async_tp = AsyncTensorParallel(tp_group=dist.group.WORLD, async_op=True)

                if request.input_ids:
                    input_ids = torch.tensor([list(request.input_ids)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        prev_output = None
                        for layer_module in self._get_layer_modules():
                            hidden = layer_module(input_ids if prev_output is None else prev_output)
                            prev_output = async_tp.forward_overlap(layer_module, input_ids if prev_output is None else prev_output, prev_output)
                        async_tp.synchronize()
                        logits = prev_output
                elif request.hidden_states is not None:
                    hidden = _tp_proto_to_tensor(request.hidden_states, device)
                    with torch.no_grad():
                        output = model(inputs_embeds=hidden, use_cache=request.use_cache)
                        logits = output.logits
                        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    return node_pb2.ForwardPassResponse(success=False)

                response = node_pb2.ForwardPassResponse(success=True)
                response.output.CopyFrom(_tp_tensor_to_proto(logits))
                return response
            except Exception as e:
                logger.error(f"TP rank {rank} ForwardPassAsync failed: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return node_pb2.ForwardPassResponse(success=False)

        def _get_layer_modules(self):
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
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    logger.info(f"TP worker rank {rank}/{world_size} on {device}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        trust = trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust,
            revision=hf_revision(),
        )

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust,
                revision=hf_revision(),
                low_cpu_mem_usage=True,
            )

        mesh = DeviceMesh("cuda", list(range(world_size)))

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
    raise RuntimeError(
        "tp_forward requires gRPC transport which was removed. "
        "Use Ray-native transport instead."
    )

    first = responses[0]
    if not first.success:
        raise RuntimeError(first.error_message or "tensor-parallel worker failed")
    return _tp_proto_to_tensor(first.output)


def main():
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
