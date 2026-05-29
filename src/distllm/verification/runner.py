"""Accuracy verification runner — orchestrates local vs distributed comparison.

Loads a model in both single-node and distributed configurations, runs
the same prompts through both, collects intermediate outputs (logits,
hidden states, token IDs), and compares them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from loguru import logger

from distllm.config.settings import DistLLMSettings
from distllm.core.inference_engine import InferenceEngine
from distllm.core.token_generator import TokenGenerator
from distllm.dist.pipeline import PipelineOrchestrator
from distllm.models.partitioner import ModelPartitioner, get_model_info
from distllm.dist.worker import WorkerNode
from distllm.verification.comparator import (
    OutputComparison,
    compare_hidden_states,
    compare_logits,
    compare_text,
    compare_tokens,
    evaluate_comparison,
)
from distllm.verification.hash_registry import GenerationOutput, OutputHashRegistry
from distllm.verification.report import VerificationReport, generate_report


@dataclass
class AccuracyVerifier:
    """Compares single-node (reference) vs distributed (candidate) inference.

    Steps:
      1. Load model on a single node (reference).
      2. Load the same model split across *num_nodes* (distributed).
      3. Run identical prompts through both paths.
      4. Collect logits, hidden states, and token IDs at each step.
      5. Compare outputs and generate a report.

    For a quick check without instantiating a multi-node cluster, use
    the ``run_pipeline_locally`` configuration which runs a simulated
    2-node pipeline in a single process.
    """

    model_name: str = ""
    device: str = "auto"
    dtype: str = "float16"
    num_nodes: int = 2
    trust_remote_code: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 32
    skip_text_comparison: bool = False
    preferred_backend: str = ""
    grpc_mode: bool = False
    grpc_base_port: int = 51050

    _token_gen: TokenGenerator = field(default_factory=TokenGenerator)
    _hash_registry: OutputHashRegistry = field(default_factory=OutputHashRegistry)

    _backend_name: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        from distllm.backends import select_backend

        if not self.model_name:
            raise ValueError("model_name is required")
        if self.preferred_backend:
            name = self.preferred_backend
            backend_cls = select_backend(preferred_backend=name)
            if backend_cls is None:
                logger.warning(
                    f"Preferred backend '{name}' not available, falling back to PyTorch"
                )
            self._backend_name = name
        else:
            selected = select_backend(device_type=self.device)
            if selected is not None:
                self._backend_name = getattr(selected, "display_name", lambda: "")()
                logger.info(f"Auto-selected backend: {self._backend_name}")

    def verify(
        self,
        prompts: str | list[str],
        collect_hidden_states: bool = False,
        thresholds: dict[str, float] | None = None,
    ) -> VerificationReport:
        """Run accuracy verification on one or more prompts.

        Args:
            prompts: Single prompt string or list of prompts.
            collect_hidden_states: If True, capture intermediate hidden
                states (memory intensive).
            thresholds: Custom metric thresholds.

        Returns:
            A ``VerificationReport`` with per-prompt comparisons and
            a summary.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Load reference model
        logger.info(f"Loading reference (single-node) model: {self.model_name}")
        ref_partitioner = ModelPartitioner(
            model_name=self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        ref_partitioner.load_full_model()
        device = next(ref_partitioner.full_model.parameters()).device

        per_prompt: list[dict[str, Any]] = []
        workers: list[WorkerNode] = []
        servers: list[NodeServer] = []

        try:
            if self.grpc_mode:
                logger.info(
                    f"Loading distributed ({self.num_nodes}-node) model via real gRPC"
                )
                logger.info(
                    f"Starting {self.num_nodes} gRPC worker nodes on ports "
                    f"{self.grpc_base_port}-{self.grpc_base_port + self.num_nodes - 1}"
                )
                workers, servers = self._start_grpc_workers()

                for prompt in prompts:
                    result = self._verify_single_grpc(
                        prompt=prompt,
                        ref_partitioner=ref_partitioner,
                        workers=workers,
                        device=device,
                    )
                    per_prompt.append(result)
            else:
                logger.info(
                    f"Loading distributed ({self.num_nodes}-node) model (simulated in-process)"
                )
                dist_partitioners = self._load_distributed(ref_partitioner)

                for prompt in prompts:
                    result = self._verify_single(
                        prompt=prompt,
                        ref_partitioner=ref_partitioner,
                        dist_partitioners=dist_partitioners,
                        device=device,
                        collect_hidden_states=collect_hidden_states,
                    )
                    per_prompt.append(result)

                for p in dist_partitioners:
                    if p is not None:
                        del p

            report = generate_report(
                comparisons=[r["comparison"] for r in per_prompt],
                per_prompt_data=[
                    {
                        "prompt": r["prompt"],
                        "reference": r["reference"],
                        "candidate": r["candidate"],
                    }
                    for r in per_prompt
                ],
                hash_registry=self._hash_registry,
                thresholds=thresholds,
                model_name=self.model_name,
                num_nodes=self.num_nodes,
                dtype=self.dtype,
            )
            return report

        finally:
            for s in servers:
                try:
                    s.stop(grace=1.0)
                except Exception:
                    pass
            for w in workers:
                try:
                    w.stop()
                except Exception:
                    pass
            del ref_partitioner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _verify_single(
        self,
        prompt: str,
        ref_partitioner: ModelPartitioner,
        dist_partitioners: list[ModelPartitioner | None],
        device: torch.device,
        collect_hidden_states: bool,
    ) -> dict[str, Any]:
        """Run reference and distributed inference on a single prompt."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # ── Reference: single-node forward ─────────────────────────────
        ref_output = self._run_reference(
            ref_partitioner, input_ids, tokenizer,
            collect_hidden_states=collect_hidden_states,
        )

        # ── Candidate: distributed (2+ node) forward ──────────────────
        cand_output = self._run_distributed(
            dist_partitioners, input_ids, tokenizer, device,
            collect_hidden_states=collect_hidden_states,
        )

        # ── Compare outputs ────────────────────────────────────────────
        logit_metrics = compare_logits(
            ref_output.step_logits[-1], cand_output.step_logits[-1]
        )

        token_metrics = compare_tokens(
            ref_output.token_ids,
            cand_output.token_ids,
        )

        text_metrics = {}
        if not self.skip_text_comparison:
            text_metrics = compare_text(ref_output.text, cand_output.text)

        hidden_metrics = {}
        if (
            collect_hidden_states
            and ref_output.step_hidden_states
            and cand_output.step_hidden_states
        ):
            hidden_metrics = compare_hidden_states(
                ref_output.step_hidden_states[-1],
                cand_output.step_hidden_states[-1],
            )

        metrics = {
            "token_exact_match": token_metrics["exact_match"],
            "token_edit_distance": token_metrics["edit_distance"],
            "logit_cosine_sim": logit_metrics["cosine_sim"],
            "logit_kl_div": logit_metrics["kl_div"],
            "logit_max_abs_diff": logit_metrics["max_abs_diff"],
        }
        if hidden_metrics:
            metrics["hidden_cosine_sim"] = hidden_metrics["cosine_sim"]
            metrics["hidden_max_abs_diff"] = hidden_metrics["max_abs_diff"]
            metrics["hidden_relative_error"] = hidden_metrics["relative_error"]

        comparison = evaluate_comparison(metrics)

        # Store hashes
        self._hash_registry.store_reference(prompt, ref_output)
        self._hash_registry.store_candidate(prompt, cand_output)

        logger.info(
            f"Prompt: {prompt[:50]}... "
            f"token_match={token_metrics['exact_match']:.2%} "
            f"logit_cosim={logit_metrics['cosine_sim']:.6f} "
            f"kl_div={logit_metrics['kl_div']:.6f} "
            f"pass={comparison.pass_threshold}"
        )

        return {
            "prompt": prompt,
            "comparison": comparison,
            "reference": ref_output,
            "candidate": cand_output,
        }

    def _start_grpc_workers(self) -> tuple[list[WorkerNode], list[NodeServer]]:
        """Start real gRPC worker nodes for distributed verification.

        Each worker loads its layer subset and starts a gRPC server.
        Returns (workers, servers) for shutdown by the caller.
        """
        from distllm.dist.node_service import NodeServer

        model_info = get_model_info(self.model_name, self.trust_remote_code)
        total = model_info.get("num_layers", 0)
        if total == 0:
            raise ValueError(f"Could not determine layer count for {self.model_name}")
        layers_per_node = (total + self.num_nodes - 1) // self.num_nodes

        workers: list[WorkerNode] = []
        servers: list[NodeServer] = []
        for i in range(self.num_nodes):
            start = i * layers_per_node
            end = min((i + 1) * layers_per_node, total)
            if start >= total:
                break
            port = self.grpc_base_port + i
            node_id = f"verify-{i}"
            worker = WorkerNode(
                node_id=node_id,
                model_name=self.model_name,
                start_layer=start,
                end_layer=end,
                total_layers=total,
                port=port,
                device=self.device,
                dtype=self.dtype,
            )
            worker.load_model()
            # Start gRPC server directly (without block via wait())
            from distllm.dist.node_service import NodeServer
            server = NodeServer(worker, port=port, max_workers=1)
            server.start()
            workers.append(worker)
            servers.append(server)

        if not workers:
            raise RuntimeError("No workers could be started")
        logger.info(
            f"Started {len(workers)} gRPC worker(s) on ports "
            f"{self.grpc_base_port}-{self.grpc_base_port + len(workers) - 1}"
        )
        return workers, servers

    def _verify_single_grpc(
        self,
        prompt: str,
        ref_partitioner: ModelPartitioner,
        workers: list[WorkerNode],
        device: torch.device,
    ) -> dict[str, Any]:
        """Run reference and real gRPC distributed inference on a single prompt."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Reference
        ref_output = self._run_reference(
            ref_partitioner, input_ids, tokenizer,
        )

        # Distributed via gRPC
        cand_output = self._run_distributed_grpc(
            workers, input_ids, tokenizer, device,
        )

        # Compare
        logit_metrics = compare_logits(
            ref_output.step_logits[-1], cand_output.step_logits[-1]
        )
        token_metrics = compare_tokens(
            ref_output.token_ids, cand_output.token_ids,
        )
        text_metrics = {}
        if not self.skip_text_comparison:
            text_metrics = compare_text(ref_output.text, cand_output.text)

        metrics = {
            "token_exact_match": token_metrics["exact_match"],
            "token_edit_distance": token_metrics["edit_distance"],
            "logit_cosine_sim": logit_metrics["cosine_sim"],
            "logit_kl_div": logit_metrics["kl_div"],
            "logit_max_abs_diff": logit_metrics["max_abs_diff"],
        }
        comparison = evaluate_comparison(metrics)
        self._hash_registry.store_reference(prompt, ref_output)
        self._hash_registry.store_candidate(prompt, cand_output)

        logger.info(
            f"Prompt: {prompt[:50]}... "
            f"token_match={token_metrics['exact_match']:.2%} "
            f"logit_cosim={logit_metrics['cosine_sim']:.6f} "
            f"pass={comparison.pass_threshold}"
        )

        return {
            "prompt": prompt,
            "comparison": comparison,
            "reference": ref_output,
            "candidate": cand_output,
        }

    def _run_distributed_grpc(
        self,
        workers: list[WorkerNode],
        input_ids: torch.Tensor,
        tokenizer: Any,
        device: torch.device,
    ) -> GenerationOutput:
        """Run generation through real gRPC worker nodes.

        Connects to each worker via NodeClient, sends ForwardPass requests
        in pipeline order. First node gets input_ids, middle nodes get
        hidden_states, last node returns logits.
        """
        from distllm.dist import node_pb2
        from distllm.dist.node_client import create_node_client
        from distllm.dist.node_service import tensor_to_proto, tensor_from_proto

        generated = input_ids.clone()
        step_logits: list[torch.Tensor] = []
        is_first = True
        request_id = "verify-grpc"

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                current = generated if is_first else generated[:, -1:]

                # Forward through each worker via gRPC
                for i, worker in enumerate(workers):
                    client = create_node_client(
                        "localhost", worker.port, timeout_s=10.0,
                    )
                    try:
                        is_last = (i == len(workers) - 1)

                        if i == 0:
                            # First node: send input_ids
                            req = node_pb2.ForwardPassRequest(
                                request_id=request_id,
                                input_ids=current[0].tolist(),
                                is_first_pass=is_first,
                                is_last_pass=is_last,
                                use_cache=True,
                            )
                        else:
                            # Middle/last node: send hidden_states
                            hidden_pb = tensor_to_proto(current)
                            req = node_pb2.ForwardPassRequest(
                                request_id=request_id,
                                hidden_states=hidden_pb,
                                is_first_pass=is_first,
                                is_last_pass=is_last,
                                use_cache=True,
                            )

                        resp = client.stub.ForwardPass(req, timeout=60.0)

                        if not resp.success:
                            raise RuntimeError(
                                f"Worker {worker.node_id} ForwardPass failed: "
                                f"{resp.error_message}"
                            )

                        # Decode output for next iteration
                        if resp.output and resp.output.raw_data:
                            current = tensor_from_proto(
                                resp.output, device=str(device)
                            )

                    finally:
                        client.close()

                # Collect logits from last worker output
                logits = current
                if logits.dim() >= 3:
                    logits = logits[:, -1, :]
                step_logits.append(logits.clone())

                # Sample next token
                next_token = self._token_gen.sample(
                    step_logits[-1],
                    temperature=self.temperature,
                    top_p=1.0,
                    top_k=0,
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=-1)
                is_first = False
                if next_token.item() == tokenizer.eos_token_id:
                    break

        text = tokenizer.decode(
            generated[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        return GenerationOutput(
            token_ids=generated[0].tolist(),
            text=text,
            step_logits=step_logits,
            model_name=self.model_name,
            temperature=self.temperature,
            prompt=tokenizer.decode(input_ids[0], skip_special_tokens=False),
        )

    def _run_reference(
        self,
        partitioner: ModelPartitioner,
        input_ids: torch.Tensor,
        tokenizer: Any,
        collect_hidden_states: bool = False,
    ) -> GenerationOutput:
        """Run single-node reference generation with logit capture."""
        generated = input_ids.clone()
        step_logits: list[torch.Tensor] = []
        step_hidden: list[torch.Tensor] | None = (
            [] if collect_hidden_states else None
        )

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                outputs = partitioner.full_model(generated)
                logits = outputs.logits[:, -1, :]
                step_logits.append(logits.clone())

                if collect_hidden_states and step_hidden is not None:
                    last_hidden = (
                        outputs.hidden_states[-1][:, -1, :].clone()
                        if hasattr(outputs, "hidden_states") and outputs.hidden_states
                        else logits.clone()
                    )
                    step_hidden.append(last_hidden)

                next_token = self._token_gen.sample(
                    logits,
                    temperature=self.temperature,
                    top_p=1.0,
                    top_k=0,
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        text = tokenizer.decode(
            generated[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        return GenerationOutput(
            token_ids=generated[0].tolist(),
            text=text,
            step_logits=step_logits,
            step_hidden_states=step_hidden,
            model_name=self.model_name,
            temperature=self.temperature,
            prompt=tokenizer.decode(input_ids[0], skip_special_tokens=False),
        )

    def _run_distributed(
        self,
        partitioners: list[ModelPartitioner | None],
        input_ids: torch.Tensor,
        tokenizer: Any,
        device: torch.device,
        collect_hidden_states: bool = False,
    ) -> GenerationOutput:
        """Simulate distributed pipeline in-process.

        Routes hidden states through each partitioner sequentially,
        mimicking the gRPC-based pipeline without network overhead.
        Includes optional INT8 quantization between stages to match
        real distributed behavior.
        """
        generated = input_ids.clone()
        step_logits: list[torch.Tensor] = []
        step_hidden: list[torch.Tensor] | None = (
            [] if collect_hidden_states else None
        )
        is_first = True
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                current = generated if is_first else generated[:, -1:]

                # Pass through each partitioner in order
                for i, partitioner in enumerate(partitioners):
                    if partitioner is None:
                        continue

                    if i == 0 and is_first:
                        # First node: embed input
                        hidden = partitioner.embed_input(current)
                    elif i == 0:
                        hidden = current
                    else:
                        hidden = current

                    # Simulate INT8 quantize/dequantize between stages
                    # (matches pipeline.py _tensor_quantize behavior)
                    if i > 0:
                        scale = hidden.abs().max().clamp(min=1e-5) / 127.0
                        hidden = (
                            (hidden / scale).round().clamp(-128, 127).to(torch.int8)
                        )
                        hidden = (hidden.to(hidden.dtype) * scale).to(hidden.dtype)

                    if i < len(partitioners) - 1:
                        # Middle node: forward, pass hidden to next
                        hidden, _ = partitioner.forward(
                            hidden, past_key_values=None,
                        )
                    else:
                        # Last node: forward with optional KV cache
                        hidden, new_kv = partitioner.forward(
                            hidden, past_key_values=kv_cache,
                        )
                        if kv_cache is None:
                            kv_cache = new_kv
                        else:
                            kv_cache = [
                                (torch.cat([k, nk], dim=-2), torch.cat([v, nv], dim=-2))
                                for (k, v), (nk, nv) in zip(kv_cache, new_kv)
                            ]

                # Last partitioner output: get logits
                logits = partitioners[-1].get_logits(hidden)
                step_logits.append(logits.clone())

                if collect_hidden_states and step_hidden is not None:
                    step_hidden.append(hidden.clone())

                next_token = self._token_gen.sample(
                    logits[:, -1, :] if logits.dim() > 2 else logits,
                    temperature=self.temperature,
                    top_p=1.0,
                    top_k=0,
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=-1)
                is_first = False
                if next_token.item() == tokenizer.eos_token_id:
                    break

        text = tokenizer.decode(
            generated[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        return GenerationOutput(
            token_ids=generated[0].tolist(),
            text=text,
            step_logits=step_logits,
            step_hidden_states=step_hidden,
            model_name=self.model_name,
            temperature=self.temperature,
            prompt=tokenizer.decode(input_ids[0], skip_special_tokens=False),
        )

    def _load_distributed(
        self, ref_partitioner: ModelPartitioner
    ) -> list[ModelPartitioner | None]:
        """Load layer subsets into N partitioners for distributed simulation.

        Splits the reference model's layers across *num_nodes* partitioners,
        each loading only its assigned layers. This mirrors what happens on
        real cluster nodes.
        """
        total = ref_partitioner.total_layers
        layers_per_node = (total + self.num_nodes - 1) // self.num_nodes

        partitioners: list[ModelPartitioner | None] = []
        for i in range(self.num_nodes):
            start = i * layers_per_node
            end = min((i + 1) * layers_per_node, total)
            if start >= total:
                partitioners.append(None)
                continue
            p = ModelPartitioner(
                model_name=self.model_name,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            )
            p.load_layer_subset(start, end, total)
            partitioners.append(p)
        return partitioners


def verify_accuracy(
    model_name: str,
    prompts: str | list[str],
    num_nodes: int = 2,
    dtype: str = "float16",
    temperature: float = 0.0,
    max_new_tokens: int = 32,
    collect_hidden_states: bool = False,
    thresholds: dict[str, float] | None = None,
    preferred_backend: str = "",
    grpc_mode: bool = False,
    grpc_base_port: int = 51050,
    trust_remote_code: bool = False,
) -> VerificationReport:
    """Convenience function: create a verifier, run it, return the report.

    Args:
        model_name: HuggingFace model name or path.
        prompts: Single prompt string or list of prompts.
        num_nodes: Number of distributed nodes to simulate.
        dtype: Model dtype.
        temperature: Sampling temperature (0 = greedy).
        max_new_tokens: Max tokens to generate per prompt.
        collect_hidden_states: Capture intermediate hidden states.
        thresholds: Custom metric thresholds.
        preferred_backend: Preferred inference backend name.
        grpc_mode: Use real gRPC workers instead of in-process simulation.
        grpc_base_port: Base port for gRPC workers.
        trust_remote_code: Trust remote code in HuggingFace models.

    Usage:
        report = verify_accuracy("HuggingFaceTB/SmolLM-135M", ["Hello"])
        print(report.summary())
    """
    verifier = AccuracyVerifier(
        model_name=model_name,
        dtype=dtype,
        num_nodes=num_nodes,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        preferred_backend=preferred_backend,
        grpc_mode=grpc_mode,
        grpc_base_port=grpc_base_port,
        trust_remote_code=trust_remote_code,
    )
    return verifier.verify(
        prompts=prompts,
        collect_hidden_states=collect_hidden_states,
        thresholds=thresholds,
    )


def run_verification_cli(
    model_name: str,
    num_nodes: int = 2,
    dtype: str = "float16",
    temperature: float = 0.0,
    max_new_tokens: int = 32,
    prompts: list[str] | None = None,
    preferred_backend: str = "",
    grpc_mode: bool = False,
    grpc_base_port: int = 51050,
) -> None:
    """CLI entry point for accuracy verification.

    Runs verification and prints a formatted report to stdout.
    """
    if not prompts:
        prompts = [
            "The capital of France is",
            "In the beginning",
            "The meaning of life is",
            "Write a haiku about",
        ]

    report = verify_accuracy(
        model_name=model_name,
        prompts=prompts,
        num_nodes=num_nodes,
        dtype=dtype,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        preferred_backend=preferred_backend,
        grpc_mode=grpc_mode,
        grpc_base_port=grpc_base_port,
    )

    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    console.print(
        Panel(
            f"[bold]Model:[/] {report.model_name}\n"
            f"[bold]Nodes:[/] {report.num_nodes}  "
            f"[bold]Dtype:[/] {report.dtype}  "
            f"[bold]Temp:[/] {report.temperature}",
            title="[bold cyan]Accuracy Verification[/]",
        )
    )

    table = Table(title="Per-Prompt Results")
    table.add_column("Prompt", style="cyan", no_wrap=False)
    table.add_column("Token Match", justify="right")
    table.add_column("Logit Cosim", justify="right")
    table.add_column("KL Div", justify="right")
    table.add_column("Pass", justify="center")

    for result in report.per_prompt:
        comp = result["comparison"]
        pass_str = (
            "[green]PASS[/]" if comp.pass_threshold else "[red]FAIL[/]"
        )
        table.add_row(
            result["prompt"][:40],
            f"{comp.token_exact_match:.1%}",
            f"{comp.logit_cosine_sim:.6f}",
            f"{comp.logit_kl_div:.6f}",
            pass_str,
        )

    console.print(table)
    summary = report.summary()
    console.print(
        f"\n[bold]Summary:[/] {summary['passed']}/{summary['total']} passed "
        f"({summary['pass_rate']:.1%})"
    )
    if summary.get("hash_comparison"):
        h = summary["hash_comparison"]
        console.print(
            f"Hash registry: {h['passed']}/{h['total']} match "
            f"({h['pass_rate']:.1%})"
        )
