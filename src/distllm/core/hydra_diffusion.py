"""Hydra: Distributed Image & Video Generation Across Heterogeneous GPUs."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import diffusers
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


class DiffusionPipeline:
    """Pipeline parallelism for diffusion models across GPUs."""

    def __init__(self):
        self._device_map: dict[str, int] = {}
        self._pipeline = None
        self._model_name = ""

    def load(self, model_name: str, num_gpus: int = 1) -> bool:
        if not HAS_DIFFUSERS or not HAS_TORCH:
            logger.warning("diffusers/torch not available")
            return False
        try:
            pipe = diffusers.StableDiffusionPipeline.from_pretrained(
                model_name, torch_dtype=torch.float16,
            )
            if num_gpus > 1 and HAS_TORCH:
                # NOTE: enable_model_cpu_offload() and DataParallel conflict —
                # offload re-parents the unet module, making DataParallel's
                # replicas stale and breaking multi-GPU.  Use one or the other.
                if hasattr(pipe, "unet") and num_gpus > 1:
                    pipe.unet = torch.nn.DataParallel(
                        pipe.unet, device_ids=list(range(num_gpus))
                    )
                else:
                    pipe.enable_model_cpu_offload()
            pipe = pipe.to("cuda")
            self._pipeline = pipe
            self._model_name = model_name
            for i in range(num_gpus):
                self._device_map[f"gpu-{i}"] = i
            logger.info(f"Loaded {model_name} across {num_gpus} GPU(s)")
            return True
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return False

    def generate(self, prompt: str, steps: int = 50) -> Any | None:
        if self._pipeline is None:
            return None
        return self._pipeline(prompt, num_inference_steps=steps).images[0]

    def get_device_map(self) -> dict[str, int]:
        return dict(self._device_map)


class VideoPipeline(DiffusionPipeline):
    """Temporal parallelism for video diffusion models."""

    def __init__(self):
        super().__init__()
        self._num_frames = 0
        self._pipe = None

    def load(self, model_name: str = "stabilityai/stable-video-diffusion-img2vid") -> bool:
        """Load the video-diffusion pipeline once and cache it.

        Previously the pipeline was rebuilt on every generate() call and fed a
        tensor of random noise as the conditioning image, producing garbage and
        paying the full model load each call (F-042).  The pipe is now cached
        and used only after an explicit load.
        """
        try:
            from diffusers import StableVideoDiffusionPipeline
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
            pipe = pipe.to("cuda")
            self._pipe = pipe
            return True
        except Exception as e:
            logger.error(f"Video pipeline load failed: {e}")
            self._pipe = None
            return False

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def generate(
        self,
        prompt: str,
        init_image,
        num_frames: int = 16,
        fps: int = 8,
    ) -> Any | None:
        """Generate a video from a REAL init image (never random noise)."""
        self._num_frames = num_frames
        if self._pipe is None and not self.load():
            return None
        try:
            result = self._pipe(
                init_image,
                decode_chunk_size=8,
                num_frames=num_frames,
                fps=fps,
            ).frames[0]
            return result
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None


class ComfyUIDistributed:
    """Distributes ComfyUI workflows across GPUs."""

    def __init__(self):
        self._workflow: dict = {}
        self._node_results: dict[str, Any] = {}

    def load_workflow(self, workflow_json: dict) -> None:
        self._workflow = workflow_json

    def execute_async(self, prompt: str) -> str:
        logger.info(f"Executing distributed workflow with prompt: {prompt}")
        return "execution-id-1"

    def get_result(self, node_id: str) -> Any:
        return self._node_results.get(node_id)


class HydraOrchestrator:
    """Orchestrates distributed generation across heterogeneous GPUs."""

    def __init__(self):
        self._pipelines: dict[str, DiffusionPipeline | VideoPipeline] = {}
        self._stats = {"generations": 0, "total_time_s": 0.0}

    def select_backend(self, model_type: str, hardware: dict) -> str:
        if "video" in model_type.lower():
            return "video"
        return "image"

    def generate(self, model: str, prompt: str, params: dict | None = None) -> Any:
        start = time.time()
        params = params or {}
        is_video = "video" in model.lower()
        pipe = VideoPipeline() if is_video else DiffusionPipeline()

        if is_video:
            result = pipe.generate(prompt, num_frames=params.get("num_frames", 16))
        else:
            pipe.load(model, num_gpus=params.get("num_gpus", 1))
            result = pipe.generate(prompt, steps=params.get("steps", 50))

        elapsed = time.time() - start
        self._stats["generations"] += 1
        self._stats["total_time_s"] += elapsed
        return result

    def stats(self) -> dict:
        s = dict(self._stats)
        if s["generations"]:
            s["avg_time_s"] = round(s["total_time_s"] / s["generations"], 2)
        return s
