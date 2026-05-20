"""Multi-modal Vision-Language Model (VLM) support.

Supports LLaVA, Qwen-VL, InternVL and other AutoModelForVision2Seq models.
Handles image encoding via vision tower and projection to LLM embedding space.
"""

import base64
import io
import torch
from loguru import logger
from PIL import Image


# Image content from OpenAI chat format
class ImageContent:
    """Represents an image in chat messages (url or base64)."""

    def __init__(self, url: str | None = None, base64_data: str | None = None):
        self.url = url
        self.base64_data = base64_data

    @classmethod
    def from_dict(cls, data: dict) -> "ImageContent":
        if "url" in data:
            url = data["url"]
            if url.startswith("data:image"):
                # data:image/jpeg;base64,<data>
                base64_part = url.split(",", 1)[1] if "," in url else url
                return cls(base64_data=base64_part)
            return cls(url=url)
        if "base64" in data:
            return cls(base64_data=data["base64"])
        raise ValueError("ImageContent requires 'url' or 'base64' key")

    def to_pil(self) -> Image.Image:
        """Load image from URL or base64 data."""
        if self.base64_data:
            img_data = base64.b64decode(self.base64_data)
            return Image.open(io.BytesIO(img_data))
        if self.url:
            if self.url.startswith("data:image"):
                # Handle data URI that wasn't parsed by from_dict
                base64_part = self.url.split(",", 1)[1] if "," in self.url else self.url
                img_data = base64.b64decode(base64_part)
                return Image.open(io.BytesIO(img_data))
            elif self.url.startswith("http://") or self.url.startswith("https://"):
                import urllib.request
                with urllib.request.urlopen(self.url, timeout=30) as resp:
                    return Image.open(io.BytesIO(resp.read()))
            elif self.url.startswith("file://"):
                return Image.open(self.url[7:])
            else:
                return Image.open(self.url)
        raise ValueError("No image data available")


class VisionTower:
    """Vision encoder for VLMs (CLIP, SigLIP, etc.).

    Encodes images into visual embeddings that are projected into the
    LLM's embedding space.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        self.vision_model = None
        self.processor = None
        self.projector = None  # Linear layer mapping vision -> LLM dims

    def load(self) -> bool:
        """Load vision encoder and processor.

        Returns:
            True if loaded successfully.
        """
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            logger.error("transformers not installed")
            return False

        logger.info(f"Loading vision tower: {self.model_name}")
        device_map = "auto" if self.device == "auto" else self.device

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        torch_dtype = getattr(torch, self.dtype, torch.float16)
        self.vision_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.vision_model.eval()

        logger.info(f"Vision tower loaded: {self.model_name}")
        return True

    def encode_images(
        self,
        images: list[Image.Image] | torch.Tensor,
    ) -> torch.Tensor:
        """Encode images into visual features.

        Args:
            images: List of PIL images or pre-processed image tensor.

        Returns:
            Visual embeddings [batch, num_patches, vision_dim].
        """
        if self.vision_model is None:
            raise RuntimeError("Vision tower not loaded")

        if isinstance(images, torch.Tensor):
            # Already preprocessed
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=images)
                if hasattr(outputs, "last_hidden_state"):
                    return outputs.last_hidden_state
                return outputs[0]

        # Process PIL images
        if self.processor is None:
            raise RuntimeError("Processor not loaded")

        inputs = self.processor(images=images, return_tensors="pt")
        device = next(self.vision_model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs[0]

    def set_projector(self, projector: torch.nn.Module) -> None:
        """Set the vision-to-LLM projector."""
        self.projector = projector

    def project(
        self,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Project visual features into LLM embedding space.

        Args:
            visual_features: [batch, num_patches, vision_dim].

        Returns:
            [batch, num_patches, llm_hidden_size].
        """
        if self.projector is None:
            raise RuntimeError("Projector not set")
        return self.projector(visual_features)

    @property
    def vision_dim(self) -> int | None:
        if self.vision_model is None:
            return None
        return getattr(self.vision_model.config, "hidden_size", None)


class VLMPipeline:
    """End-to-end VLM processing for chat with images.

    Parses multi-modal messages, encodes images, builds the combined
    prompt with image tokens, and extracts the response.
    """

    # Image token placeholders used by different VLM families
    IMAGE_TOKENS = {
        "llava": "<image>",
        "qwen2_vl": "<|image_pad|>",
        "qwen_vl": "<img>",
        "internvl": "<IMG_CONTEXT>",
        "phi3_v": "<|image|>",
    }

    def __init__(
        self,
        vision_model_name: str | None = None,
        llm_hidden_size: int = 4096,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ):
        self.vision_tower: VisionTower | None = None
        self.llm_hidden_size = llm_hidden_size
        self.device = device
        self.dtype = dtype

        if vision_model_name:
            self.vision_tower = VisionTower(
                model_name=vision_model_name,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    def load_vision_tower(self) -> bool:
        if self.vision_tower is None:
            return False
        return self.vision_tower.load()

    def set_projector(self, projector: torch.nn.Module) -> None:
        if self.vision_tower:
            self.vision_tower.set_projector(projector)

    def parse_messages(
        self,
        messages: list[dict],
    ) -> tuple[str, list[ImageContent]]:
        """Parse OpenAI-format chat messages, extracting text and images.

        Args:
            messages: List of {role, content} dicts. Content can be string
                     or list of {type: "text"|"image_url", ...} dicts.

        Returns:
            (text_prompt, list of ImageContent)
        """
        images = []
        text_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                text_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Multi-modal content list
                msg_text = []
                for item in content:
                    item_type = item.get("type", "")
                    if item_type == "text":
                        msg_text.append(item.get("text", ""))
                    elif item_type == "image_url":
                        img = ImageContent.from_dict(item.get("image_url", {}))
                        images.append(img)
                        msg_text.append(self._get_image_token())
                text_parts.append(f"{role}: {''.join(msg_text)}")

        return "\n".join(text_parts), images

    def encode_images_to_embeddings(
        self,
        images: list[ImageContent],
    ) -> torch.Tensor | None:
        """Encode images into LLM embedding space.

        Args:
            images: List of ImageContent objects.

        Returns:
            [num_images * num_patches, llm_hidden_size] or None.
        """
        if not images or self.vision_tower is None:
            return None

        pil_images = [img.to_pil() for img in images]
        visual_features = self.vision_tower.encode_images(pil_images)

        # Project to LLM space
        if self.vision_tower.projector is not None:
            return self.vision_tower.project(visual_features)

        # Fallback: return visual features as-is (may need dimension match)
        return visual_features

    def build_prompt_with_images(
        self,
        text_prompt: str,
        image_embeddings: torch.Tensor | None,
    ) -> tuple[str, torch.Tensor | None]:
        """Build the final prompt, noting where image embeddings should be inserted.

        For systems that can prepend image tokens to the embedding stream,
        returns the prompt and the image embedding tensor separately.

        Returns:
            (prompt_text, image_embeddings or None)
        """
        return text_prompt, image_embeddings

    def count_image_tokens(self, image: Image.Image) -> int:
        """Calculate how many LLM tokens an image will occupy.

        Uses model-specific formulas:
        - LLaVA: fixed 576 tokens per image (24x24 grid)
        - Qwen2-VL: depends on resolution after dynamic resizing
        - Generic: estimate from vision_dim patches
        """
        if self.vision_tower is None:
            return 576  # LLaVA default

        model_lower = self.vision_tower.model_name.lower()

        # Qwen2-VL uses dynamic resolution
        if "qwen2_vl" in model_lower:
            w, h = image.size
            # Qwen2-VL resizes to multiples of 28, min 28x28, max 1280x1280
            factor = 28
            w = max(factor, min(1280, (w // factor) * factor))
            h = max(factor, min(1280, (h // factor) * factor))
            # Patch size is 14x14, each patch = 1 token after pooling
            patches_w = w // 14
            patches_h = h // 14
            return patches_w * patches_h

        # LLaVA uses fixed 24x24 grid from CLIP ViT-L
        if "llava" in model_lower:
            return 576  # 24x24 grid

        # InternVL uses dynamic grid
        if "internvl" in model_lower:
            w, h = image.size
            grid = min(max(w, h) // 224, 6)
            return grid * grid * 256  # 16x16 patches per grid cell

        # Generic estimate from vision tower config
        vision_dim = self.vision_tower.vision_dim
        if vision_dim is not None:
            # Assume ~1 token per 4 vision features (after pooling)
            config = getattr(self.vision_tower.vision_model, "config", None)
            if config and hasattr(config, "image_size") and hasattr(config, "patch_size"):
                img_size = config.image_size
                patch_size = config.patch_size
                patches = (img_size // patch_size) ** 2
                return patches
        return 576  # fallback

    def forward_to_llm(
        self,
        text_prompt: str,
        image_embeddings: torch.Tensor | None,
        llm_module,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Forward combined text + image embeddings through the LLM for generation.

        Args:
            text_prompt: The text prompt string.
            image_embeddings: Pre-computed image embedding tensor [num_tokens, hidden_dim].
            llm_module: The LLM model with forward(embeddings=...) support.
            tokenizer: Tokenizer for encoding text and decoding output.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        if image_embeddings is not None:
            # Build input embeddings: image tokens + text tokens
            text_tokens = tokenizer.encode(text_prompt, return_tensors="pt")
            if hasattr(llm_module, "get_input_embeddings"):
                text_embeds = llm_module.get_input_embeddings()(text_tokens.to(image_embeddings.device))
                # Concatenate image embeddings before text embeddings
                input_embeds = torch.cat([
                    image_embeddings.unsqueeze(0),
                    text_embeds,
                ], dim=1)
            else:
                input_embeds = image_embeddings.unsqueeze(0)

            # Generate using embeddings
            if hasattr(llm_module, "generate") and "inputs_embeds" in str(
                llm_module.generate.__code__.co_varnames
            ):
                outputs = llm_module.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
                return tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Fallback: try standard generate with position_ids adjustment
            outputs = llm_module.generate(
                input_ids=text_tokens.to(image_embeddings.device),
                inputs_embeds=input_embeds,
                max_new_tokens=max_new_tokens,
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Text-only generation
            input_ids = tokenizer.encode(text_prompt, return_tensors="pt")
            outputs = llm_module.generate(
                input_ids.to(next(llm_module.parameters()).device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def create_projector(
        self,
        vision_dim: int | None = None,
        llm_dim: int | None = None,
    ) -> torch.nn.Module:
        """Create a trainable projector module.

        Args:
            vision_dim: Vision tower output dimension (auto-detected if None).
            llm_dim: LLM hidden dimension (uses self.llm_hidden_size if None).

        Returns:
            Linear projector module.
        """
        if vision_dim is None:
            if self.vision_tower is None or self.vision_tower.vision_dim is None:
                raise ValueError("vision_dim not available. Load vision tower first.")
            vision_dim = self.vision_tower.vision_dim

        llm_dim = llm_dim or self.llm_hidden_size

        # Multi-layer projector for better representation learning
        projector = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, llm_dim),
            torch.nn.GELU(),
            torch.nn.Linear(llm_dim, llm_dim),
        )
        return projector

    def _get_image_token(self) -> str:
        """Get the image token placeholder for the detected model family."""
        if self.vision_tower is None:
            return "<image>"

        model_lower = self.vision_tower.model_name.lower()
        for family, token in self.IMAGE_TOKENS.items():
            if family in model_lower:
                return token
        return "<image>"

    def is_multimodal_message(
        self,
        messages: list[dict],
    ) -> bool:
        """Check if messages contain any image content."""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        return True
        return False
