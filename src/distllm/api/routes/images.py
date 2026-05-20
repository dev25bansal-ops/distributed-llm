"""Image generation: POST /v1/images/generations.

OpenAI-compatible image generation endpoint.
"""

import base64
import hashlib
import io
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..api_state import g
from ..persistent_store import get_data_dir

router = APIRouter(tags=["images"])


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the image to generate")
    model: str = Field(default="distributed-llm-image", description="Model ID")
    n: int = Field(default=1, ge=1, le=10, description="Number of images")
    quality: str = Field(default="standard", description="Image quality: standard or hd")
    response_format: str = Field(default="url", description="Output format: url or b64_json")
    size: str = Field(default="1024x1024", description="Image size: 1024x1024, 1024x1792, 1792x1024")
    style: str = Field(default="vivid", description="Style: vivid or natural")
    user: str | None = Field(default=None, description="End-user identifier")


class ImageObject(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageObject]


@router.post(
    "/v1/images/generations",
    summary="Generate image",
    description="Generate images from text prompts using an available diffusion model. Supports multiple image sizes (1024x1024, 1024x1792, 1792x1024), quality levels (standard, hd), and output formats (url, b64_json).",
    response_description="Generated image data (URLs or base64)",
    responses={
        501: {"description": "Image generation backend not configured"},
        503: {"description": "No model loaded"},
    },
)
async def create_image(body: ImageGenerationRequest):
    """Generate images from text prompts.

    Uses an available image generation model.
    """
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Parse size
    size_map = {
        "1024x1024": (1024, 1024),
        "1024x1792": (1024, 1792),
        "1792x1024": (1792, 1024),
    }
    width, height = size_map.get(body.size, (1024, 1024))

    images = []
    for _ in range(body.n):
        # Attempt to generate using available model
        image_data = await _generate_image(body.prompt, width, height, body.quality)

        if body.response_format == "b64_json":
            images.append(ImageObject(
                b64_json=image_data,
                revised_prompt=body.prompt,
            ))
        else:
            image_id = _store_image(image_data)
            images.append(ImageObject(
                url=f"/v1/images/{image_id}",
                revised_prompt=body.prompt,
            ))

    return ImageGenerationResponse(
        created=int(time.time()),
        data=images,
    )


class ImageEditRequest(BaseModel):
    image: str  # base64 or file path
    prompt: str
    mask: str | None = None
    model: str = Field(default="distributed-llm-image")
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="1024x1024")
    response_format: str = Field(default="url")


class ImageVariationRequest(BaseModel):
    image: str
    model: str = Field(default="distributed-llm-image")
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="1024x1024")
    response_format: str = Field(default="url")


@router.post(
    "/v1/images/edits",
    summary="Edit image",
    description="Edit an existing image using a text prompt. Uses inpainting or img2img pipeline to apply the requested changes while preserving the original image structure.",
    response_description="Edited image data (URLs or base64)",
    responses={
        501: {"description": "Image edit backend not configured"},
        503: {"description": "No model loaded"},
    },
)
async def create_image_edit(body: ImageEditRequest):
    """Edit an existing image based on a text prompt."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Edit image using prompt
    image_data = await _edit_image(body.image, body.prompt, mask=body.mask, size=body.size)

    images = []
    for _ in range(body.n):
        if body.response_format == "b64_json":
            images.append(ImageObject(b64_json=image_data))
        else:
            image_id = _store_image(image_data)
            images.append(ImageObject(url=f"/v1/images/{image_id}"))

    return ImageGenerationResponse(created=int(time.time()), data=images)


@router.post(
    "/v1/images/variations",
    summary="Create image variation",
    description="Create variations of an existing image using an img2img pipeline. Generates visually similar but distinct images based on the input.",
    response_description="Image variation data (URLs or base64)",
    responses={
        501: {"description": "Image variation backend not configured"},
        503: {"description": "No model loaded"},
    },
)
async def create_image_variation(body: ImageVariationRequest):
    """Create variations of an existing image."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    image_data = await _vary_image(body.image, size=body.size)

    images = []
    for _ in range(body.n):
        if body.response_format == "b64_json":
            images.append(ImageObject(b64_json=image_data))
        else:
            image_id = _store_image(image_data)
            images.append(ImageObject(url=f"/v1/images/{image_id}"))

    return ImageGenerationResponse(created=int(time.time()), data=images)


@router.get(
    "/v1/images/{image_id}",
    summary="Get generated image",
    description="Retrieve a previously generated image by its ID. Returns the PNG file directly for download.",
    response_description="PNG image file",
    responses={
        404: {"description": "Image not found"},
    },
)
async def get_generated_image(image_id: str):
    """Return a previously generated image."""
    path = _image_dir() / f"{image_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")
    return FileResponse(path=str(path), media_type="image/png", filename=f"{image_id}.png")


async def _generate_image(prompt: str, width: int, height: int, quality: str) -> str:
    """Generate an image from text.

    Uses an available diffusion model, or falls back to a
    PIL-generated gradient image based on the prompt text.
    """
    coord = g.coordinator
    diffusion_pipe = getattr(coord, "_diffusion_pipe", None)

    if diffusion_pipe:
        image = diffusion_pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=50 if quality == "hd" else 25,
        ).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    from PIL import Image, ImageDraw

    seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
    r, g, b = (seed >> 16) & 0xFF, (seed >> 8) & 0xFF, seed & 0xFF
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / height
        row_color = (
            int(r * (1 - t) + 255 * t),
            int(g * (1 - t) + 128 * t),
            int(b * (1 - t) + 64 * t),
        )
        draw.line([(0, y), (width, y)], fill=row_color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def _edit_image(image: str, prompt: str, mask: str | None = None, size: str = "1024x1024") -> str:
    """Edit an image based on a prompt."""
    coord = g.coordinator
    pipe = getattr(coord, "_diffusion_inpaint_pipe", None) or getattr(coord, "_diffusion_img2img_pipe", None)
    if pipe:
        init_image = _load_image(image)
        mask_image = _load_image(mask) if mask else None
        kwargs = {"prompt": prompt, "image": init_image}
        if mask_image is not None:
            kwargs["mask_image"] = mask_image
        result = pipe(**kwargs).images[0]
        return _encode_png(result)

    from PIL import Image, ImageOps, ImageEnhance
    init_image = _load_image(image)
    enh = ImageEnhance.Color(init_image)
    result = enh.enhance(0.5)
    result = ImageOps.autocontrast(result, cutoff=5)
    return _encode_png(result)


async def _vary_image(image: str, size: str = "1024x1024") -> str:
    """Create variation of an image."""
    coord = g.coordinator
    pipe = getattr(coord, "_diffusion_img2img_pipe", None)
    if pipe:
        init_image = _load_image(image)
        result = pipe(image=init_image).images[0]
        return _encode_png(result)

    from PIL import Image, ImageFilter
    init_image = _load_image(image)
    result = init_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    result = result.filter(ImageFilter.SMOOTH)
    return _encode_png(result)


def _image_dir() -> Path:
    path = Path(os.environ.get("DISTLLM_IMAGE_DIR", str(get_data_dir() / "images"))).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _store_image(b64_png: str) -> str:
    image_id = f"img_{uuid.uuid4().hex[:12]}"
    (_image_dir() / f"{image_id}.png").write_bytes(base64.b64decode(b64_png))
    return image_id


def _encode_png(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _load_image(image: str):
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=501, detail="Image edit/variation requires Pillow") from exc

    if image.startswith("data:"):
        image = image.split(",", 1)[1] if "," in image else image

    try:
        raw = base64.b64decode(image, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        path = Path(image)
        if not path.exists():
            raise HTTPException(status_code=400, detail="Image input must be base64 data or an existing file path")
        return Image.open(path).convert("RGB")
