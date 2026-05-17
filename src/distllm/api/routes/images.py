"""Image generation: POST /v1/images/generations.

OpenAI-compatible image generation endpoint.
"""

import base64
import io
import os
import time
import uuid
from typing import List, Optional, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["images"])


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the image to generate")
    model: str = Field(default="distributed-llm-image", description="Model ID")
    n: int = Field(default=1, ge=1, le=10, description="Number of images")
    quality: str = Field(default="standard", description="Image quality: standard or hd")
    response_format: str = Field(default="url", description="Output format: url or b64_json")
    size: str = Field(default="1024x1024", description="Image size: 1024x1024, 1024x1792, 1792x1024")
    style: str = Field(default="vivid", description="Style: vivid or natural")
    user: Optional[str] = Field(default=None, description="End-user identifier")


class ImageObject(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageObject]


@router.post("/v1/images/generations")
async def create_image(body: ImageGenerationRequest):
    """Generate images from text prompts.

    Uses available image generation model or returns placeholder images
    when no model is configured.
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
            # Return URL (placeholder - in production this would be a CDN URL)
            image_id = f"img_{uuid.uuid4().hex[:12]}"
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
    mask: Optional[str] = None
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


@router.post("/v1/images/edits")
async def create_image_edit(body: ImageEditRequest):
    """Edit an existing image based on a text prompt."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Edit image using prompt
    image_data = await _edit_image(body.image, body.prompt, mask=body.mask)

    images = []
    for _ in range(body.n):
        if body.response_format == "b64_json":
            images.append(ImageObject(b64_json=image_data))
        else:
            images.append(ImageObject(url=f"/v1/images/{uuid.uuid4().hex[:12]}"))

    return ImageGenerationResponse(created=int(time.time()), data=images)


@router.post("/v1/images/variations")
async def create_image_variation(body: ImageVariationRequest):
    """Create variations of an existing image."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    image_data = await _vary_image(body.image)

    images = []
    for _ in range(body.n):
        if body.response_format == "b64_json":
            images.append(ImageObject(b64_json=image_data))
        else:
            images.append(ImageObject(url=f"/v1/images/{uuid.uuid4().hex[:12]}"))

    return ImageGenerationResponse(created=int(time.time()), data=images)


async def _generate_image(prompt: str, width: int, height: int, quality: str) -> str:
    """Generate an image from text.

    Attempts to use available diffusion model, falls back to placeholder.
    """
    coord = g.coordinator
    diffusion_model = getattr(coord, "_diffusion_model", None)
    diffusion_pipe = getattr(coord, "_diffusion_pipe", None)

    if diffusion_pipe:
        import torch
        image = diffusion_pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=50 if quality == "hd" else 25,
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    # Fallback: generate a simple colored placeholder image
    try:
        from PIL import Image
        img = Image.new('RGB', (width, height), color=(60, 60, 120))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    except ImportError:
        # Return minimal PNG
        return base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100).decode()


async def _edit_image(image: str, prompt: str, mask: Optional[str] = None) -> str:
    """Edit an image based on a prompt."""
    # Placeholder: return original image encoded
    if image.startswith('data:'):
        return image.split(',')[1] if ',' in image else image
    return image


async def _vary_image(image: str) -> str:
    """Create variation of an image."""
    return image
