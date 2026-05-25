"""Image generation, edit, variation, and retrieval tests."""

import base64
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


def _png_b64(size=(64, 64), color="red"):
    from PIL import Image
    import io
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestCreateImage:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord._diffusion_pipe = None
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_generate_image_url_format(self):
        resp = TestClient(app).post(
            "/v1/images/generations",
            json={"prompt": "a cat", "n": 1, "response_format": "url"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "created" in data
        assert len(data["data"]) == 1
        img = data["data"][0]
        assert img["url"].startswith("/v1/images/")
        assert img["revised_prompt"] == "a cat"

    def test_generate_image_b64_format(self):
        resp = TestClient(app).post(
            "/v1/images/generations",
            json={"prompt": "a dog", "n": 1, "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        img = data["data"][0]
        assert img["b64_json"] is not None
        raw = base64.b64decode(img["b64_json"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_quality_hd(self):
        resp = TestClient(app).post(
            "/v1/images/generations",
            json={"prompt": "a cat", "quality": "hd", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        raw = base64.b64decode(resp.json()["data"][0]["b64_json"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_invalid_size_defaults(self):
        resp = TestClient(app).post(
            "/v1/images/generations",
            json={"prompt": "test", "size": "512x512", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        raw = base64.b64decode(resp.json()["data"][0]["b64_json"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_size_1024x1792(self):
        resp = TestClient(app).post(
            "/v1/images/generations",
            json={"prompt": "test", "size": "1024x1792", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        raw = base64.b64decode(resp.json()["data"][0]["b64_json"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_size_1792x1024(self):
        resp = TestClient(app).post(
            "/v1/images/generations",
            json={"prompt": "test", "size": "1792x1024", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        raw = base64.b64decode(resp.json()["data"][0]["b64_json"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_multiple_images(self):
        resp = TestClient(app).post(
            "/v1/images/generations",
            json={"prompt": "test", "n": 3, "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 3

    def test_generate_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/images/generations",
                json={"prompt": "test"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestGetGeneratedImage:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord._diffusion_pipe = None
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_get_generated_image(self):
        resp = TestClient(app).post(
            "/v1/images/generations",
            json={"prompt": "test", "n": 1, "response_format": "url"},
        )
        assert resp.status_code == 200
        url = resp.json()["data"][0]["url"]

        resp = TestClient(app).get(url)
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "image/png"
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_get_nonexistent_image(self):
        resp = TestClient(app).get("/v1/images/img_nonexistent")
        assert resp.status_code == 404


class TestCreateImageEdit:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord._diffusion_inpaint_pipe = None
        coord._diffusion_img2img_pipe = None
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_edit_with_size(self):
        resp = TestClient(app).post(
            "/v1/images/edits",
            json={"image": _png_b64(), "prompt": "make it blue", "size": "1024x1792", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        raw = base64.b64decode(resp.json()["data"][0]["b64_json"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_edit_image(self):
        resp = TestClient(app).post(
            "/v1/images/edits",
            json={"image": _png_b64(), "prompt": "make it blue", "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        raw = base64.b64decode(data["data"][0]["b64_json"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_edit_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/images/edits",
                json={"image": _png_b64(), "prompt": "test"},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original

    def test_edit_invalid_image_returns_400(self):
        resp = TestClient(app).post(
            "/v1/images/edits",
            json={"image": "not-valid-base64!!", "prompt": "test"},
        )
        assert resp.status_code == 400

    def test_edit_missing_pil_returns_501(self):
        from fastapi import HTTPException
        def broken_load(img):
            raise HTTPException(status_code=501, detail="Image edit/variation requires Pillow")
        with patch("distllm.api.routes.images._load_image", broken_load):
            resp = TestClient(app).post(
                "/v1/images/edits",
                json={"image": "dGVzdA==", "prompt": "test"},
            )
        assert resp.status_code == 501

    def test_vary_missing_pil_returns_501(self):
        from fastapi import HTTPException
        def broken_load(img):
            raise HTTPException(status_code=501, detail="Image edit/variation requires Pillow")
        with patch("distllm.api.routes.images._load_image", broken_load):
            resp = TestClient(app).post(
                "/v1/images/variations",
                json={"image": "dGVzdA=="},
            )
        assert resp.status_code == 501


class TestCreateImageVariation:
    @pytest.fixture(autouse=True)
    def setup(self):
        original = g.coordinator
        coord = MagicMock()
        coord.model_name = "test-model"
        coord.nodes = {}
        coord._shutting_down = False
        coord._diffusion_img2img_pipe = None
        g.coordinator = coord
        yield
        g.coordinator = original

    def test_vary_image(self):
        resp = TestClient(app).post(
            "/v1/images/variations",
            json={"image": _png_b64(), "response_format": "b64_json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        raw = base64.b64decode(data["data"][0]["b64_json"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_vary_no_coordinator(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/images/variations",
                json={"image": _png_b64()},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original

    def test_vary_invalid_image_returns_400(self):
        resp = TestClient(app).post(
            "/v1/images/variations",
            json={"image": "bad-data"},
        )
        assert resp.status_code == 400


class TestLoadImage:
    def test_load_base64_png(self):
        from distllm.api.routes.images import _load_image
        from PIL import Image
        img = _load_image(_png_b64())
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_base64_png_with_data_uri(self):
        from distllm.api.routes.images import _load_image
        from PIL import Image
        img = _load_image("data:image/png;base64," + _png_b64())
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_from_file_path(self, tmp_path):
        from distllm.api.routes.images import _load_image
        from PIL import Image
        f = tmp_path / "test.png"
        img_in = Image.new("RGB", (4, 4), color="blue")
        img_in.save(str(f))
        img_out = _load_image(str(f))
        assert isinstance(img_out, Image.Image)
        assert img_out.mode == "RGB"
        assert img_out.size == (4, 4)
