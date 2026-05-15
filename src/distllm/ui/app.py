"""DistLLM UI - Web interface for Distributed LLM."""

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
import uvicorn


UI_DIR = Path(__file__).parent

ui_app = FastAPI(
    title="DistLLM UI",
    description="Web interface for Distributed LLM",
    version="0.3.0",
)

# Mount static files
static_dir = UI_DIR / "static"
if static_dir.exists():
    ui_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates = Jinja2Templates(directory=str(UI_DIR / "templates"))

# Default API server URL
API_URL = "http://localhost:8000"


@ui_app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request, "api_url": API_URL})


@ui_app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Cluster dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request, "api_url": API_URL})


@ui_app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """Model browser."""
    return templates.TemplateResponse("models.html", {"request": request, "api_url": API_URL})


@ui_app.get("/api/health")
async def health_proxy():
    """Proxy health check from API server."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/health", timeout=10.0)
            return response.json()
    except Exception:
        return {"status": "unavailable", "reason": "API server not reachable"}


def main():
    """Run the UI server."""
    import argparse

    parser = argparse.ArgumentParser(description="DistLLM Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="UI server host")
    parser.add_argument("--port", type=int, default=8500, help="UI server port")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API server URL")

    args = parser.parse_args()

    global API_URL
    API_URL = args.api_url

    print(f"Starting DistLLM UI on http://{args.host}:{args.port}")
    print(f"Connecting to API: {API_URL}")
    print(f"Open http://localhost:{args.port} in your browser")

    uvicorn.run(ui_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
