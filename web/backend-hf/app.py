"""HF Space entry point — FastAPI + static frontend (Docker SDK, free tier)."""
import os

# 默认禁用 Chronos 大模型，节省硬件配额（回退到轻量合成预测）
if "LOAD_CHRONOS" not in os.environ:
    os.environ["LOAD_CHRONOS"] = "false"

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from server import app as fastapi_app

static_dir = Path(__file__).parent.parent  # serve from web/ root

# Serve static assets (CSS, JS, data)
fastapi_app.mount("/css", StaticFiles(directory=str(static_dir / "css")), name="css")
fastapi_app.mount("/js", StaticFiles(directory=str(static_dir / "js")), name="js")
fastapi_app.mount("/data", StaticFiles(directory=str(static_dir / "data")), name="data")

# Serve index.html at root
@fastapi_app.get("/")
async def serve_frontend():
    return FileResponse(str(static_dir / "index.html"))

app = fastapi_app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
