"""FastAPI application."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.demo import router as demo_router
from app.api.routes import router
from app.db import Base, make_engine

_DEMO_PAGE = Path(__file__).resolve().parents[1] / "demo" / "page.html"
_CUSTOM_PAGE = Path(__file__).resolve().parents[1] / "demo" / "custom.html"


def create_app(database_url: str | None = None) -> FastAPI:
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///./freight.db")
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    app = FastAPI(title="城市配送智能配载", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.state.database_url = url
    app.include_router(router, prefix="/v1")
    app.include_router(demo_router, prefix="/v1")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo_page() -> str:
        return _DEMO_PAGE.read_text(encoding="utf-8")

    @app.get("/custom", response_class=HTMLResponse, include_in_schema=False)
    def custom_page() -> str:
        return _CUSTOM_PAGE.read_text(encoding="utf-8")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000)
