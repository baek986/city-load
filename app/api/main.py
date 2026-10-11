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

_HOME = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>城市配送智能配载</title>
  <style>
    body { font-family: "Segoe UI", sans-serif; margin: 2rem auto; max-width: 48rem; line-height: 1.5; color: #1a1a1a; }
    code { background: #f2f2f2; padding: 0.1rem 0.3rem; }
    li { margin: 0.25rem 0; }
    .method { display: inline-block; min-width: 4.5rem; font-weight: 600; }
  </style>
</head>
<body>
  <h1>城市配送智能配载</h1>
    <p>规划接口仍然从 <code>POST /v1/plans</code> 进入。要看路线和车厢，打开 <a href="/demo">配送演示</a>。</p>
  <h2>接口</h2>
  <ul id="paths"><li>正在读取接口列表…</li></ul>
  <script>
    fetch("/openapi.json").then(function (response) { return response.json(); }).then(function (spec) {
      var list = document.getElementById("paths");
      list.innerHTML = "";
      Object.keys(spec.paths || {}).sort().forEach(function (path) {
        Object.keys(spec.paths[path]).forEach(function (method) {
          var item = document.createElement("li");
          item.innerHTML = "<span class=\\"method\\">" + method.toUpperCase() + "</span> <code>" + path + "</code>";
          list.appendChild(item);
        });
      });
    }).catch(function () {
      document.getElementById("paths").innerHTML = "<li>接口列表读取失败</li>";
    });
  </script>
</body>
</html>
"""


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
    @app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return _HOME

    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo_page() -> str:
        return _DEMO_PAGE.read_text(encoding="utf-8")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000)
