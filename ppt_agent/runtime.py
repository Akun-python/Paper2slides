from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import uuid


DEFAULT_RUNTIME_ROOT = "runtime"


def slugify_project_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "")
    slug = slug.strip("_")
    return slug or "project"


def make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{timestamp}_{suffix}"


def resolve_runtime_root(runtime_root: str | Path | None = None) -> Path:
    value = runtime_root or DEFAULT_RUNTIME_ROOT
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class RunPaths:
    runtime_root: Path
    project_slug: str
    run_id: str
    run_dir: Path
    uploads_dir: Path
    cache_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    figures_dir: Path
    figures_selected_dir: Path
    slides_dir: Path

    @property
    def run_log_path(self) -> Path:
        return self.logs_dir / "run.log"

    @property
    def error_log_path(self) -> Path:
        return self.logs_dir / "error.log"


def build_run_paths(
    project_name: str,
    runtime_root: str | Path | None = None,
    run_id: str | None = None,
    create_dirs: bool = True,
    reuse_existing: bool = False,
) -> RunPaths:
    root = resolve_runtime_root(runtime_root)
    project_slug = slugify_project_name(project_name)
    selected_run_id = run_id or make_run_id()
    run_dir = root / "runs" / project_slug / selected_run_id

    if not reuse_existing:
        counter = 1
        while run_dir.exists():
            run_dir = root / "runs" / project_slug / f"{selected_run_id}_{counter}"
            counter += 1
        selected_run_id = run_dir.name

    paths = RunPaths(
        runtime_root=root,
        project_slug=project_slug,
        run_id=selected_run_id,
        run_dir=run_dir,
        uploads_dir=run_dir / "uploads",
        cache_dir=run_dir / "cache",
        logs_dir=run_dir / "logs",
        artifacts_dir=run_dir / "artifacts",
        figures_dir=run_dir / "figures",
        figures_selected_dir=run_dir / "figures_selected",
        slides_dir=run_dir / "slides",
    )

    if create_dirs:
        paths.uploads_dir.mkdir(parents=True, exist_ok=True)
        paths.cache_dir.mkdir(parents=True, exist_ok=True)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        paths.figures_dir.mkdir(parents=True, exist_ok=True)
        paths.figures_selected_dir.mkdir(parents=True, exist_ok=True)
        paths.slides_dir.mkdir(parents=True, exist_ok=True)

    return paths


def build_service_log_path(
    service_name: str,
    runtime_root: str | Path | None = None,
) -> Path:
    root = resolve_runtime_root(runtime_root)
    service_dir = root / "service"
    service_dir.mkdir(parents=True, exist_ok=True)
    return service_dir / f"{service_name}.log"


def build_project_cache_root(
    project_name: str,
    runtime_root: str | Path | None = None,
) -> Path:
    root = resolve_runtime_root(runtime_root)
    project_slug = slugify_project_name(project_name)
    cache_root = root / "cache" / project_slug
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root
