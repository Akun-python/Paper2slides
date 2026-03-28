from __future__ import annotations

import builtins
import contextlib
import contextvars
import io
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from dataclasses import dataclass

from ppt_agent.runtime import RunPaths, build_service_log_path


LOGGER_NAMESPACE = "paper2slides"
DEFAULT_LOG_LEVEL = "INFO"
_CURRENT_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "paper2slides_run_id",
    default="-",
)


@dataclass(frozen=True)
class RunLogPaths:
    run_log_path: Path
    error_log_path: Path


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str | None = None) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self._run_id or _CURRENT_RUN_ID.get()
        return True


class _StdStreamBridge(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self._logger.log(self._level, line)
        return len(s)

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.strip())
        self._buffer = ""


def _coerce_log_level(log_level: str | int | None) -> int:
    if isinstance(log_level, int):
        return log_level
    if not log_level:
        return logging.INFO
    value = str(log_level).strip().upper()
    return getattr(logging, value, logging.INFO)


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | run_id=%(run_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(
    log_level: str | int | None = None,
    run_paths: RunPaths | None = None,
    service_name: str | None = None,
    runtime_root: str | Path | None = None,
) -> logging.Logger:
    level = _coerce_log_level(log_level or DEFAULT_LOG_LEVEL)
    base_logger = logging.getLogger(LOGGER_NAMESPACE)
    base_logger.setLevel(level)
    base_logger.propagate = False

    for handler in list(base_logger.handlers):
        if getattr(handler, "_paper2slides_run_handler", False):
            base_logger.removeHandler(handler)
            handler.close()

    formatter = _build_formatter()
    run_id_filter = _RunIdFilter()

    console_handler = None
    for handler in base_logger.handlers:
        if getattr(handler, "_paper2slides_console_handler", False):
            console_handler = handler
            break
    if console_handler is None:
        console_handler = logging.StreamHandler(stream=sys.__stderr__)
        console_handler._paper2slides_console_handler = True  # type: ignore[attr-defined]
        console_handler.addFilter(run_id_filter)
        console_handler.setFormatter(formatter)
        base_logger.addHandler(console_handler)
    console_handler.setLevel(level)

    if service_name:
        service_path = build_service_log_path(service_name, runtime_root=runtime_root)
        existing = None
        for handler in base_logger.handlers:
            if getattr(handler, "_paper2slides_service_handler", False):
                existing = handler
                break
        if existing is None or Path(getattr(existing, "baseFilename", "")) != service_path:
            if existing is not None:
                base_logger.removeHandler(existing)
                existing.close()
            service_handler = RotatingFileHandler(
                service_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            service_handler._paper2slides_service_handler = True  # type: ignore[attr-defined]
            service_handler.addFilter(run_id_filter)
            service_handler.setFormatter(formatter)
            service_handler.setLevel(level)
            base_logger.addHandler(service_handler)
        else:
            existing.setLevel(level)

    if run_paths is not None:
        attach_run_file_handlers(base_logger, run_paths, level)

    return base_logger


def attach_run_file_handlers(
    base_logger: logging.Logger,
    run_paths: RunPaths,
    log_level: str | int | None = None,
) -> RunLogPaths:
    level = _coerce_log_level(log_level or DEFAULT_LOG_LEVEL)
    formatter = _build_formatter()
    run_id_filter = _RunIdFilter(run_paths.run_id)

    run_handler = logging.FileHandler(run_paths.run_log_path, mode="w", encoding="utf-8")
    run_handler._paper2slides_run_handler = True  # type: ignore[attr-defined]
    run_handler.addFilter(run_id_filter)
    run_handler.setFormatter(formatter)
    run_handler.setLevel(level)
    base_logger.addHandler(run_handler)

    error_handler = logging.FileHandler(run_paths.error_log_path, mode="w", encoding="utf-8")
    error_handler._paper2slides_run_handler = True  # type: ignore[attr-defined]
    error_handler.addFilter(run_id_filter)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    base_logger.addHandler(error_handler)

    return RunLogPaths(
        run_log_path=run_paths.run_log_path,
        error_log_path=run_paths.error_log_path,
    )


@contextlib.contextmanager
def run_logging_context(
    run_paths: RunPaths,
    log_level: str | int | None = None,
    capture_stdio: bool = True,
):
    base_logger = configure_logging(log_level=log_level, run_paths=run_paths)
    stdout_logger = get_logger(f"{LOGGER_NAMESPACE}.stdout")
    stderr_logger = get_logger(f"{LOGGER_NAMESPACE}.stderr")
    token = _CURRENT_RUN_ID.set(run_paths.run_id)
    stdout_bridge = _StdStreamBridge(stdout_logger, logging.INFO)
    stderr_bridge = _StdStreamBridge(stderr_logger, logging.ERROR)
    try:
        if capture_stdio:
            with contextlib.redirect_stdout(stdout_bridge), contextlib.redirect_stderr(stderr_bridge):
                yield base_logger
        else:
            yield base_logger
    finally:
        stdout_bridge.flush()
        stderr_bridge.flush()
        _CURRENT_RUN_ID.reset(token)
        for handler in list(base_logger.handlers):
            if getattr(handler, "_paper2slides_run_handler", False):
                base_logger.removeHandler(handler)
                handler.close()


def make_log_print(
    logger_or_name: logging.Logger | str,
    default_level: int = logging.INFO,
):
    logger = logger_or_name if isinstance(logger_or_name, logging.Logger) else get_logger(logger_or_name)

    def _printer(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
        if file not in {None, sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__}:
            builtins.print(*args, sep=sep, end=end, file=file, flush=flush)
            return
        message = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            message += end.rstrip("\n")
        if not message.strip():
            return
        for line in message.splitlines():
            text = line.strip()
            if not text:
                continue
            lower = text.lower()
            if "warning" in lower:
                level = logging.WARNING
            elif "exception" in lower or "traceback" in lower or " error" in lower or "failed" in lower:
                level = logging.ERROR
            else:
                level = default_level
            logger.log(level, text)
        if flush:
            for handler in logger.handlers:
                handler.flush()

    return _printer
