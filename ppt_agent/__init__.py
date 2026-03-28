"""ppt_agent package.

This package contains tools, agent construction, and CLI helpers
for generating LaTeX Beamer PPT files from a paper PDF and figures.
"""

__all__ = ["build_agent", "build_model"]


def build_agent(*args, **kwargs):
    from .agent import build_agent as _build_agent

    return _build_agent(*args, **kwargs)


def build_model(*args, **kwargs):
    from .agent import build_model as _build_model

    return _build_model(*args, **kwargs)
