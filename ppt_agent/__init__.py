"""ppt_agent package.

This package contains tools, agent construction, and CLI helpers
for generating LaTeX Beamer PPT files from a paper PDF and figures.
"""

from .agent import build_agent, build_model

__all__ = ["build_agent", "build_model"]
