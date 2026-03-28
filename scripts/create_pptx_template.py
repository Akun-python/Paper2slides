from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppt_agent.logging_utils import configure_logging, get_logger

try:
    from pptx import Presentation  # type: ignore[import]
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "python-pptx is not installed. Please install it first, for example: 'pip install python-pptx'"
    ) from exc


logger = get_logger("paper2slides.tools")


def create_template(output_path: str = "template.pptx") -> str:
    """Create a minimal PPTX template suitable for PptxWriterTool."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.save(str(out))
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal PPTX template for Paper2slides.")
    parser.add_argument(
        "-o",
        "--output",
        default="template.pptx",
        help="Output PPTX path. Defaults to ./template.pptx",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity for this helper script.",
    )
    args = parser.parse_args()

    configure_logging(log_level=args.log_level)
    path = create_template(args.output)
    logger.info("Template PPTX created at: %s", path)


if __name__ == "__main__":
    main()
