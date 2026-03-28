from __future__ import annotations

import argparse
import shutil
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


def _remove_all_slides(prs: Presentation) -> None:
    """Remove all slides from a Presentation, keeping masters/layouts."""
    slide_id_list = prs.slides._sldIdLst  # type: ignore[attr-defined]
    for index in range(len(slide_id_list) - 1, -1, -1):
        slide_id = slide_id_list[index]
        prs.part.drop_rel(slide_id.rId)
        del slide_id_list[index]


def convert_to_template(src: str, dst: str | None = None) -> str:
    """Convert an existing PPTX into a reusable template-like PPTX."""
    src_path = Path(src).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"source PPTX not found: {src_path}")

    dst_path = (
        Path(dst).expanduser().resolve()
        if dst is not None
        else src_path.with_name(f"{src_path.stem}_template.pptx")
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_path, dst_path)

    prs = Presentation(str(dst_path))
    _remove_all_slides(prs)
    prs.save(str(dst_path))
    return str(dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an existing PPTX deck into a template-like PPTX.")
    parser.add_argument("src", help="Path to the source PPTX.")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output PPTX path. Defaults to '<src_stem>_template.pptx' next to the source.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity for this helper script.",
    )
    args = parser.parse_args()

    configure_logging(log_level=args.log_level)
    out_path = convert_to_template(args.src, args.output)
    logger.info("Template PPTX created at: %s", out_path)


if __name__ == "__main__":
    main()
