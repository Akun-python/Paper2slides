from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from pptx import Presentation  # type: ignore[import]
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "python-pptx is not installed. Please install it first, for example: 'pip install python-pptx'"
    ) from exc


def _remove_all_slides(prs: Presentation) -> None:
    """Remove all slides from a Presentation, keeping masters/layouts.

    python-pptx does not expose an official public API to delete slides, so
    we use the standard workaround of manipulating the underlying XML.
    This is safe for the purpose of turning a finished deck into a template.
    """

    slide_id_list = prs.slides._sldIdLst  # type: ignore[attr-defined]
    # Remove slides from back to front, deleting both the slide XML element
    # and its relationship using the python-pptx "drop_rel" helper.
    for i in range(len(slide_id_list) - 1, -1, -1):
        slide_id = slide_id_list[i]
        rId = slide_id.rId
        prs.part.drop_rel(rId)
        del slide_id_list[i]


def convert_to_template(src: str, dst: str | None = None) -> str:
    """Convert an existing PPTX into a template-like PPTX.

    Steps:
    - Copy the original PPTX to a new path (dst).
    - Open the copy, remove all normal slides, keep masters/layouts.
    - Save the result. This file is suitable for use as --pptx-template.
    """

    src_path = Path(src).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"source PPTX not found: {src_path}")

    if dst is None:
        dst_path = src_path.with_name(src_path.stem + "_template.pptx")
    else:
        dst_path = Path(dst).expanduser().resolve()

    # Copy first so we never touch the original file.
    shutil.copy2(src_path, dst_path)

    prs = Presentation(str(dst_path))

    # Remove all existing slides; masters/layouts remain intact.
    _remove_all_slides(prs)

    prs.save(str(dst_path))
    return str(dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an existing PPTX deck into a template-like PPTX.")
    parser.add_argument(
        "src",
        help="Path to the source PPTX (e.g. '蓝染色-非遗赛道.pptx').",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output PPTX path. Defaults to '<src_stem>_template.pptx' next to the source.",
    )
    args = parser.parse_args()

    out_path = convert_to_template(args.src, args.output)
    print(f"Template PPTX created at: {out_path}")


if __name__ == "__main__":
    main()
