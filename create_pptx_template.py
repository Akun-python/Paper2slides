from pathlib import Path

try:
    from pptx import Presentation  # type: ignore[import]
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "python-pptx is not installed. Please install it first, for example: 'pip install python-pptx'"
    ) from exc


def create_template(output_path: str = "template.pptx") -> str:
    """Create a minimal PPTX template suitable for PptxWriterTool.

    The resulting file will:
    - Use the default PowerPoint template from python-pptx.
    - Provide layout 0 as a title slide (title + optional subtitle placeholder).
    - Provide layout 1 as a title-and-content slide (title + content placeholder).

    You can open the generated template in PowerPoint and customize:
    - Background, colors, and fonts via Slide Master.
    - Logos and footers on the master/individual layouts.
    - Animations attached to layouts/placeholders.
    """

    out = Path(output_path).expanduser().resolve()
    prs = Presentation()

    # Ensure that there is at least a title slide layout (index 0)
    # and a title-and-content layout (index 1). The default template
    # provided by python-pptx already satisfies this, so we do not
    # need to modify layouts here.

    prs.save(str(out))
    return str(out)


if __name__ == "__main__":
    path = create_template()
    print(f"Template PPTX created at: {path}")
