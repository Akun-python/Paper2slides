from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage

from ppt_agent.agent import build_model, build_pipeline_agents, build_agent
from ppt_agent.tools import (
    pdf_loader_tool,
    figure_indexer_tool,
    pdf_figure_extract_tool,
    figure_caption_align_tool,
    latex_writer_tool,
    compile_check_tool,
    ppt_slide_export_tool,
    pptx_writer_tool,
    pptx_com_writer_tool,
    analyze_image_with_vision,
    get_image_dimensions,
    load_text_chunks_with_fallback,
    evaluate_text_quality,
)
from ppt_agent.rag import build_keyword_index, keyword_search

DEFAULT_MAX_FRAMES = 88
REPORT_TYPE_GENERAL = "general"
REPORT_TYPE_ZHENGDA = "zhengda_cup"

ZHENGDA_TEMPLATE_SECTIONS_ZH = [
    "背景分析",
    "文献综述",
    "调查框架设计",
    "访谈分析",
    "网络舆情分析",
    "问卷分析",
]

ZHENGDA_TEMPLATE_SECTIONS_EN = [
    "Background Analysis",
    "Literature Review",
    "Survey Framework Design",
    "Interview Analysis",
    "Online Public Opinion Analysis",
    "Questionnaire Analysis",
]

ZHENGDA_MODEL_ALIASES_ZH = [
    ("结构方程模型", ("结构方程", "sem")),
    ("有序Logit模型", ("有序logit", "ordered logit", "ologit")),
    ("Logit模型", (" logit", "logit ", "logit模型", "二元logit")),
    ("Probit模型", ("probit", "ordered probit", "有序probit")),
    ("多元回归模型", ("多元回归", "线性回归", "最小二乘", "ols")),
    ("有序逻辑回归模型", ("有序逻辑回归", "ordinal logistic", "ordered logistic")),
    ("决策树模型", ("决策树", "cart")),
    ("随机森林模型", ("随机森林", "random forest")),
    ("XGBoost模型", ("xgboost", "梯度提升树", "gbdt")),
    ("因子分析模型", ("因子分析", "主成分分析", "pca")),
    ("聚类模型", ("聚类", "k-means", "kmeans", "k-modes", "kmodes")),
    ("中介效应模型", ("中介效应", "bootstrap", "sobel")),
    ("调节效应模型", ("调节效应", "交互项")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Beamer PPT from a research PDF")
    parser.add_argument("--pdf", help="path to input PDF file")
    parser.add_argument("--figures", help="root directory for figures (will contain per-project subfolders)")
    parser.add_argument("--template", default="ppt_template.tex", help="LaTeX template path")
    parser.add_argument("--output", default="ppt论文汇报_agent.tex", help="output LaTeX file name")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="root directory for per-run outputs (each run creates a subfolder)",
    )
    parser.add_argument(
        "--run-id",
        help="optional run identifier for output subfolder (defaults to timestamp)",
    )
    parser.add_argument(
        "--export-images",
        action="store_true",
        help="export each PPT page as image after compile",
    )
    parser.add_argument("--slides-dir", default="export/slides", help="directory for exported slide images")
    parser.add_argument("--title", help="override LaTeX title")
    parser.add_argument("--subtitle", help="override LaTeX subtitle")
    parser.add_argument("--author", help="override LaTeX author")
    parser.add_argument("--institute", help="override LaTeX institute")
    parser.add_argument("--date", help="override LaTeX date")
    parser.add_argument("--lang", choices=["zh", "en"], help="PPT language: zh or en")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "controller"],
        default="pipeline",
        help="Execution mode: 'pipeline' runs the deterministic CLI pipeline; 'controller' delegates the full PDF->PPT workflow to a ReAct controller agent.",
    )
    parser.add_argument(
        "--pptx-template",
        help=(
            "optional .pptx template file used by PptxWriterTool for visual style, background, logo, and animations; "
            "when provided, the generator will only fill in placeholders and will not override fonts, colors, or add custom borders."
        ),
    )
    parser.add_argument(
        "--pptx-theme-palette",
        help="optional comma-separated hex colors for PPTX theme styling (e.g. '#34dbcb,#3498db')",
    )
    parser.add_argument(
        "--pptx-aspect-ratio",
        choices=["16:9", "16:10", "16:8", "4:3", "1:1", "auto"],
        help="pptx slide aspect ratio (e.g. 16:9, 4:3, 1:1). Use 'auto' to keep template/default.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="maximum number of content frames (excluding title and TOC)",
    )
    parser.add_argument(
        "--max-frames-per-section",
        type=int,
        help="soft cap for frames per section/subsection (adaptive planner, suggested 4-10)",
    )
    parser.add_argument(
        "--detail-profile",
        choices=["compact", "standard", "detailed"],
        help="content detail profile; defaults to detailed",
    )
    parser.add_argument(
        "--report-type",
        choices=[REPORT_TYPE_GENERAL, REPORT_TYPE_ZHENGDA],
        help="report template type; use 'zhengda_cup' to enforce Zhengda Cup chapter template",
    )
    parser.add_argument(
        "--target-content-pages-min",
        type=int,
        help="target minimum number of content pages (excluding cover/toc/thanks)",
    )
    parser.add_argument(
        "--target-content-pages-max",
        type=int,
        help="target maximum number of content pages (excluding cover/toc/thanks)",
    )
    parser.add_argument(
        "--mojibake-policy",
        choices=["auto_fallback", "strict_fail", "warn_only"],
        help="how to handle low-quality/garbled text extraction",
    )
    parser.add_argument(
        "--preferred-text-source",
        help="optional preferred text source file path (e.g., companion .docx)",
    )
    parser.add_argument(
        "--enable-ocr-fallback",
        action="store_true",
        help="force enable OCR fallback for text extraction",
    )
    parser.add_argument(
        "--disable-ocr-fallback",
        action="store_true",
        help="disable OCR fallback for text extraction",
    )
    parser.add_argument(
        "--text-quality-debug",
        action="store_true",
        help="print detailed text quality diagnostics",
    )
    parser.add_argument(
        "--auto-fix-latex",
        action="store_true",
        help="use LLM to automatically fix LaTeX compile errors (may rewrite the .tex file)",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=2,
        help="maximum number of auto-fix attempts when --auto-fix-latex is enabled",
    )
    return parser.parse_args()


def _slugify_project_name(name: str) -> str:
    """Create an ASCII-safe slug for directory names derived from the PDF stem.

    This avoids issues where LaTeX \includegraphics cannot handle non-ASCII
    characters (such as Chinese) in file or directory names.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    slug = slug.strip("_")
    return slug or "project"


def _sanitize_filename(filename: str, fallback_stem: str = "figure") -> str:
    """Create an ASCII-safe filename while preserving the original extension."""
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    if not safe_stem:
        safe_stem = fallback_stem
    return f"{safe_stem}{suffix}"


def _in_reference_section(section_name: str | None) -> bool:
    if not section_name:
        return False
    name = section_name.strip().lower()
    if not name:
        return False
    if name in {"references", "reference", "bibliography"}:
        return True
    return "参考文献" in section_name or "文献" in section_name


def _in_back_matter_section(section_name: str | None) -> bool:
    if not section_name:
        return False
    name = section_name.strip().lower()
    if not name:
        return False
    if _in_reference_section(section_name):
        return True
    if name in {"appendix", "appendices", "supplementary", "supplementary material"}:
        return True
    if name in {"acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments"}:
        return True
    if "附录" in section_name or "附件" in section_name:
        return True
    if "致谢" in section_name or "鸣谢" in section_name:
        return True
    return False


def _build_section_blocks(pdf_chunks):
    """Build coarse section/subsection blocks from PDF text chunks.

    Each block corresponds to a contiguous span of text under a logical
    section/subsection, and contains:
      - section: top-level section title (string)
      - subsection: subsection title or None
      - text: concatenated raw text for this block
      - pages: sorted list of page numbers covered by this block
      - chunk_indices: indices of pdf_chunks contributing to this block
    """
    blocks = []
    current_section = None
    current_subsection = None
    current_lines: list[str] = []
    current_pages: set[int] = set()
    current_chunk_indices: list[int] = []

    # Heading pattern like "1 Introduction", "2.1 Background", etc.
    heading_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$")
    chapter_heading_pattern = re.compile(
        r"^\s*第([一二三四五六七八九十百千万0-9]+)(章|节|部分)\s*[:：.、]?\s*(.+)?$"
    )
    zh_top_heading_pattern = re.compile(r"^\s*([一二三四五六七八九十百千万]+)[、.．]\s*(.+)$")
    zh_sub_heading_pattern = re.compile(r"^\s*[（(]([一二三四五六七八九十百千万]+)[)）]\s*(.+)$")
    simple_en_heading_pattern = re.compile(
        r"^\s*(abstract|introduction|background|related work|literature review|method[s]?|model[s]?|experiments?|results?|conclusion[s]?|discussion|references|acknowledg(?:ements?|ments?))\s*$",
        re.IGNORECASE,
    )
    simple_zh_heading_pattern = re.compile(
        r"^\s*(摘要|引言|绪论|相关工作|研究背景|研究方法|方法|模型|实验|实验结果|结果分析|讨论|结论|总结|总结与展望|展望|参考文献|致谢)\s*$"
    )

    def _clean_subsection_title(text: str | None) -> str | None:
        t = (text or "").strip()
        if not t:
            return None
        # Numeric fragments like "0.17" or "(1.2)" are often false subsection
        # detections from equations/tables and should not become headings.
        if re.fullmatch(r"[\d\.\-/\\%+()（）]+", t):
            return None
        digit_count = sum(ch.isdigit() for ch in t)
        letter_like_count = sum(ch.isalpha() or ("\u4e00" <= ch <= "\u9fff") for ch in t)
        if len(t) <= 12 and digit_count >= 2 and letter_like_count <= 1:
            return None
        return t

    def _clean_section_title(text: str | None) -> str | None:
        t = (text or "").strip()
        if not t:
            return None
        if re.fullmatch(r"[\d\.\-/\\%+()（）]+", t):
            return None
        return t

    def _looks_like_heading(text: str) -> bool:
        if not text:
            return False
        t = text.strip()
        if len(t) > 80:
            return False
        if re.fullmatch(r"[\d\.\-/\\%+()（）]+", t):
            return False
        if re.search(r"\b(19|20)\d{2}\b", t):
            return False
        if re.search(r"\bdoi\b|\bhttp\b|\bwww\b", t, flags=re.IGNORECASE):
            return False
        comma_count = t.count(",") + t.count("，")
        semi_count = t.count(";") + t.count("；")
        if comma_count >= 3 or semi_count >= 2:
            return False
        digit_count = sum(ch.isdigit() for ch in t)
        if digit_count >= max(3, int(len(t) * 0.35)):
            return False
        letter_like_count = sum(ch.isalpha() or ("\u4e00" <= ch <= "\u9fff") for ch in t)
        if len(t) <= 12 and digit_count >= 2 and letter_like_count <= 1:
            return False
        return True

    def _strip_toc_suffix(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return t
        t = re.sub(r"[\.。·…]{2,}\s*\d+\s*$", "", t).strip()
        return t

    def flush_block() -> None:
        nonlocal current_lines, current_pages, current_chunk_indices
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            current_pages = set()
            current_chunk_indices = []
            return
        section_name = _clean_section_title(current_section) or "Paper Content"
        subsection_name = _clean_subsection_title(current_subsection)
        block = {
            "section": section_name,
            "subsection": subsection_name,
            "text": text,
            "pages": sorted(current_pages) if current_pages else [],
            "chunk_indices": list(current_chunk_indices),
        }
        blocks.append(block)
        current_lines = []
        current_pages = set()
        current_chunk_indices = []

    stop_at_references = os.getenv("PIPELINE_DROP_REFERENCES", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    drop_back_matter = os.getenv("PIPELINE_DROP_BACK_MATTER", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    for idx, chunk in enumerate(pdf_chunks or []):
        text = (chunk or {}).get("page_content") or ""
        metadata = (chunk or {}).get("metadata") or {}
        page = metadata.get("page")
        if not text:
            continue
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if _is_toc_like_line(line):
                continue
            heading_line = _strip_toc_suffix(line)

            m_chapter = chapter_heading_pattern.match(heading_line)
            if m_chapter:
                flush_block()
                chapter_title = (m_chapter.group(3) or "").strip()
                if not chapter_title:
                    chapter_title = f"第{m_chapter.group(1)}{m_chapter.group(2)}"
                if not _looks_like_heading(chapter_title):
                    if line.strip():
                        current_lines.append(line)
                        if page is not None:
                            current_pages.add(page)
                        if idx not in current_chunk_indices:
                            current_chunk_indices.append(idx)
                    continue
                current_section = _clean_section_title(chapter_title) or "Paper Content"
                current_subsection = None
                if stop_at_references and _in_reference_section(current_section):
                    flush_block()
                    return blocks
                if drop_back_matter and _in_back_matter_section(current_section):
                    flush_block()
                    return blocks
                continue

            m_zh_top = zh_top_heading_pattern.match(heading_line)
            if m_zh_top:
                flush_block()
                title_text = (m_zh_top.group(2) or "").strip()
                if not _looks_like_heading(title_text):
                    if line.strip():
                        current_lines.append(line)
                        if page is not None:
                            current_pages.add(page)
                        if idx not in current_chunk_indices:
                            current_chunk_indices.append(idx)
                    continue
                current_section = _clean_section_title(title_text) or "Paper Content"
                current_subsection = None
                if stop_at_references and _in_reference_section(current_section):
                    flush_block()
                    return blocks
                if drop_back_matter and _in_back_matter_section(current_section):
                    flush_block()
                    return blocks
                continue

            m_zh_sub = zh_sub_heading_pattern.match(heading_line)
            if m_zh_sub:
                flush_block()
                title_text = (m_zh_sub.group(2) or "").strip()
                if not _looks_like_heading(title_text):
                    if line.strip():
                        current_lines.append(line)
                        if page is not None:
                            current_pages.add(page)
                        if idx not in current_chunk_indices:
                            current_chunk_indices.append(idx)
                    continue
                cleaned_sub = _clean_subsection_title(title_text)
                if cleaned_sub:
                    current_subsection = cleaned_sub
                if current_section is None:
                    current_section = "Paper Content"
                continue

            m = heading_pattern.match(heading_line)
            if m:
                if _in_reference_section(current_section):
                    # Avoid treating numbered references as new headings.
                    current_lines.append(line)
                    if page is not None:
                        current_pages.add(page)
                    if idx not in current_chunk_indices:
                        current_chunk_indices.append(idx)
                    continue
                # Flush previous accumulated text before starting a new block
                flush_block()
                number_str = m.group(1)
                title_text = m.group(2).strip()
                if not _looks_like_heading(title_text):
                    # Likely a numbered reference or list item.
                    current_lines.append(line)
                    if page is not None:
                        current_pages.add(page)
                    if idx not in current_chunk_indices:
                        current_chunk_indices.append(idx)
                    continue
                if "." in number_str:
                    # Subsection heading
                    cleaned_sub = _clean_subsection_title(title_text)
                    if cleaned_sub:
                        current_subsection = cleaned_sub
                    else:
                        # Treat noisy numeric-like pseudo headings as normal text.
                        current_lines.append(line)
                        if page is not None:
                            current_pages.add(page)
                        if idx not in current_chunk_indices:
                            current_chunk_indices.append(idx)
                        continue
                    if current_section is None:
                        current_section = _clean_section_title(title_text) or "Paper Content"
                else:
                    # New top-level section
                    current_section = _clean_section_title(title_text) or "Paper Content"
                    current_subsection = None
                    if stop_at_references and _in_reference_section(current_section):
                        # Stop collecting further blocks once references begin.
                        flush_block()
                        return blocks
                    if drop_back_matter and _in_back_matter_section(current_section):
                        flush_block()
                        return blocks
                # Do not include heading line as content
                continue
            stripped = line.strip()
            if stripped:
                stripped_for_heading = _strip_toc_suffix(stripped)
                if simple_en_heading_pattern.match(stripped_for_heading) or simple_zh_heading_pattern.match(
                    stripped_for_heading
                ):
                    flush_block()
                    current_section = _clean_section_title(stripped_for_heading) or "Paper Content"
                    current_subsection = None
                    if stop_at_references and _in_reference_section(current_section):
                        flush_block()
                        return blocks
                    if drop_back_matter and _in_back_matter_section(current_section):
                        flush_block()
                        return blocks
                    if page is not None:
                        current_pages.add(page)
                    if idx not in current_chunk_indices:
                        current_chunk_indices.append(idx)
                    continue
            # Normal content line
            if line.strip():
                current_lines.append(line)
                if page is not None:
                    current_pages.add(page)
                if idx not in current_chunk_indices:
                    current_chunk_indices.append(idx)

    # Flush the final block
    flush_block()

    # Merge adjacent small blocks under the same logical section/subsection to
    # avoid over-fragmented planning requests while preserving document order.
    if blocks:
        merged_blocks: list[dict] = []
        for block in blocks:
            sec = (block.get("section") or "Paper Content").strip() or "Paper Content"
            sub = (block.get("subsection") or "").strip() or None
            text = (block.get("text") or "").strip()
            pages = list(block.get("pages") or [])
            chunk_indices = list(block.get("chunk_indices") or [])
            if not text:
                continue
            if merged_blocks:
                prev = merged_blocks[-1]
                prev_sec = (prev.get("section") or "Paper Content").strip() or "Paper Content"
                prev_sub = (prev.get("subsection") or "").strip() or None
                prev_text = (prev.get("text") or "").strip()
                same_group = prev_sec == sec and prev_sub == sub
                should_merge = same_group and (len(prev_text) < 1200 or len(text) < 320)
                if should_merge:
                    prev["text"] = (prev_text + "\n" + text).strip()
                    prev_pages = set(prev.get("pages") or [])
                    prev_pages.update(pages)
                    prev["pages"] = sorted(prev_pages)
                    prev_chunks = set(prev.get("chunk_indices") or [])
                    prev_chunks.update(chunk_indices)
                    prev["chunk_indices"] = sorted(prev_chunks)
                    continue
            merged_blocks.append(
                {
                    "section": sec,
                    "subsection": sub,
                    "text": text,
                    "pages": sorted(set(pages)),
                    "chunk_indices": sorted(set(chunk_indices)),
                }
            )
        if len(merged_blocks) != len(blocks):
            print(
                f"[Pipeline] Merged adjacent section blocks: {len(blocks)} -> {len(merged_blocks)}"
            )
        blocks = merged_blocks

    # Fallback: if we still have no blocks but there are chunks with text,
    # create a single block covering the whole document.
    if not blocks and pdf_chunks:
        all_text_parts: list[str] = []
        pages: set[int] = set()
        indices: list[int] = []
        for idx, chunk in enumerate(pdf_chunks):
            text = (chunk or {}).get("page_content") or ""
            metadata = (chunk or {}).get("metadata") or {}
            page = metadata.get("page")
            if text.strip():
                all_text_parts.append(text)
                if page is not None:
                    pages.add(page)
                indices.append(idx)
        if all_text_parts:
            blocks.append(
                {
                    "section": "Paper Content",
                    "subsection": None,
                    "text": "\n".join(all_text_parts),
                    "pages": sorted(pages),
                    "chunk_indices": indices,
                }
            )

    print(f"[Pipeline] Built {len(blocks)} section blocks from {len(pdf_chunks or [])} chunks")
    max_blocks_raw = os.getenv("PIPELINE_MAX_SECTION_BLOCKS", "80").strip()
    try:
        max_blocks = max(1, int(max_blocks_raw))
    except ValueError:
        max_blocks = 80
    if len(blocks) > max_blocks:
        print(
            f"[Pipeline] Too many section blocks ({len(blocks)}); merging to {max_blocks} blocks."
        )
        merged: list[dict] = []
        current = None
        target_size = max(1, len(blocks) // max_blocks)
        for block in blocks:
            if current is None:
                current = {
                    "section": block.get("section"),
                    "subsection": block.get("subsection"),
                    "text": block.get("text") or "",
                    "pages": list(block.get("pages") or []),
                    "chunk_indices": list(block.get("chunk_indices") or []),
                    "_count": 1,
                }
                continue
            current["_count"] += 1
            current["text"] += "\n" + (block.get("text") or "")
            for p in block.get("pages") or []:
                if p not in current["pages"]:
                    current["pages"].append(p)
            for ci in block.get("chunk_indices") or []:
                if ci not in current["chunk_indices"]:
                    current["chunk_indices"].append(ci)
            if current["_count"] >= target_size:
                current.pop("_count", None)
                merged.append(current)
                current = None
        if current is not None:
            current.pop("_count", None)
            merged.append(current)
        blocks = merged

    return blocks


def _normalize_section_titles_with_llm(model, instructions: dict, section_blocks: list[dict]):
    """Use the LLM to normalize noisy section titles based on their content.

    The heuristic section builder often produces titles like "3" or author
    names. This helper sends a compact summary of all distinct section titles
    together with short text samples to the LLM and asks it to map each
    original title to a cleaner, paper-aligned section title (in the target
    language). The mapping is then applied in-place to section_blocks.
    """
    lang = (instructions.get("lang") or "zh").lower()

    # Collect distinct section titles with a short text sample and pages.
    records: list[dict] = []
    seen: set[str] = set()
    for block in section_blocks or []:
        section = (block.get("section") or "").strip()
        if not section or section in seen:
            continue
        seen.add(section)
        text = (block.get("text") or "").strip()
        sample = text[:600]
        pages = block.get("pages") or []
        records.append(
            {
                "original_section": section,
                "sample_text": sample,
                "pages": pages,
            }
        )

    if not records:
        return section_blocks

    batch_raw = os.getenv("PIPELINE_SECTION_NORMALIZATION_BATCH_SIZE", "20").strip()
    try:
        batch_size = max(1, int(batch_raw))
    except ValueError:
        batch_size = 20

    print(
        f"[Pipeline] Normalizing {len(records)} distinct section titles with LLM "
        f"(batch_size={batch_size})..."
    )

    timeout_raw = os.getenv("PIPELINE_SECTION_TITLE_TIMEOUT", "180").strip()
    try:
        timeout_s = int(timeout_raw)
    except ValueError:
        timeout_s = 180

    mapping_dict: dict[str, str] = {}
    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        print(
            f"[Pipeline] Section title normalization progress: {min(start + batch_size, total)}/{total} titles"
        )
        payload = {
            "task": "normalize_section_titles",
            "lang": lang,
            "sections": batch,
        }
        try:
            if timeout_s > 0:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        model.invoke,
                        {
                            "messages": [
                                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                            ]
                        },
                    )
                    response = future.result(timeout=timeout_s)
            else:
                response = model.invoke(
                    {
                        "messages": [
                            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                        ]
                    }
                )
        except TimeoutError:
            print(
                f"[Pipeline] Section title normalization batch timed out after {timeout_s}s; keep originals for this batch."
            )
            continue
        except Exception as exc:  # noqa: BLE001
            print(
                f"[Pipeline] Section title normalization LLM call failed for batch; keep originals: {exc}"
            )
            continue

        # Prefer structured_output when the agent is configured with ToolStrategy,
        # but fall back to manual JSON parsing if it is not available.
        obj = None
        if isinstance(response, dict):
            sr = response.get("structured_response")
            if sr is not None:
                try:
                    if hasattr(sr, "model_dump"):
                        obj = sr.model_dump()
                    elif hasattr(sr, "dict"):
                        obj = sr.dict()
                    elif isinstance(sr, dict):
                        obj = sr
                except Exception:  # noqa: BLE001
                    obj = None

        if obj is None:
            raw = ""
            if isinstance(response, dict):
                messages = response.get("messages") or []
                if messages:
                    last = messages[-1]
                    raw = (getattr(last, "content", None) or "").strip()
            else:
                raw = (getattr(response, "content", None) or "").strip()
            if not raw:
                continue
            cleaned = raw
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1].strip()
            if cleaned and not cleaned.lstrip().startswith("{"):
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned = cleaned[start_idx : end_idx + 1]

            cleaned = _normalize_newlines_inside_strings(cleaned)

            try:
                obj = json.loads(cleaned)
            except Exception as exc:  # noqa: BLE001
                print(f"[Pipeline] Failed to parse section title normalization JSON: {exc}")
                continue

        mappings = obj.get("normalized_sections") or []
        if not isinstance(mappings, list):
            continue

        for item in mappings:
            if not isinstance(item, dict):
                continue
            orig = (item.get("original_section") or "").strip()
            norm = (item.get("normalized_section") or "").strip()
            if orig and norm:
                mapping_dict[orig] = norm

    if not mapping_dict:
        return section_blocks

    changed = 0
    for block in section_blocks:
        section = (block.get("section") or "").strip()
        new_title = mapping_dict.get(section)
        if new_title and new_title != section:
            block["section"] = new_title
            changed += 1

    print(
        f"[Pipeline] Normalized section titles for {changed} blocks using LLM (distinct originals={len(mapping_dict)})"
    )
    return section_blocks


def _limit_top_level_sections(section_blocks: list[dict], max_sections: int = 9) -> list[dict]:
    """Ensure the number of distinct top-level sections does not exceed max_sections.

    This helper walks the section_blocks in order and builds a remapping from
    original section titles to at most max_sections normalized titles. The
    first max_sections distinct titles are kept as-is; any subsequent new
    titles are merged into the most recently seen section so that the overall
    Beamer outline (\section{...}) has a bounded size while preserving
    document order.
    """

    if not section_blocks:
        return section_blocks

    seen_order: list[str] = []
    remap: dict[str, str] = {}

    for block in section_blocks:
        orig = (block.get("section") or "Paper Content").strip()
        if orig in remap:
            continue
        if len(seen_order) < max_sections:
            seen_order.append(orig)
            remap[orig] = orig
        else:
            # Merge any further distinct sections into the last kept section so
            # that we do not exceed max_sections in the final outline.
            target = seen_order[-1]
            remap[orig] = target

    # Apply the remap to normalize or merge section titles.
    for block in section_blocks:
        sec = (block.get("section") or "Paper Content").strip()
        block["section"] = remap.get(sec, sec)
    return section_blocks


def _filter_out_reference_blocks(section_blocks: list[dict]) -> list[dict]:
    if not section_blocks:
        return section_blocks
    drop_back_matter = os.getenv("PIPELINE_DROP_BACK_MATTER", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    filtered: list[dict] = []
    for block in section_blocks:
        sec = (block.get("section") or "").strip()
        sub = (block.get("subsection") or "").strip()
        if _in_reference_section(sec) or _in_reference_section(sub):
            continue
        if drop_back_matter and (_in_back_matter_section(sec) or _in_back_matter_section(sub)):
            continue
        filtered.append(block)
    return filtered


def _is_toc_section_title(section_name: str | None) -> bool:
    s = (section_name or "").strip().lower()
    if not s:
        return False
    if s in {"目录", "目次", "contents", "table of contents", "toc"}:
        return True
    if s.startswith("目录"):
        return True
    if "table of contents" in s:
        return True
    return False


def _is_toc_like_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if re.search(r"[\.。·…]{4,}", s):
        return True
    if re.search(r"(?:[\.。·…]{2,}\s*\d+\s*$)", s):
        return True
    if re.match(r"^\s*(?:\d+(?:\.\d+)*|[一二三四五六七八九十]+[、.]).{0,90}\s+\d+\s*$", s):
        return True
    return False


def _strip_toc_like_lines(text: str) -> tuple[str, int]:
    lines = (text or "").splitlines()
    kept: list[str] = []
    dropped = 0
    for line in lines:
        if _is_toc_like_line(line):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept).strip(), dropped


def _is_toc_like_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return False
    dotted = sum(1 for ln in lines if re.search(r"[\.。·…]{4,}", ln))
    trailing_page = sum(1 for ln in lines if re.search(r"(?:[\.。·…]{2,}\s*\d+\s*$|\s+\d+\s*$)", ln))
    indexed = sum(
        1
        for ln in lines
        if re.match(r"^\s*(?:\d+(?:\.\d+)*|[一二三四五六七八九十]+[、.])", ln)
    )
    head = lines[0].lower()
    has_toc_head = "目录" in lines[0] or "contents" in head or "table of contents" in head
    if dotted >= 3 or trailing_page >= 4:
        return True
    if len(lines) >= 10 and indexed >= 5 and (has_toc_head or dotted >= 2):
        return True
    return False


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    hay = (text or "").lower()
    return any(k.lower() in hay for k in keywords)


def _extract_questionnaire_model_label(section: str, subsection: str | None, text: str, lang: str = "zh") -> str | None:
    hay = "\n".join(
        part for part in [section or "", subsection or "", text or ""] if part
    ).lower()
    for label, aliases in ZHENGDA_MODEL_ALIASES_ZH:
        if _contains_any_keyword(hay, aliases):
            return label if (lang or "zh").lower().startswith("zh") else label

    m = re.search(r"(?:模型|model)\s*([一二三四五六七八九十0-9a-zA-Z]+)", hay, flags=re.IGNORECASE)
    if m:
        suffix = m.group(1).strip()
        if suffix:
            if (lang or "zh").lower().startswith("zh"):
                return f"模型{suffix.upper()}"
            return f"Model {suffix.upper()}"
    return None


def _infer_zhengda_base_section(section: str, subsection: str | None, text: str, lang: str = "zh") -> str | None:
    zh = (lang or "zh").lower().startswith("zh")
    hay = "\n".join(part for part in [section or "", subsection or "", text or ""] if part).lower()
    background = (
        "背景",
        "引言",
        "绪论",
        "问题提出",
        "研究意义",
        "行业现状",
        "选题",
    )
    literature = ("文献", "综述", "相关研究", "理论基础", "研究述评")
    framework = (
        "调查框架",
        "研究框架",
        "技术路线",
        "调查设计",
        "研究设计",
        "指标体系",
        "变量定义",
        "问卷设计",
        "样本设计",
        "抽样",
        "信度",
        "效度",
        "预调查",
        "预测试",
        "数据来源",
    )
    interview = ("访谈", "深度访谈", "扎根理论", "编码分析")
    public_opinion = ("舆情", "网络文本", "网络评论", "微博", "小红书", "贴吧", "情感分析", "话题分析")
    questionnaire = (
        "问卷",
        "survey",
        "questionnaire",
        "实证",
        "模型",
        "回归",
        "logit",
        "probit",
        "sem",
        "结构方程",
        "随机森林",
        "决策树",
        "xgboost",
        "因子分析",
        "主成分",
        "聚类",
        "卡方",
        "中介效应",
        "调节效应",
    )
    questionnaire_design_only = ("问卷设计", "量表设计", "预测试", "预调查", "信度", "效度")
    questionnaire_model_signal = (
        "模型",
        "回归",
        "logit",
        "probit",
        "sem",
        "结构方程",
        "随机森林",
        "决策树",
        "xgboost",
        "因子分析",
        "主成分",
        "聚类",
        "卡方",
        "中介效应",
        "调节效应",
    )

    if _contains_any_keyword(hay, interview):
        return "访谈分析" if zh else "Interview Analysis"
    if _contains_any_keyword(hay, public_opinion):
        return "网络舆情分析" if zh else "Online Public Opinion Analysis"
    if _contains_any_keyword(hay, literature):
        return "文献综述" if zh else "Literature Review"
    if _contains_any_keyword(hay, background):
        return "背景分析" if zh else "Background Analysis"

    has_q = _contains_any_keyword(hay, questionnaire)
    has_q_design = _contains_any_keyword(hay, questionnaire_design_only) or _contains_any_keyword(hay, framework)
    has_q_model = _contains_any_keyword(hay, questionnaire_model_signal)
    if has_q and has_q_model:
        return "问卷分析" if zh else "Questionnaire Analysis"
    if has_q and has_q_design and not has_q_model:
        return "调查框架设计" if zh else "Survey Framework Design"
    if _contains_any_keyword(hay, framework):
        return "调查框架设计" if zh else "Survey Framework Design"
    if has_q:
        return "问卷分析" if zh else "Questionnaire Analysis"
    return None


def _apply_zhengda_template_to_section_blocks(section_blocks: list[dict], lang: str = "zh") -> list[dict]:
    if not section_blocks:
        return section_blocks
    lang_norm = (lang or "zh").lower()
    zh = lang_norm.startswith("zh")
    base_sections = _zhengda_template_sections(lang_norm)
    questionnaire_base = base_sections[-1]
    default_section = base_sections[2] if len(base_sections) >= 3 else questionnaire_base

    normalized: list[dict] = []
    remapped = 0
    last_sec = base_sections[0]
    model_section_rank: dict[str, int] = {}

    for idx, block in enumerate(section_blocks):
        if not isinstance(block, dict):
            continue
        sec_raw = str(block.get("section") or "").strip()
        sub_raw = str(block.get("subsection") or "").strip() or None
        text_raw = str(block.get("text") or "").strip()
        inferred = _infer_zhengda_base_section(sec_raw, sub_raw, text_raw, lang=lang_norm)
        if inferred is None:
            inferred = last_sec if last_sec else default_section
        final_sec = inferred
        if inferred == questionnaire_base:
            model_label = _extract_questionnaire_model_label(sec_raw, sub_raw, text_raw, lang=lang_norm)
            if model_label:
                if zh:
                    final_sec = f"{questionnaire_base}：{model_label}"
                else:
                    final_sec = f"{questionnaire_base}: {model_label}"
        if final_sec not in model_section_rank and final_sec.startswith(questionnaire_base):
            model_section_rank[final_sec] = len(model_section_rank)
        if final_sec != sec_raw:
            remapped += 1
        new_block = dict(block)
        new_block["section"] = final_sec
        normalized.append(new_block)
        last_sec = inferred

    base_order = {name: idx for idx, name in enumerate(base_sections)}
    questionnaire_order = base_order.get(questionnaire_base, len(base_sections))

    def _first_page(block: dict) -> int:
        for p in block.get("pages") or []:
            if isinstance(p, int):
                return p
        return 10**6

    def _sort_key(block: dict) -> tuple[int, int, int]:
        sec = str(block.get("section") or "")
        if sec in base_order:
            return base_order[sec], _first_page(block), 0
        if sec.startswith(questionnaire_base):
            return questionnaire_order, _first_page(block), model_section_rank.get(sec, 10**5)
        return len(base_sections) + 1, _first_page(block), 0

    normalized.sort(key=_sort_key)
    if remapped:
        print(
            f"[Pipeline] Applied Zhengda Cup template remapping for {remapped} section block(s)."
        )
    return normalized


def _canonical_section_title(section: str, text: str, lang: str = "zh") -> str:
    sec = (section or "").strip()
    lang_norm = (lang or "zh").lower()
    if sec:
        sec_clean = _normalize_section_text(sec)
        sec_l = sec_clean.lower()
        generic_names = {
            "paper content",
            "content",
            "main content",
            "section",
            "sections",
            "chapter",
            "chapters",
            "正文",
            "章节",
            "研究内容",
            "研究内容分析",
            "内容概览",
        }
        if sec_clean and sec_l not in generic_names and not re.fullmatch(r"(?:section|chapter|part)\s*\d*", sec_l):
            return sec_clean
    hay = f"{sec}\n{text or ''}".lower()
    if lang_norm.startswith("zh"):
        if any(k in hay for k in ("结论", "建议", "启示", "展望", "未来工作", "政策建议")):
            return "结论与建议"
        if any(k in hay for k in ("背景", "引言", "绪论", "文献", "相关工作", "研究意义")):
            return "研究背景与问题"
        if any(k in hay for k in ("抽样", "样本", "问卷", "数据", "调查设计", "变量定义", "预处理")):
            return "数据与调查设计"
        if any(k in hay for k in ("方法", "模型", "算法", "回归", "sem", "k-modes", "随机森林", "变量筛选")):
            return "研究方法与模型"
        if any(k in hay for k in ("结果", "实验", "分析", "检验", "显著性", "对比", "评估")):
            return "结果与分析"
        if any(k in hay for k in ("讨论", "案例", "解释", "机制")):
            return "讨论与启示"
        return "研究内容分析"
    if any(k in hay for k in ("conclusion", "recommend", "future work", "implication")):
        return "Conclusion and Recommendations"
    if any(k in hay for k in ("background", "introduction", "related work", "literature")):
        return "Background and Problem"
    if any(k in hay for k in ("data", "sample", "survey", "questionnaire", "preprocess")):
        return "Data and Sampling"
    if any(k in hay for k in ("method", "model", "algorithm", "regression", "sem")):
        return "Methods and Models"
    if any(k in hay for k in ("result", "experiment", "analysis", "evaluation", "comparison")):
        return "Results and Analysis"
    if any(k in hay for k in ("discussion", "case", "interpret")):
        return "Discussion and Insights"
    return "Main Analysis"


def _normalize_section_text(section: str) -> str:
    s = (section or "").strip()
    if not s:
        return s
    s = re.sub(r"^\s*(?:第?[一二三四五六七八九十百千万0-9]+(?:章|节|部分|篇)?[、.．:：-]?)\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip("：:;；，,。.")
    return s.strip()


def _refine_section_blocks_for_outline(section_blocks: list[dict], lang: str = "zh") -> list[dict]:
    if not section_blocks:
        return section_blocks
    lang_norm = (lang or "zh").lower()
    refined: list[dict] = []
    dropped_toc_blocks = 0
    renamed_sections = 0
    dropped_toc_lines = 0
    generic_names = {
        "paper content",
        "content",
        "正文",
        "章节",
        "section",
        "main content",
        "目录",
        "contents",
        "table of contents",
        "toc",
    }

    for block in section_blocks:
        if not isinstance(block, dict):
            continue
        sec_raw = (block.get("section") or "").strip()
        sub_raw = (block.get("subsection") or "").strip()
        text_raw = (block.get("text") or "").strip()
        if not text_raw:
            continue

        cleaned_text, local_dropped = _strip_toc_like_lines(text_raw)
        dropped_toc_lines += local_dropped
        if not cleaned_text:
            cleaned_text = text_raw

        sec = _normalize_section_text(sec_raw)
        if not sec:
            sec = "Paper Content" if not lang_norm.startswith("zh") else "研究内容"

        sec_quality = evaluate_text_quality(sec, lang=lang_norm)
        looks_noisy = (
            _is_toc_section_title(sec)
            or _is_toc_like_text(cleaned_text)
            or sec.lower() in generic_names
            or sec_quality.get("is_mojibake")
            or len(sec) > 26
        )
        if _is_toc_section_title(sec) and _is_toc_like_text(cleaned_text):
            dropped_toc_blocks += 1
            continue

        sec_final = sec
        if looks_noisy:
            sec_final = _canonical_section_title(sec, cleaned_text, lang=lang_norm)
        else:
            sec_final = _canonical_section_title(sec, sec, lang=lang_norm) if sec.lower() in generic_names else sec

        if sec_final != sec_raw:
            renamed_sections += 1

        new_block = dict(block)
        new_block["section"] = sec_final
        if sub_raw and (_is_toc_section_title(sub_raw) or evaluate_text_quality(sub_raw, lang=lang_norm).get("is_mojibake")):
            new_block["subsection"] = None
        new_block["text"] = cleaned_text
        refined.append(new_block)

    if not refined:
        return section_blocks

    merged: list[dict] = []
    for block in refined:
        if (
            merged
            and (merged[-1].get("section") or "") == (block.get("section") or "")
            and (merged[-1].get("subsection") or None) == (block.get("subsection") or None)
        ):
            prev = merged[-1]
            prev_text = (prev.get("text") or "").strip()
            cur_text = (block.get("text") or "").strip()
            if cur_text:
                prev["text"] = f"{prev_text}\n{cur_text}".strip() if prev_text else cur_text
            prev_pages = set(prev.get("pages") or [])
            prev_pages.update(block.get("pages") or [])
            prev["pages"] = sorted(prev_pages)
            prev_chunks = set(prev.get("chunk_indices") or [])
            prev_chunks.update(block.get("chunk_indices") or [])
            prev["chunk_indices"] = sorted(prev_chunks)
            continue
        merged.append(block)

    if dropped_toc_blocks or renamed_sections or dropped_toc_lines or len(merged) != len(section_blocks):
        print(
            "[Pipeline] Outline refinement: "
            f"blocks {len(section_blocks)} -> {len(merged)}, "
            f"dropped_toc_blocks={dropped_toc_blocks}, "
            f"renamed_sections={renamed_sections}, "
            f"dropped_toc_lines={dropped_toc_lines}"
        )
    return merged


def _ensure_conclusion_section_block(
    section_blocks: list[dict],
    pdf_chunks: list[dict],
    lang: str = "zh",
) -> tuple[list[dict], bool]:
    if not section_blocks:
        return section_blocks, False
    lang_norm = (lang or "zh").lower()
    if lang_norm.startswith("zh"):
        conclusion_title = "结论与建议"
        keywords = ("结论", "建议", "展望", "启示", "未来工作", "政策建议")
    else:
        conclusion_title = "Conclusion and Recommendations"
        keywords = ("conclusion", "recommendation", "future work", "implication")

    for block in section_blocks:
        title = str(block.get("section") or "")
        text = str(block.get("text") or "")
        hay = f"{title}\n{text}".lower()
        if any(k.lower() in hay for k in keywords):
            return section_blocks, False

    tail_blocks = section_blocks[-2:] if len(section_blocks) >= 2 else section_blocks[-1:]
    text_parts: list[str] = []
    pages: set[int] = set()
    chunk_indices: set[int] = set()
    for block in tail_blocks:
        txt = (block.get("text") or "").strip()
        if txt:
            text_parts.append(txt[:2000])
        for p in block.get("pages") or []:
            if isinstance(p, int):
                pages.add(p)
        for ci in block.get("chunk_indices") or []:
            if isinstance(ci, int):
                chunk_indices.add(ci)

    if not text_parts and pdf_chunks:
        for idx in range(max(0, len(pdf_chunks) - 3), len(pdf_chunks)):
            chunk = pdf_chunks[idx]
            txt = (chunk or {}).get("page_content") or ""
            if txt.strip():
                text_parts.append(txt.strip()[:1200])
            meta = (chunk or {}).get("metadata") or {}
            page = meta.get("page")
            if isinstance(page, int):
                pages.add(page)
            chunk_indices.add(idx)

    synthetic_text = "\n".join(text_parts).strip()
    if not synthetic_text:
        if lang_norm.startswith("zh"):
            synthetic_text = "归纳研究发现并给出可执行建议。"
        else:
            synthetic_text = "Summarize key findings and provide actionable recommendations."

    appended = list(section_blocks)
    appended.append(
        {
            "section": conclusion_title,
            "subsection": None,
            "text": synthetic_text,
            "pages": sorted(pages),
            "chunk_indices": sorted(chunk_indices),
        }
    )
    print(f"[Pipeline] Appended synthetic section block: '{conclusion_title}'.")
    return appended, True


def _build_outline_from_section_blocks(
    section_blocks: list[dict],
    lang: str = "zh",
    report_type: str = REPORT_TYPE_GENERAL,
) -> list[str]:
    if not section_blocks:
        if _normalize_report_type(report_type) == REPORT_TYPE_ZHENGDA:
            return _zhengda_template_sections(lang=lang)
        return []
    lang_norm = (lang or "zh").lower()
    report_type_norm = _normalize_report_type(report_type)
    if report_type_norm == REPORT_TYPE_ZHENGDA:
        template_sections = _zhengda_template_sections(lang=lang_norm)
        questionnaire_base = template_sections[-1]
        model_sections: list[str] = []
        plain_questionnaire_seen = False
        for block in section_blocks:
            sec = _normalize_section_text(str(block.get("section") or ""))
            if not sec or _is_toc_section_title(sec):
                continue
            if sec == questionnaire_base:
                plain_questionnaire_seen = True
                continue
            if sec.startswith(f"{questionnaire_base}：") or sec.startswith(f"{questionnaire_base}:"):
                if sec not in model_sections:
                    model_sections.append(sec)
        outline: list[str] = []
        for sec in template_sections:
            if sec != questionnaire_base:
                outline.append(sec)
                continue
            if model_sections:
                outline.extend(model_sections)
            elif plain_questionnaire_seen:
                outline.append(questionnaire_base)
            else:
                outline.append(questionnaire_base)
        return outline

    outline: list[str] = []
    seen: set[str] = set()
    for block in section_blocks:
        sec = _normalize_section_text(str(block.get("section") or ""))
        if not sec:
            continue
        if _is_toc_section_title(sec):
            continue
        sec = _canonical_section_title(sec, str(block.get("text") or ""), lang=lang_norm)
        if sec in seen:
            continue
        seen.add(sec)
        outline.append(sec)
    return outline


def _build_global_planning_context(
    section_blocks: list[dict],
    outline_sections: list[str] | None = None,
    summary_frame: dict | None = None,
    lang: str = "zh",
    max_chars: int = 2600,
) -> str:
    lang_norm = (lang or "zh").lower()
    zh = lang_norm.startswith("zh")
    lines: list[str] = []

    if summary_frame:
        title = str(summary_frame.get("frame_title") or "").strip()
        bullets = summary_frame.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        if title:
            lines.append(f"总览：{title}" if zh else f"Overview: {title}")
        for b in bullets[:4]:
            txt = str(b).strip()
            if txt:
                lines.append(f"- {txt}")

    section_order = [s for s in (outline_sections or []) if str(s or "").strip()]
    if not section_order:
        seen_sec: set[str] = set()
        for block in section_blocks or []:
            sec = str(block.get("section") or "").strip()
            if not sec or sec in seen_sec:
                continue
            seen_sec.add(sec)
            section_order.append(sec)

    if section_order:
        if zh:
            lines.append("全文逻辑主线（按章节顺序）：")
        else:
            lines.append("Global storyline (section order):")

    for idx, sec in enumerate(section_order, start=1):
        evidence = ""
        for block in section_blocks or []:
            if str(block.get("section") or "").strip() != sec:
                continue
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            sents = _split_sentences(text[:1200], lang_norm)
            if sents:
                evidence = sents[0][:120]
            else:
                evidence = text[:120]
            break
        if evidence:
            line = f"{idx}. {sec}：{evidence}" if zh else f"{idx}. {sec}: {evidence}"
        else:
            line = f"{idx}. {sec}"
        lines.append(line)

    context = "\n".join(line for line in lines if line).strip()
    if len(context) > max_chars:
        context = context[:max_chars].rstrip() + "..."
    return context


def _ensure_outline_coverage_frames(
    frames: list[dict],
    section_blocks: list[dict],
    outline_sections: list[str],
    lang: str = "zh",
) -> tuple[list[dict], int]:
    if not frames or not outline_sections:
        return frames, 0
    present_sections = {
        _normalize_section_text(str(frame.get("section") or ""))
        for frame in frames
        if str(frame.get("section") or "").strip()
    }
    block_by_section: dict[str, dict] = {}
    for block in section_blocks or []:
        sec = _normalize_section_text(str(block.get("section") or ""))
        if sec and sec not in block_by_section:
            block_by_section[sec] = block

    injected = 0
    out = list(frames)
    for sec in outline_sections:
        if sec in present_sections:
            continue
        block = block_by_section.get(sec)
        source_text = str((block or {}).get("text") or "")
        fallback_frames = _fallback_plan_frames(
            sec,
            None,
            source_text,
            "",
            [],
            lang,
            max_frames_per_section=1,
        )
        if not fallback_frames:
            continue
        f = dict(fallback_frames[0])
        f["section"] = sec
        f["subsection"] = None
        f["frame_title"] = sec
        f["source_pages"] = list((block or {}).get("pages") or [])
        f["source_chunk_indices"] = list((block or {}).get("chunk_indices") or [])
        out.append(f)
        present_sections.add(sec)
        injected += 1
    if injected:
        print(f"[Pipeline] Injected {injected} fallback frame(s) to cover missing outline sections.")
    return out, injected


def _map_frames_to_outline_sections(
    frames: list[dict],
    section_blocks: list[dict],
    outline_sections: list[str],
    lang: str = "zh",
) -> tuple[list[dict], int]:
    if not frames:
        return frames, 0
    outline = [_normalize_section_text(s) for s in (outline_sections or []) if _normalize_section_text(s)]
    outline_set = set(outline)
    preserve_sections = {"研究概述", "Overview"}
    page_to_section: dict[int, str] = {}
    for block in section_blocks or []:
        sec = _normalize_section_text(str(block.get("section") or ""))
        if not sec:
            continue
        for p in block.get("pages") or []:
            if isinstance(p, int) and p not in page_to_section:
                page_to_section[p] = sec

    def _choose_from_pages(pages: list[int]) -> str | None:
        if not pages:
            return None
        hits: dict[str, int] = {}
        for p in pages:
            sec = page_to_section.get(p)
            if not sec:
                continue
            hits[sec] = hits.get(sec, 0) + 1
        if not hits:
            return None
        return sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def _alias_match(sec: str) -> str | None:
        s = _normalize_section_text(sec)
        if not s or not outline:
            return None
        for out_sec in outline:
            if s == out_sec:
                return out_sec
            if s in out_sec or out_sec in s:
                return out_sec
        return None

    rewritten = 0
    mapped_frames: list[dict] = []
    last_outline_section = outline[0] if outline else None
    lang_norm = (lang or "zh").lower()
    conclusion_section = "结论与建议" if lang_norm.startswith("zh") else "Conclusion and Recommendations"
    if conclusion_section not in outline_set and outline:
        conclusion_section = outline[-1]

    for frame in frames:
        f = dict(frame)
        sec_raw = str(f.get("section") or "").strip()
        sec = _normalize_section_text(sec_raw)
        if sec in preserve_sections:
            mapped_frames.append(f)
            continue

        pages = [p for p in (f.get("source_pages") or []) if isinstance(p, int)]
        mapped = _choose_from_pages(pages)
        if not mapped:
            mapped = _alias_match(sec)
        if not mapped and _is_conclusion_recommendation_text(
            " ".join(str(f.get(k) or "") for k in ("frame_title", "section", "subsection"))
        ):
            mapped = conclusion_section
        if not mapped and sec in outline_set:
            mapped = sec
        if not mapped and last_outline_section:
            mapped = last_outline_section
        if not mapped and outline:
            mapped = outline[0]
        if not mapped:
            mapped = sec or ("研究内容分析" if lang_norm.startswith("zh") else "Main Analysis")

        if mapped != sec_raw:
            rewritten += 1
            f["section"] = mapped
        else:
            f["section"] = mapped
        if mapped in outline_set:
            last_outline_section = mapped
        mapped_frames.append(f)

    if rewritten:
        print(f"[Pipeline] Remapped {rewritten} frame section label(s) to outline sections.")
    return mapped_frames, rewritten


def _align_frame_titles_to_outline(
    frames: list[dict],
    outline_sections: list[str],
    lang: str = "zh",
) -> tuple[list[dict], int]:
    if not frames:
        return frames, 0
    lang_norm = (lang or "zh").lower()
    caption_like = re.compile(
        r"^\s*(?:图|表|figure|fig\.?|table)\s*\d+",
        flags=re.IGNORECASE,
    )
    section_order = {sec: idx for idx, sec in enumerate(outline_sections or [])}
    counters: dict[str, int] = {}
    rewritten = 0
    normalized_frames: list[dict] = []

    for frame in frames:
        f = dict(frame)
        section = _normalize_section_text(str(f.get("section") or ""))
        if not section:
            section = outline_sections[0] if outline_sections else (
                "研究内容分析" if lang_norm.startswith("zh") else "Main Analysis"
            )
        f["section"] = section

        title = str(f.get("frame_title") or "").strip()
        suspicious = (
            not title
            or _is_toc_like_line(title)
            or _looks_like_mojibake_text(title, lang=lang_norm)
            or bool(caption_like.match(title))
            or len(title) > 56
        )
        if suspicious:
            counters[section] = counters.get(section, 0) + 1
            idx = counters[section]
            if lang_norm.startswith("zh"):
                title = f"{section}：关键分析 {idx}"
            else:
                title = f"{section}: Key Analysis {idx}"
            rewritten += 1
        f["frame_title"] = title
        normalized_frames.append(f)

    normalized_frames.sort(
        key=lambda fr: (
            section_order.get(_normalize_section_text(str(fr.get("section") or "")), 10**6),
            0 if _is_conclusion_recommendation_text(
                " ".join(str(fr.get(k) or "") for k in ("frame_title", "section", "subsection"))
            ) else 1,
        )
    )
    if rewritten:
        print(f"[Pipeline] Rewrote {rewritten} frame title(s) to align with outline sections.")
    return normalized_frames, rewritten


def _extract_summary_source(pdf_chunks: list[dict], max_chunks: int = 3) -> tuple[str, list[int], list[int]]:
    if not pdf_chunks:
        return "", [], []

    def _get_page(chunk: dict) -> int | None:
        meta = chunk.get("metadata") or {}
        page_val = meta.get("page")
        return page_val if isinstance(page_val, int) else None

    # Prefer an abstract-like chunk if present.
    for idx, chunk in enumerate(pdf_chunks):
        text = chunk.get("page_content") or ""
        if _is_toc_like_text(text):
            continue
        if re.search(r"\babstract\b", text, flags=re.IGNORECASE) or "摘要" in text:
            pages = []
            page_val = _get_page(chunk)
            if page_val is not None:
                pages.append(page_val)
            return text.strip(), pages, [idx]

    # Fallback: use the first few chunks as a rough overview.
    selected: list[int] = []
    for idx, chunk in enumerate(pdf_chunks):
        text = chunk.get("page_content") or ""
        if not text.strip():
            continue
        if _is_toc_like_text(text):
            continue
        selected.append(idx)
        if len(selected) >= max_chunks:
            break
    if not selected:
        selected = list(range(min(max_chunks, len(pdf_chunks))))
    texts: list[str] = []
    pages: list[int] = []
    for idx in selected:
        chunk = pdf_chunks[idx]
        text = chunk.get("page_content") or ""
        if text:
            texts.append(text.strip())
        page_val = _get_page(chunk)
        if page_val is not None and page_val not in pages:
            pages.append(page_val)
    return "\n\n".join(texts), pages, selected


def _build_summary_frame(
    summary_agent,
    instructions: dict,
    pdf_chunks: list[dict],
    section_blocks: list[dict],
    cached_summary: dict | None = None,
) -> dict | None:
    source_text = ""
    pages: list[int] = []
    chunk_indices: list[int] = []
    if cached_summary is None:
        if summary_agent is None or not pdf_chunks:
            return None
        source_text, pages, chunk_indices = _extract_summary_source(pdf_chunks)
        if not source_text:
            return None

    lang = (instructions.get("lang") or "zh").lower()
    summary_section = "研究概述" if lang.startswith("zh") else "Overview"
    section_titles = []
    for block in section_blocks or []:
        sec = (block.get("section") or "").strip()
        if sec and sec not in section_titles:
            section_titles.append(sec)

    if cached_summary is None:
        payload = {
            "task": "summary_slide",
            "instructions": instructions,
            "source_text": source_text,
            "section_titles": section_titles,
        }
        obj = None
        try:
            timeout_raw = os.getenv("PIPELINE_SUMMARY_TIMEOUT", "180").strip()
            try:
                timeout_s = int(timeout_raw)
            except ValueError:
                timeout_s = 180
            if timeout_s > 0:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        summary_agent.invoke,
                        {
                            "messages": [
                                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                            ]
                        },
                    )
                    response = future.result(timeout=timeout_s)
            else:
                response = summary_agent.invoke(
                    {
                        "messages": [
                            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                        ]
                    }
                )
            if isinstance(response, dict):
                messages = response.get("messages") or []
                content = getattr(messages[-1], "content", "") if messages else ""
            else:
                content = getattr(response, "content", "") or ""
            obj = _parse_json_payload(content)
        except TimeoutError:
            print(
                "[Pipeline] Summary LLM call timed out; using heuristic summary."
            )
            obj = None
        except Exception as exc:  # noqa: BLE001
            print(f"[Pipeline] Summary LLM call failed: {exc}")
            obj = None

        if not obj:
            sentences = _split_sentences(source_text, lang)
            bullets = sentences[:4]
            para = " ".join(sentences[:2]).strip()
            paragraphs = [para] if para else []
            obj = {
                "title": summary_section,
                "bullets": bullets,
                "paragraphs": paragraphs,
            }
    else:
        obj = cached_summary
    title = (obj.get("title") or summary_section).strip()
    bullets = obj.get("bullets") or []
    paragraphs = obj.get("paragraphs") or []
    if isinstance(bullets, str):
        bullets = [bullets]
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    if _is_toc_section_title(title) or _is_toc_like_text(
        "\n".join([title] + [str(x) for x in bullets] + [str(x) for x in paragraphs])
    ):
        title = summary_section
        bullets = [s for s in (section_titles or []) if s and not _is_toc_section_title(s)][:5]
        if not bullets:
            if lang.startswith("zh"):
                bullets = ["研究背景与问题", "研究方法与数据", "结果分析", "结论与建议"]
            else:
                bullets = ["Background and Problem", "Methods and Data", "Results", "Conclusions"]
        paragraphs = []

    return {
        "section": summary_section,
        "subsection": None,
        "frame_title": title or summary_section,
        "bullets": bullets,
        "paragraphs": paragraphs,
        "figure_slugs": [],
        "source_pages": pages,
        "source_chunk_indices": chunk_indices,
    }
def _collect_formula_candidates(pdf_chunks):
    """Collect heuristic formula candidates from PDF text chunks.

    This uses simple pattern matching on lines that look math-heavy (e.g.,
    contain equality signs and typical math tokens). The result is a list of
    lightweight formula records with raw text, page, and chunk index.
    """
    candidates = []
    for idx, chunk in enumerate(pdf_chunks or []):
        text = (chunk or {}).get("page_content") or ""
        metadata = (chunk or {}).get("metadata") or {}
        page = metadata.get("page")
        if not text:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if len(line) < 10:
                continue
            score = 0
            if "=" in line:
                score += 1
            if any(sym in line for sym in ["\\", "_", "^", "∈", "≤", "≥"]):
                score += 1
            if any(tok in line for tok in ["E[", "E(", "Var(", "argmax", "arg max", "argmin", "arg min"]):
                score += 1
            if score >= 2:
                candidates.append(
                    {
                        "id": f"eq_{len(candidates) + 1:03d}",
                        "raw_text": line,
                        "page": page,
                        "chunk_index": idx,
                    }
                )
    print(
        f"[Pipeline] Collected {len(candidates)} formula candidates from {len(pdf_chunks or [])} chunks"
    )
    return candidates


def _collect_formula_candidates_with_llm(model, pdf_chunks):
    """Use the LLM to extract formula candidates from PDF chunks.

    The model sees each chunk (with page and chunk_index metadata) and must
    return JSON of the form {"formulas": [{...}]}. We then normalize the
    records into a uniform list of {id, raw_text, latex, description, page,
    chunk_index}. If the LLM yields no usable formulas, we fall back to the
    simple heuristic collector so that equations are not completely lost.
    """
    all_formulas: list[dict] = []
    seen_keys: set[str] = set()

    # Prepare batched requests for all non-empty chunks. The provided model is
    # a specialized formula-extraction agent created via build_pipeline_agents,
    # so we only need to send HumanMessage payloads; its system prompt is
    # configured at agent construction time.
    requests: list[dict] = []
    metas: list[tuple[int, int | None]] = []
    for idx, chunk in enumerate(pdf_chunks or []):
        text = (chunk or {}).get("page_content") or ""
        metadata = (chunk or {}).get("metadata") or {}
        page = metadata.get("page")
        if not text or not text.strip():
            continue
        payload = {
            "task": "extract_formulas_for_chunk",
            "page": page,
            "chunk_index": idx,
            "text": text,
        }
        messages = [
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        requests.append({"messages": messages})
        metas.append((idx, page))

    if not requests:
        print(
            f"[Pipeline] LLM-based formula extraction: no non-empty chunks to process (total {len(pdf_chunks or [])} chunks)"
        )
        return []

    # Run batched calls with limited concurrency; if batch is unsupported,
    # fall back to sequential invoke on the same agent.
    batch_timeout_raw = os.getenv("PIPELINE_FORMULA_BATCH_TIMEOUT", "300").strip()
    try:
        batch_timeout_s = int(batch_timeout_raw)
    except ValueError:
        batch_timeout_s = 300
    try:
        print(
            f"[Pipeline] LLM-based formula extraction: batching {len(requests)} chunks "
            f"(max_concurrency={_get_max_concurrency()})"
        )
        if batch_timeout_s > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    model.batch,
                    requests,
                    config={"max_concurrency": _get_max_concurrency()},
                )
                responses = future.result(timeout=batch_timeout_s)
        else:
            responses = model.batch(
                requests,
                config={"max_concurrency": _get_max_concurrency()},
            )
    except TimeoutError:
        print(
            f"[Pipeline] Formula extraction batch timed out after {batch_timeout_s}s; "
            "falling back to sequential invokes."
        )
        responses = []
        for i, req in enumerate(requests):
            try:
                if i % 5 == 0:
                    print(
                        f"[Pipeline] Formula extraction progress: {i}/{len(requests)} chunks processed..."
                    )
                resp = model.invoke(req)
            except Exception as exc2:  # noqa: BLE001
                idx, _ = metas[i]
                print(f"[Pipeline] Formula extraction LLM call failed for chunk {idx}: {exc2}")
                resp = None
            responses.append(resp)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[Pipeline] Formula extraction batch call failed, falling back to sequential invokes: {exc}"
        )
        responses = []
        for i, req in enumerate(requests):
            try:
                if i % 5 == 0:
                    print(
                        f"[Pipeline] Formula extraction progress: {i}/{len(requests)} chunks processed..."
                    )
                resp = model.invoke(req)
            except Exception as exc2:  # noqa: BLE001
                idx, _ = metas[i]
                print(f"[Pipeline] Formula extraction LLM call failed for chunk {idx}: {exc2}")
                resp = None
            responses.append(resp)

    for (idx, page), response in zip(metas, responses):
        if response is None:
            continue

        obj = None
        if isinstance(response, dict):
            sr = response.get("structured_response")
            if sr is not None:
                try:
                    if hasattr(sr, "model_dump"):
                        obj = sr.model_dump()
                    elif hasattr(sr, "dict"):
                        obj = sr.dict()
                    elif isinstance(sr, dict):
                        obj = sr
                except Exception:  # noqa: BLE001
                    obj = None

        if obj is None:
            raw = ""
            if isinstance(response, dict):
                msgs = response.get("messages") or []
                if msgs:
                    last = msgs[-1]
                    raw = (getattr(last, "content", None) or "").strip()
            else:
                raw = (getattr(response, "content", None) or "").strip()
            if not raw:
                continue
            cleaned = raw
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1].strip()
            if cleaned and not cleaned.lstrip().startswith("{"):
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    cleaned = cleaned[start : end + 1]

            cleaned = _normalize_newlines_inside_strings(cleaned)

            try:
                obj = json.loads(cleaned)
            except Exception:
                continue

        formulas = obj.get("formulas") or []
        if not isinstance(formulas, list):
            continue

        for f in formulas:
            if not isinstance(f, dict):
                continue
            raw_text = (f.get("raw_text") or "").strip()
            if not raw_text:
                continue
            key = raw_text
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rec = {
                "id": f.get("id") or f"eq_{len(all_formulas) + 1:03d}",
                "raw_text": raw_text,
                "latex": f.get("latex") or raw_text,
                "description": f.get("description"),
                "page": f.get("page", page),
                "chunk_index": f.get("chunk_index", idx),
            }
            all_formulas.append(rec)

    if not all_formulas:
        print(
            f"[Pipeline] LLM-based formula extraction returned 0 formulas from {len(pdf_chunks or [])} chunks"
        )
    else:
        print(
            f"[Pipeline] Collected {len(all_formulas)} formula candidates from LLM extraction over {len(pdf_chunks or [])} chunks"
        )

    return all_formulas


def _collect_figure_metadata_with_llm(model, pdf_chunks):
    """Use the LLM to extract figure/table captions from PDF chunks.

    This does not depend on extracted images. Each chunk is sent with its
    page and index, and the model must return JSON of the form
    {"figures": [{...}]}. We normalize into a list compatible with
    aligned_figures items (slug, caption, context, page, figure_number,
    path?) so that downstream planning can treat them uniformly.
    """
    all_figs: list[dict] = []
    seen_keys: set[tuple[int | None, str]] = set()

    # Prepare batched requests for all non-empty chunks. The provided model is
    # a specialized figure-metadata agent, so we only send HumanMessage
    # payloads; its system prompt is configured at agent construction time.
    requests: list[dict] = []
    metas: list[tuple[int, int | None]] = []
    for idx, chunk in enumerate(pdf_chunks or []):
        text = (chunk or {}).get("page_content") or ""
        metadata = (chunk or {}).get("metadata") or {}
        page = metadata.get("page")
        if not text or not text.strip():
            continue
        payload = {
            "task": "extract_figures_for_chunk",
            "page": page,
            "chunk_index": idx,
            "text": text,
        }
        messages = [
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        requests.append({"messages": messages})
        metas.append((idx, page))

    if not requests:
        print(
            f"[Pipeline] LLM-based figure/table extraction: no non-empty chunks to process (total {len(pdf_chunks or [])} chunks)"
        )
        return []

    # Run batched calls with limited concurrency; if batch is unsupported,
    # fall back to sequential invoke on the same agent.
    batch_timeout_raw = os.getenv("PIPELINE_FIGURE_BATCH_TIMEOUT", "300").strip()
    try:
        batch_timeout_s = int(batch_timeout_raw)
    except ValueError:
        batch_timeout_s = 300
    try:
        print(
            f"[Pipeline] LLM-based figure/table extraction: batching {len(requests)} chunks "
            f"(max_concurrency={_get_max_concurrency()})"
        )
        if batch_timeout_s > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    model.batch,
                    requests,
                    config={"max_concurrency": _get_max_concurrency()},
                )
                responses = future.result(timeout=batch_timeout_s)
        else:
            responses = model.batch(
                requests,
                config={"max_concurrency": _get_max_concurrency()},
            )
    except TimeoutError:
        print(
            f"[Pipeline] Figure/table extraction batch timed out after {batch_timeout_s}s; "
            "falling back to sequential invokes."
        )
        responses = []
        for i, req in enumerate(requests):
            try:
                if i % 5 == 0:
                    print(
                        f"[Pipeline] Figure/table extraction progress: {i}/{len(requests)} chunks processed..."
                    )
                resp = model.invoke(req)
            except Exception as exc2:  # noqa: BLE001
                idx, _ = metas[i]
                print(f"[Pipeline] Figure caption extraction LLM call failed for chunk {idx}: {exc2}")
                resp = None
            responses.append(resp)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[Pipeline] Figure/table extraction batch call failed, falling back to sequential invokes: {exc}"
        )
        responses = []
        for i, req in enumerate(requests):
            try:
                if i % 5 == 0:
                    print(
                        f"[Pipeline] Figure/table extraction progress: {i}/{len(requests)} chunks processed..."
                    )
                resp = model.invoke(req)
            except Exception as exc2:  # noqa: BLE001
                idx, _ = metas[i]
                print(f"[Pipeline] Figure caption extraction LLM call failed for chunk {idx}: {exc2}")
                resp = None
            responses.append(resp)

    for (idx, page), response in zip(metas, responses):
        if response is None:
            continue

        obj = None
        if isinstance(response, dict):
            sr = response.get("structured_response")
            if sr is not None:
                try:
                    if hasattr(sr, "model_dump"):
                        obj = sr.model_dump()
                    elif hasattr(sr, "dict"):
                        obj = sr.dict()
                    elif isinstance(sr, dict):
                        obj = sr
                except Exception:  # noqa: BLE001
                    obj = None

        if obj is None:
            raw = ""
            if isinstance(response, dict):
                msgs = response.get("messages") or []
                if msgs:
                    last = msgs[-1]
                    raw = (getattr(last, "content", None) or "").strip()
            else:
                raw = (getattr(response, "content", None) or "").strip()
            if not raw:
                continue
            cleaned = raw
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1].strip()
            if cleaned and not cleaned.lstrip().startswith("{"):
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    cleaned = cleaned[start : end + 1]

            cleaned = _normalize_newlines_inside_strings(cleaned)

            try:
                obj = json.loads(cleaned)
            except Exception:
                continue

        figures = obj.get("figures") or []
        if not isinstance(figures, list):
            continue

        for f in figures:
            if not isinstance(f, dict):
                continue
            caption = (f.get("caption") or "").strip()
            if not caption:
                continue
            f_page = f.get("page", page)
            key = (f_page, caption)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            kind = (f.get("kind") or "figure").lower()
            prefix = "table" if kind.startswith("tab") else "figure"
            number = f.get("number")
            if isinstance(number, int):
                slug = f"{prefix}_{number:02d}"
            else:
                slug = f"{prefix}_{len(all_figs) + 1:03d}"
            rec = {
                "slug": slug,
                "caption": caption,
                "context": (f.get("context") or "").strip() or None,
                "page": f_page,
                "figure_number": number if isinstance(number, int) else None,
                # path is intentionally omitted; LatexWriterTool will simply
                # skip \includegraphics when no path is present.
            }
            all_figs.append(rec)

    if all_figs:
        print(
            f"[Pipeline] Collected {len(all_figs)} LLM-based figure/table captions from {len(pdf_chunks or [])} chunks"
        )

    return all_figs


def _align_llm_figures_with_image_index(llm_figures, image_index_items):
    """Align LLM-extracted figure/table captions with concrete image files.

    Strategy:
    - Prefer page-based alignment for images whose filenames encode page/index,
      e.g. "page_002_img_01" produced by PdfFigureExtractTool.
    - Fall back to figure-number based alignment for user-provided images whose
      filenames or guessed captions contain the figure number.
    - Captions that cannot be confidently matched keep no 'path' field so that
      downstream LaTeX rendering will skip \includegraphics for them, avoiding
      random or invalid images.
    """

    if not llm_figures:
        return []
    if not image_index_items:
        return list(llm_figures)

    images_by_page: dict[int, list[dict]] = {}
    generic_images: list[dict] = []
    slug_index: dict[str, list[dict]] = {}

    def _normalize_key(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"\.(png|jpe?g|pdf|svg)$", "", s)
        # Preserve ASCII word characters, digits, and common CJK characters,
        # normalize everything else to spaces so that Chinese or mixed
        # figure titles in filenames can still be matched against captions.
        s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
        return " ".join(s.split())

    for item in image_index_items or []:
        path_str = item.get("path")
        if not path_str:
            continue
        slug = item.get("slug") or Path(path_str).stem
        guessed_caption = (item.get("guessed_caption") or slug).lower()
        stem = Path(path_str).stem
        # Recognize auto-extracted images named like 'page_002_img_01.*' so that
        # we can anchor them to the correct PDF page when aligning with
        # figure/table captions.
        m = re.search(r"page_(\d+)_img_(\d+)", stem)
        img_rec = {
            "path": path_str,
            "slug": slug,
            "guessed_caption": guessed_caption,
        }
        norm_keys: set[str] = set()
        norm_slug = _normalize_key(slug)
        if norm_slug:
            norm_keys.add(norm_slug)
        norm_guess = _normalize_key(guessed_caption)
        if norm_guess:
            norm_keys.add(norm_guess)
        # Store a representative normalized key on the record for potential
        # debugging or downstream use, but index under all distinct keys so
        # that either the raw slug or the more human-readable guessed
        # caption can be matched against PDF captions.
        img_rec["norm_key"] = next(iter(norm_keys)) if norm_keys else ""
        for nk in norm_keys:
            slug_index.setdefault(nk, []).append(img_rec)
        if m:
            page = int(m.group(1))
            img_index = int(m.group(2))
            img_rec["page"] = page
            img_rec["img_index"] = img_index
            images_by_page.setdefault(page, []).append(img_rec)
        else:
            generic_images.append(img_rec)

    for page, imgs in images_by_page.items():
        imgs.sort(key=lambda x: x.get("img_index", 0))

    used_paths: set[str] = set()

    def _take_from_list(imgs: list[dict]) -> dict | None:
        for img in imgs:
            path_str = img.get("path")
            if not path_str or path_str in used_paths:
                continue
            used_paths.add(path_str)
            return img
        return None

    def _match_by_number(fig: dict, fig_number: int | None) -> dict | None:
        if fig_number is None:
            return None

        # Only consider images that (1) look like the same kind (figure/table)
        # and (2) explicitly mention this figure number near a "fig"/"figure" or
        # "table"/"tab" marker in their filename or guessed_caption. This avoids
        # falsely matching numbers that appear only inside dataset names such as
        # "XJTU3-1" or "Bearing3_3" when trying to align, for example, "Fig. 1".
        caption_text = (fig.get("caption") or "")
        caption_lower = caption_text.lower()
        is_table = caption_lower.lstrip().startswith("table") or caption_lower.lstrip().startswith("tables")
        caption_tokens = _simple_tokenize(caption_lower)

        best_img: dict | None = None
        best_score = 0

        number_str = str(fig_number)
        # Accept leading zeros (e.g., "fig07" for figure 7) but forbid extra
        # digits around the figure number so that figure 1 does not match 11.
        core_number_pattern = rf"(?<!\d)0*{re.escape(number_str)}(?!\d)"

        for img in generic_images:
            path_str = img.get("path")
            if not path_str or path_str in used_paths:
                continue
            text = f"{img.get('slug', '')} {img.get('guessed_caption', '')}".lower()
            img_is_table = "table" in text
            if img_is_table != is_table:
                continue

            if is_table:
                pattern = rf"tab(?:les?)?[^\d]*{core_number_pattern}"
            else:
                pattern = rf"fig(?:s|ure)?[^\d]*{core_number_pattern}"
            if not re.search(pattern, text, flags=re.IGNORECASE):
                continue

            img_tokens = _simple_tokenize(text)
            score = len(caption_tokens & img_tokens)

            # Slightly favor images whose slug clearly starts with "fig" or
            # "figure", which is common for hand-prepared figure files.
            slug_lower = (img.get("slug") or "").lower()
            if slug_lower.startswith("fig") or slug_lower.startswith("figure"):
                score += 1

            if score > best_score:
                best_score = score
                best_img = img

        if best_img is not None and best_score > 0:
            used_paths.add(best_img.get("path"))
            return best_img
        return None

    aligned: list[dict] = []
    for fig in llm_figures:
        page = fig.get("page")

        # Derive the figure or table number from the caption text itself
        # instead of trusting any LLM-provided figure_number field, which can
        # occasionally be off (e.g., mis-reading "Figs. 4" as 14).
        caption_text = fig.get("caption") or ""
        caption_lower = caption_text.lower()
        fig_number: int | None = None
        m_fig = re.search(r"figs?\.?\s*(\d+)", caption_lower)
        m_tab = re.search(r"tables?\.?\s*(\d+)", caption_lower)
        m = m_fig or m_tab
        if m:
            try:
                fig_number = int(m.group(1))
            except Exception:  # noqa: BLE001
                fig_number = None

        chosen: dict | None = None

        # 1) Exact caption-to-filename match via normalized key (user often keeps
        # filenames identical to figure captions).
        norm_caption_key = _normalize_key(caption_text)
        if norm_caption_key:
            candidates = slug_index.get(norm_caption_key) or []
            for img in candidates:
                path_str = img.get("path")
                if not path_str or path_str in used_paths:
                    continue
                chosen = img
                used_paths.add(path_str)
                break

        if chosen is None and page is not None and page in images_by_page:
            chosen = _take_from_list(images_by_page[page])

        if chosen is None and fig_number is not None:
            chosen = _match_by_number(fig, fig_number)

        aligned_fig = dict(fig)
        if chosen and chosen.get("path"):
            aligned_fig["path"] = chosen["path"]
        aligned.append(aligned_fig)

    return aligned


def _backfill_unmatched_user_images_with_llm_metadata(
    image_index_items: list[dict],
    aligned_figures: list[dict],
) -> list[dict]:
    """Give unmatched user images captions/context from LLM figure metadata.

    After initial alignment, some user-provided images may still have no
    corresponding aligned_figures entry with a concrete ``path`` (for example,
    hand-prepared files like ``Fig. 7.jpg`` or ``Fig. 8(d).png``). For each
    such image, try to locate a compatible figure metadata record by matching
    on the figure/table number and normalized caption key, then clone that
    metadata and attach the image path.
    """

    if not image_index_items or not aligned_figures:
        return aligned_figures

    used_paths: set[str] = set()
    for fig in aligned_figures or []:
        path_str = fig.get("path")
        if path_str:
            used_paths.add(path_str)

    # Build indices over existing figure metadata so that we can reuse titles,
    # context and page information when attaching additional image files.
    figs_by_number: dict[int, list[dict]] = {}
    figs_by_normkey: dict[str, list[dict]] = {}

    def _norm_key(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"\.(png|jpe?g|pdf|svg)$", "", s)
        s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
        return " ".join(s.split())

    for fig in aligned_figures or []:
        caption = (fig.get("caption") or "").lower()
        m_fig = re.search(r"figs?\.?\s*(\d+)", caption)
        m_tab = re.search(r"tables?\.?\s*(\d+)", caption)
        m = m_fig or m_tab
        try:
            num = int(m.group(1)) if m else None
        except Exception:  # noqa: BLE001
            num = None
        if isinstance(num, int):
            figs_by_number.setdefault(num, []).append(fig)
        nk = _norm_key(caption)
        if nk:
            figs_by_normkey.setdefault(nk, []).append(fig)

    existing_slugs: set[str] = {
        fig.get("slug") for fig in aligned_figures or [] if fig.get("slug")
    }
    new_figs: list[dict] = []

    for img in image_index_items or []:
        path_str = img.get("path")
        if not path_str or path_str in used_paths:
            continue

        slug = img.get("slug") or Path(path_str).stem
        guessed_caption = img.get("guessed_caption") or slug
        text = f"{slug} {guessed_caption}".lower()

        m_fig = re.search(r"figs?\.?\s*(\d+)", text)
        m_tab = re.search(r"tables?\.?\s*(\d+)", text)
        m = m_fig or m_tab
        try:
            num = int(m.group(1)) if m else None
        except Exception:  # noqa: BLE001
            num = None

        candidates: list[dict] = []
        nk = _norm_key(str(guessed_caption))
        if nk and nk in figs_by_normkey:
            candidates.extend(figs_by_normkey[nk])
        if not candidates and isinstance(num, int) and num in figs_by_number:
            candidates.extend(figs_by_number[num])

        # Prefer metadata that already carries non-empty context and a valid page.
        chosen: dict | None = None
        if candidates:
            def _score_candidate(fig: dict) -> int:
                score = 0
                if fig.get("context"):
                    score += 2
                page_val = fig.get("page")
                if isinstance(page_val, int):
                    score += 1
                return score

            chosen = max(candidates, key=_score_candidate)

        base_slug = slug or ((chosen or {}).get("slug") or "figure")
        new_slug = base_slug
        counter = 2
        while new_slug in existing_slugs:
            new_slug = f"{base_slug}_{counter}"
            counter += 1
        existing_slugs.add(new_slug)

        if chosen is not None:
            new_fig = dict(chosen)
            new_fig["slug"] = new_slug
            new_fig["path"] = path_str
        else:
            # Fallback: synthesize minimal metadata from the image itself so that
            # the figure can still be attached to a content frame.
            new_fig = {
                "slug": new_slug,
                "path": path_str,
                "caption": str(guessed_caption),
                "context": None,
            }

        new_figs.append(new_fig)
        used_paths.add(path_str)

    if new_figs:
        print(
            f"[Pipeline] Backfilled metadata for {len(new_figs)} user images using LLM figure captions."
        )
    return aligned_figures + new_figs


def _enrich_figures_with_vision_descriptions(aligned_figures: list[dict]) -> None:
    """Best-effort vision analysis to enrich figure metadata.

    For each figure record that has a concrete image path, call the shared
    analyze_image_with_vision helper to obtain a short natural-language
    description and store it under the 'vision_description' key. This
    description is later used to improve figure/text matching.
    """

    if not aligned_figures:
        return

    total = 0
    enriched = 0
    min_side = 500
    for fig in aligned_figures:
        path_str = fig.get("path")
        if not path_str:
            continue
        width, height = get_image_dimensions(Path(path_str))
        if width is not None and height is not None:
            if width < min_side or height < min_side:
                continue
        caption = fig.get("caption") or ""
        context = fig.get("context") or ""
        if caption and not _caption_needs_vision(caption, context):
            continue
        # Skip figures that already carry a vision-based description so that
        # repeated runs in the same process do not re-analyze the same image.
        if fig.get("vision_description"):
            continue
        total += 1
        try:
            desc = analyze_image_with_vision(path_str)
        except Exception as exc:  # noqa: BLE001
            slug = fig.get("slug") or path_str
            print(f"[Pipeline] Vision analysis failed for figure {slug}: {exc}")
            continue
        if desc:
            fig["vision_description"] = desc
            enriched += 1

    if enriched:
        print(
            f"[Pipeline] Enriched {enriched}/{total} figures with vision-based descriptions for better matching."
        )


def _simple_tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {tok for tok in re.split(r"\W+", text.lower()) if tok}


def _caption_needs_vision(caption: str, context: str | None = None) -> bool:
    caption = (caption or "").strip()
    if not caption:
        return True
    tokens = _simple_tokenize(caption)
    generic = {"fig", "figure", "figures", "table", "tables", "图", "表", "图表"}
    informative = [t for t in tokens if t not in generic and not t.isdigit() and len(t) > 1]
    if len(informative) >= 2:
        return False
    if context:
        ctx_tokens = _simple_tokenize(context)
        ctx_info = [t for t in ctx_tokens if t not in generic and not t.isdigit()]
        if len(ctx_info) >= 3:
            return False
    return True


def _get_max_concurrency(default: int = 4) -> int:
    raw = os.getenv("PIPELINE_MAX_CONCURRENCY") or os.getenv("LLM_MAX_CONCURRENCY")
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return max(1, min(val, 16))


def _get_planner_recursion_limit(default: int = 10) -> int:
    raw = os.getenv("PIPELINE_PLANNER_RECURSION_LIMIT", str(default)).strip()
    try:
        val = int(raw)
    except ValueError:
        val = default
    return max(4, min(val, 64))


def _get_planner_max_concurrency(default: int | None = None) -> int:
    if default is None:
        default = min(_get_max_concurrency(), 8)
    raw = os.getenv("PIPELINE_PLANNER_MAX_CONCURRENCY", str(default)).strip()
    try:
        val = int(raw)
    except ValueError:
        val = default
    return max(1, min(val, 16))


def _get_planner_batch_chunk_size(default: int = 12) -> int:
    raw = os.getenv("PIPELINE_PLANNER_BATCH_CHUNK_SIZE", str(default)).strip()
    try:
        val = int(raw)
    except ValueError:
        val = default
    return max(1, min(val, 64))


def _contains_model_limit_message(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False
    if "model call limits exceeded" in s:
        return True
    has_thread_limit = "thread limit" in s
    has_run_limit = "run limit" in s
    if has_thread_limit and has_run_limit:
        return True
    return False


def _get_max_frames_per_section(default: int = 6) -> int:
    raw = os.getenv("PIPELINE_MAX_FRAMES_PER_SECTION")
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return max(2, min(val, 10))


def _resolve_max_frames_per_section(instructions: dict | None = None, default: int = 6) -> int:
    if isinstance(instructions, dict):
        val = instructions.get("max_frames_per_section")
        if isinstance(val, int):
            return max(2, min(val, 10))
        if isinstance(val, str):
            try:
                parsed = int(val.strip())
            except ValueError:
                parsed = None
            if isinstance(parsed, int):
                return max(2, min(parsed, 10))
    return _get_max_frames_per_section(default=default)


def _get_max_figures_per_frame(default: int = 4) -> int:
    raw = os.getenv("PIPELINE_MAX_FIGURES_PER_FRAME")
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return max(1, min(val, 6))


def _use_appendix_images(default: bool = False) -> bool:
    raw = os.getenv("PIPELINE_ENABLE_APPENDIX_IMAGES")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


def _resolve_appendix_images(instructions: dict | None = None, default: bool = False) -> bool:
    if isinstance(instructions, dict):
        val = instructions.get("appendix_images")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "y"}
    return _use_appendix_images(default=default)


def _normalize_report_type(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in {
        REPORT_TYPE_ZHENGDA,
        "zhengda",
        "zhengda-cup",
        "zhengda_cup_report",
        "正大杯",
        "正大杯报告",
    }:
        return REPORT_TYPE_ZHENGDA
    return REPORT_TYPE_GENERAL


def _get_report_type(default: str = REPORT_TYPE_GENERAL) -> str:
    raw = os.getenv("PIPELINE_REPORT_TYPE", default)
    normalized = _normalize_report_type(raw)
    if normalized:
        return normalized
    return REPORT_TYPE_GENERAL


def _resolve_report_type(
    instructions: dict | None = None,
    default: str = REPORT_TYPE_GENERAL,
) -> str:
    if isinstance(instructions, dict):
        val = instructions.get("report_type")
        if isinstance(val, str):
            return _normalize_report_type(val)
    return _get_report_type(default=default)


def _zhengda_template_sections(lang: str = "zh") -> list[str]:
    if (lang or "zh").lower().startswith("zh"):
        return list(ZHENGDA_TEMPLATE_SECTIONS_ZH)
    return list(ZHENGDA_TEMPLATE_SECTIONS_EN)


def _get_detail_profile(default: str = "detailed") -> str:
    raw = os.getenv("PIPELINE_DETAIL_PROFILE", default).strip().lower()
    if raw not in {"compact", "standard", "detailed"}:
        return default
    return raw


def _resolve_detail_profile(instructions: dict | None = None, default: str = "detailed") -> str:
    if isinstance(instructions, dict):
        val = instructions.get("detail_profile")
        if isinstance(val, str):
            raw = val.strip().lower()
            if raw in {"compact", "standard", "detailed"}:
                return raw
    return _get_detail_profile(default=default)


def _detail_profile_defaults(profile: str) -> tuple[int, int, int]:
    p = (profile or "detailed").strip().lower()
    if p == "compact":
        return 12, 18, 4
    if p == "standard":
        return 18, 26, 6
    return 25, 35, 8


def _get_target_content_pages_min(detail_profile: str = "detailed") -> int:
    default_min, _, _ = _detail_profile_defaults(detail_profile)
    raw = os.getenv("PIPELINE_TARGET_CONTENT_PAGES_MIN", str(default_min)).strip()
    try:
        val = int(raw)
    except ValueError:
        val = default_min
    return max(6, min(val, 200))


def _get_target_content_pages_max(detail_profile: str = "detailed") -> int:
    default_min, default_max, _ = _detail_profile_defaults(detail_profile)
    raw = os.getenv("PIPELINE_TARGET_CONTENT_PAGES_MAX", str(default_max)).strip()
    try:
        val = int(raw)
    except ValueError:
        val = default_max
    val = max(8, min(val, 240))
    return max(default_min, val)


def _resolve_target_content_pages(
    instructions: dict | None = None,
    detail_profile: str = "detailed",
) -> tuple[int, int]:
    min_pages = _get_target_content_pages_min(detail_profile=detail_profile)
    max_pages = _get_target_content_pages_max(detail_profile=detail_profile)
    if isinstance(instructions, dict):
        min_val = instructions.get("target_content_pages_min")
        max_val = instructions.get("target_content_pages_max")
        if isinstance(min_val, int):
            min_pages = max(6, min(min_val, 200))
        if isinstance(max_val, int):
            max_pages = max(8, min(max_val, 240))
    if max_pages < min_pages:
        max_pages = min_pages
    return min_pages, max_pages


def _get_mojibake_policy(default: str = "auto_fallback") -> str:
    raw = os.getenv("PIPELINE_MOJIBAKE_POLICY", default).strip().lower()
    if raw not in {"auto_fallback", "strict_fail", "warn_only"}:
        return default
    return raw


def _resolve_mojibake_policy(instructions: dict | None = None, default: str = "auto_fallback") -> str:
    if isinstance(instructions, dict):
        val = instructions.get("mojibake_policy")
        if isinstance(val, str):
            raw = val.strip().lower()
            if raw in {"auto_fallback", "strict_fail", "warn_only"}:
                return raw
    return _get_mojibake_policy(default=default)


def _enable_ocr_fallback(default: bool = True) -> bool:
    raw = os.getenv("PIPELINE_ENABLE_OCR_FALLBACK")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _text_quality_debug_enabled(default: bool = True) -> bool:
    raw = os.getenv("PIPELINE_TEXT_QUALITY_DEBUG")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _preferred_text_source_path() -> str | None:
    raw = os.getenv("PIPELINE_PREFERRED_TEXT_SOURCE", "").strip()
    if not raw:
        return None
    return raw


def _write_quality_report(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Warning: failed to write quality report {path}: {exc}")


def _use_cache() -> bool:
    raw = os.getenv("PIPELINE_USE_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no"}


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{timestamp}_{suffix}"


def _load_cache_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _write_cache_json(path: Path, data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Warning: failed to write cache file {path}: {exc}")


def _filter_section_figures_by_text(
    section_figures: list[dict],
    source_text: str,
    retrieved_context: str,
) -> list[dict]:
    if not section_figures:
        return section_figures
    content_tokens = _simple_tokenize(f"{source_text} {retrieved_context}")
    if not content_tokens:
        return section_figures
    generic = {"fig", "figure", "figures", "table", "tables"}
    filtered: list[dict] = []
    for fig in section_figures:
        caption = fig.get("caption") or ""
        context = fig.get("context") or ""
        tokens = _simple_tokenize(f"{caption} {context}")
        tokens = {t for t in tokens if t not in generic and not t.isdigit()}
        if not tokens:
            filtered.append(fig)
            continue
        if tokens & content_tokens:
            filtered.append(fig)
    if len(filtered) != len(section_figures):
        print(
            f"[Pipeline] Filtered section figures by text overlap: {len(section_figures)} -> {len(filtered)}"
        )
    return filtered


def _distribute_unused_figures_into_frames(
    frames: list[dict],
    aligned_figures: list[dict],
    keyword_index: list[dict] | None = None,
) -> int:
    """Attach unused aligned figures to existing frames based on text overlap.

    For each figure that has a concrete image path and is not yet referenced by
    any frame via figure_slugs, choose a frame whose text best matches the
    figure's caption/context, preferring frames whose source_pages include the
    figure's page. The figure slug is then appended to that frame's
    figure_slugs list. Returns the number of figures newly attached.
    """

    if not frames or not aligned_figures:
        return 0

    figures_by_slug: dict[str, dict] = {}
    for fig in aligned_figures or []:
        slug = fig.get("slug")
        if not slug:
            continue
        figures_by_slug[slug] = fig

    used_slugs: set[str] = set()
    frame_infos: list[dict] = []

    for idx, frame in enumerate(frames):
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        frame["figure_slugs"] = slugs
        for slug in slugs:
            used_slugs.add(slug)

        texts: list[str] = []
        # Include section/subsection/frame_title text so that figure captions
        # can align with high-level titles like "仿真结果" or "评价指标定义" even
        # when the body paragraphs are short.
        title = frame.get("frame_title") or frame.get("section") or ""
        subsection = frame.get("subsection") or ""
        if title:
            texts.append(str(title))
        if subsection:
            texts.append(str(subsection))
        for para in frame.get("paragraphs") or []:
            texts.append(str(para))
        for bullet in frame.get("bullets") or []:
            texts.append(str(bullet))
        tokens = _simple_tokenize("\n".join(texts))
        pages = frame.get("source_pages") or []
        pages_set = set(pages)
        chunk_indices = frame.get("source_chunk_indices") or []
        chunk_indices_set = {ci for ci in chunk_indices if isinstance(ci, int)}
        has_text = bool(texts)
        frame_infos.append(
            {
                "frame": frame,
                "tokens": tokens,
                "pages_set": pages_set,
                "chunk_indices_set": chunk_indices_set,
                "index": idx,
                "has_text": has_text,
            }
        )

    attached = 0
    min_score = 2

    for fig in aligned_figures or []:
        slug = fig.get("slug")
        path_str = fig.get("path")
        if not slug or not path_str:
            continue
        if slug in used_slugs:
            continue

        caption = fig.get("caption") or ""
        context = fig.get("context") or ""
        vision_desc = fig.get("vision_description") or ""
        fig_tokens = _simple_tokenize(" ".join([caption, context, vision_desc]))
        if fig_tokens:
            # Drop generic tokens like "fig", "figure", "table" and pure
            # numbers so that alignment focuses on semantic keywords such as
            # "浠跨湡", "learning", "reward", etc.
            generic = {"fig", "figure", "figures", "table", "tables"}
            fig_tokens = {t for t in fig_tokens if t not in generic and not t.isdigit()}
        fig_page = fig.get("page")

        # Use the hybrid RAG keyword index (when available) as the *primary*
        # signal to locate the most relevant PDF chunks for this figure based
        # on its caption/context. We then try to attach the image to frames
        # whose source_chunk_indices overlap those chunks, using page and token
        # overlap only as secondary/fallback hints.
        fig_chunks_set: set[int] = set()
        if keyword_index and (caption or context or vision_desc):
            query_parts: list[str] = []
            if caption:
                query_parts.append(str(caption))
            if context:
                query_parts.append(str(context))
            if vision_desc:
                query_parts.append(str(vision_desc))
            query = " ".join(query_parts).strip()
            if query and len(query) >= 4:
                try:
                    results = keyword_search(
                        keyword_index,
                        query=query,
                        pages=None,
                        k=8,
                        preferred_pages=[fig_page]
                        if isinstance(fig_page, int)
                        else None,
                    )
                except Exception:  # noqa: BLE001
                    results = []
                for item in results or []:
                    idx_val = item.get("index")
                    if isinstance(idx_val, int):
                        fig_chunks_set.add(idx_val)

        # Candidate frames must have some text so that images are integrated
        # into content slides rather than creating standalone image-only pages.
        candidate_infos = [info for info in frame_infos if info["has_text"]]
        if not candidate_infos:
            continue

        best_frame = None
        best_score = 0

        def _score_frame(info: dict) -> int:
            """Unified scoring: RAG chunk overlap is primary, then tokens and page.

            - RAG: overlap between fig_chunks_set and frame.source_chunk_indices
            - Tokens: intersection between fig_tokens and frame tokens
            - Page: small bonus when figure page is within frame.source_pages
            """

            score = 0
            if fig_tokens:
                score += len(fig_tokens & info["tokens"])
            if fig_page is not None and fig_page in info["pages_set"]:
                score += 2
            if fig_chunks_set and info["chunk_indices_set"]:
                overlap = fig_chunks_set & info["chunk_indices_set"]
                if overlap:
                    # Base boost plus a small bonus for multiple overlapping chunks.
                    score += 5 + min(len(overlap), 3)
            return score

        # 1) Prefer frames that overlap with RAG-retrieved chunk indices.
        #    Then refine within those candidates by token/page overlap.
        if fig_chunks_set:
            rag_candidates: list[dict] = []
            for info in candidate_infos:
                if info["chunk_indices_set"] and fig_chunks_set & info["chunk_indices_set"]:
                    rag_candidates.append(info)

            if rag_candidates:
                rag_page_candidates: list[dict] = []
                if isinstance(fig_page, int):
                    for info in rag_candidates:
                        if fig_page in info["pages_set"]:
                            rag_page_candidates.append(info)

                primary_candidates = rag_page_candidates or rag_candidates
                for info in primary_candidates:
                    score = _score_frame(info)
                    if score > best_score:
                        best_score = score
                        best_frame = info["frame"]

        # 2) If RAG did not produce a useful match, fall back to page/token
        #    overlap heuristics using the same scoring function.
        if best_frame is None:
            page_candidates: list[dict] = []
            if isinstance(fig_page, int):
                for info in candidate_infos:
                    if fig_page in info["pages_set"]:
                        page_candidates.append(info)

            secondary_candidates = page_candidates or candidate_infos
            for info in secondary_candidates:
                score = _score_frame(info)
                if score > best_score:
                    best_score = score
                    best_frame = info["frame"]

        # 3) As a final fallback, or when only a zero-score match exists,
        #    attach to the text frame whose source_pages are closest to the
        #    figure page (or the earliest frame if page is unknown).
        if best_frame is None or best_score < min_score:
            continue

        if best_frame is not None:
            best_frame_slugs = best_frame.get("figure_slugs") or []
            if slug not in best_frame_slugs:
                best_frame_slugs.append(slug)
                best_frame["figure_slugs"] = best_frame_slugs
            used_slugs.add(slug)
            attached += 1

    if attached:
        print(
            f"[Pipeline] Attached {attached} previously-unused figures to existing frames based on caption/context similarity and page proximity."
        )

    return attached


def _build_extra_image_frames(
    frames: list[dict],
    aligned_figures: list[dict],
    image_index_items: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Ensure every user image appears at least once by adding extra frames.

    This helper inspects the current frames and aligned_figures to see which
    concrete image file paths are already referenced via figure_slugs. For any
    remaining image in image_index_items that is not yet used, it creates a
    simple image-only frame under an appendix-like section and adds a minimal
    figure metadata record so that LatexWriterTool can render it.
    """

    if not image_index_items:
        return [], aligned_figures

    figures_by_slug: dict[str, dict] = {}
    for fig in aligned_figures or []:
        slug = fig.get("slug")
        if not slug:
            continue
        figures_by_slug[slug] = fig

    used_paths: set[str] = set()
    for frame in frames or []:
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        for slug in slugs:
            fig = figures_by_slug.get(slug)
            if not fig:
                continue
            path_str = fig.get("path")
            if path_str:
                used_paths.add(path_str)

    existing_slugs: set[str] = {fig.get("slug") for fig in aligned_figures or [] if fig.get("slug")}
    extra_frames: list[dict] = []
    extra_figures: list[dict] = []
    extra_frame_slugs: list[str] = []
    min_score = 2
    max_figs_per_frame = _get_max_figures_per_frame()

    figs_by_path: dict[str, list[dict]] = {}
    for fig in aligned_figures or []:
        path_val = (fig.get("path") or "").strip()
        if not path_val:
            continue
        figs_by_path.setdefault(path_val, []).append(fig)

    frame_infos: list[dict] = []
    for idx, frame in enumerate(frames or []):
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        frame["figure_slugs"] = slugs
        texts: list[str] = []
        for para in frame.get("paragraphs") or []:
            texts.append(str(para))
        for bullet in frame.get("bullets") or []:
            texts.append(str(bullet))
        tokens = _simple_tokenize("\n".join(texts))
        pages = frame.get("source_pages") or []
        pages_set = {p for p in pages if isinstance(p, int)}
        chunk_indices = frame.get("source_chunk_indices") or []
        chunk_indices_set = {ci for ci in chunk_indices if isinstance(ci, int)}
        has_text = bool(texts)
        frame_infos.append(
            {
                "frame": frame,
                "tokens": tokens,
                "pages_set": pages_set,
                "chunk_indices_set": chunk_indices_set,
                "index": idx,
                "has_text": has_text,
            }
        )

    for img in image_index_items or []:
        path_str = img.get("path")
        if not path_str or path_str in used_paths:
            continue

        slug = img.get("slug") or Path(path_str).stem
        guessed_caption = img.get("guessed_caption") or slug or path_str
        base_slug = slug or "extra_fig"
        new_slug = base_slug
        counter = 2
        while new_slug in existing_slugs:
            new_slug = f"{base_slug}_{counter}"
            counter += 1
        existing_slugs.add(new_slug)

        meta_list = figs_by_path.get(path_str) or []
        meta = meta_list[0] if meta_list else {}
        if isinstance(guessed_caption, str):
            caption_val = guessed_caption
        else:
            caption_val = str(guessed_caption)
        if isinstance(meta, dict) and meta.get("caption"):
            caption_val = str(meta.get("caption"))
        if isinstance(meta, dict):
            context_val = meta.get("context") or ""
            fig_page = meta.get("page")
            fig_chunk_index = meta.get("chunk_index")
            vision_desc_val = meta.get("vision_description") or ""
        else:
            context_val = ""
            fig_page = None
            fig_chunk_index = None
            vision_desc_val = ""

        fig_tokens = _simple_tokenize(f"{caption_val} {context_val} {vision_desc_val}")

        candidate_infos = [info for info in frame_infos if info["has_text"]]
        best_frame = None
        best_score = 0
        if candidate_infos:
            for info in candidate_infos:
                score = 0
                if fig_tokens:
                    score += len(fig_tokens & info["tokens"])
                if isinstance(fig_page, int) and fig_page in info["pages_set"]:
                    score += 2
                if (
                    isinstance(fig_chunk_index, int)
                    and info["chunk_indices_set"]
                    and fig_chunk_index in info["chunk_indices_set"]
                ):
                    score += 4
                if score > best_score:
                    best_score = score
                    best_frame = info["frame"]

        if best_score < min_score:
            best_frame = None

        extra_fig = {
            "slug": new_slug,
            "path": path_str,
            "caption": caption_val,
            "context": context_val or None,
            "page": fig_page if isinstance(fig_page, int) else None,
            "vision_description": vision_desc_val or None,
        }
        extra_figures.append(extra_fig)

        attached_to_content = False
        allow_fallback = best_score >= min_score
        if candidate_infos and best_frame is None and allow_fallback:
            best_frame_fallback = None
            best_dist = 10**9
            for info in candidate_infos:
                pages_set = info["pages_set"]
                if isinstance(fig_page, int) and pages_set:
                    frame_page = min(p for p in pages_set if isinstance(p, int))
                    dist = abs(frame_page - fig_page)
                else:
                    dist = info["index"]
                if dist < best_dist:
                    best_dist = dist
                    best_frame_fallback = info["frame"]
            best_frame = best_frame_fallback

        if best_frame is not None:
            best_frame_slugs = best_frame.get("figure_slugs") or []
            if new_slug not in best_frame_slugs:
                best_frame_slugs.append(new_slug)
                best_frame["figure_slugs"] = best_frame_slugs
            used_paths.add(path_str)
            attached_to_content = True

        if attached_to_content:
            continue

        extra_frame_slugs.append(new_slug)
        used_paths.add(path_str)

    if extra_frame_slugs:
        for i in range(0, len(extra_frame_slugs), max_figs_per_frame):
            chunk = extra_frame_slugs[i : i + max_figs_per_frame]
            if not chunk:
                continue
            if len(chunk) == 1:
                frame_title = "附录图像"
            else:
                frame_title = f"附录图像 {i + 1}-{i + len(chunk)}"
            extra_frames.append(
                {
                    "section": "附录：补充图像",
                    "subsection": None,
                    "frame_title": frame_title,
                    "bullets": [],
                    "paragraphs": [],
                    "figure_slugs": chunk,
                }
            )

    if extra_frames:
        print(
            f"[Pipeline] Added {len(extra_frames)} extra image frames so that all user images appear at least once in the PPT."
        )

    return extra_frames, aligned_figures + extra_figures


def _remove_figure_only_frames(frames: list[dict]) -> list[dict]:
    """Drop frames that contain only figures and no text.

    The associated figures remain in aligned_figures and will be reattached to
    nearby text frames by _distribute_unused_figures_into_frames so that
    images are integrated into content slides instead of occupying standalone
    image-only slides.
    """

    if not frames:
        return frames

    new_frames: list[dict] = []
    removed = 0

    for frame in frames or []:
        bullets = frame.get("bullets") or []
        paragraphs = frame.get("paragraphs") or []
        has_text = bool(bullets or paragraphs)
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        frame["figure_slugs"] = slugs

        if not has_text and slugs:
            removed += 1
            continue

        new_frames.append(frame)

    if removed:
        print(
            f"[Pipeline] Removed {removed} figure-only frames; their images will be reattached to nearby text frames."
        )

    return new_frames
def _deduplicate_figure_usages_by_image_path(
    frames: list[dict],
    aligned_figures: list[dict],
) -> list[dict]:
    """Ensure each concrete image path is used at most once across all frames.

    This walks frames in their final order and, for each figure slug that
    resolves to a figure with a concrete ``path``, keeps only the first
    occurrence of that image path and drops any later references. Frames that
    end up with neither text nor figures are removed.
    """

    if not frames or not aligned_figures:
        return frames

    figures_by_slug: dict[str, dict] = {}
    for fig in aligned_figures or []:
        slug = fig.get("slug")
        if not slug:
            continue
        figures_by_slug[slug] = fig

    used_paths: set[str] = set()
    deduped_frames: list[dict] = []

    for frame in frames or []:
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]

        new_slugs: list[str] = []
        seen_slugs: set[str] = set()

        for slug in slugs:
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            fig = figures_by_slug.get(slug)
            path_str = fig.get("path") if fig else None
            if not path_str:
                # Metadata-only figure (no concrete user image); keep slug.
                new_slugs.append(slug)
                continue

            if path_str in used_paths:
                # This concrete image has already been attached to an earlier frame.
                continue

            used_paths.add(path_str)
            new_slugs.append(slug)

        new_frame = dict(frame)
        new_frame["figure_slugs"] = new_slugs

        has_text = bool(new_frame.get("bullets") or new_frame.get("paragraphs"))
        has_figs = bool(new_slugs)
        if not has_text and not has_figs:
            continue

        deduped_frames.append(new_frame)

    if len(deduped_frames) != len(frames):
        print(
            f"[Pipeline] Deduplicated figure usages by image path: frames {len(frames)} -> {len(deduped_frames)}"
        )

    return deduped_frames


def _cap_frames_per_section(
    frames: list[dict],
    aligned_figures: list[dict],
    max_frames_per_section: int,
    target_min_pages: int | None = None,
    detail_profile: str = "standard",
) -> list[dict]:
    """Soft-limit section/subsection frame count while preserving detail."""

    if not frames or not isinstance(max_frames_per_section, int) or max_frames_per_section <= 0:
        return frames

    figures_by_slug: dict[str, dict] = {}
    for fig in aligned_figures or []:
        slug = fig.get("slug")
        if not slug:
            continue
        figures_by_slug[slug] = fig

    def _has_image(frame: dict) -> bool:
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        for slug in slugs:
            fig = figures_by_slug.get(slug)
            path_str = fig.get("path") if fig else None
            if path_str:
                return True
        return False

    def _is_priority_frame(frame: dict) -> bool:
        text = " ".join(
            str(frame.get(key) or "") for key in ("frame_title", "section", "subsection")
        )
        if not text:
            return False
        cn_keywords = [
            "概述",
            "摘要",
            "贡献",
            "总结",
            "要点",
            "小结",
            "结论",
            "建议",
            "总结与展望",
            "结论与建议",
            "未来工作",
        ]
        en_keywords = [
            "overview",
            "summary",
            "contribution",
            "takeaway",
            "key result",
            "highlights",
            "conclusion",
            "recommendation",
            "future work",
        ]
        if any(kw in text for kw in cn_keywords):
            return True
        text_lower = text.lower()
        return any(kw in text_lower for kw in en_keywords)

    groups: list[tuple[tuple[str | None, str | None], list[dict]]] = []
    group_map: dict[tuple[str | None, str | None], list[dict]] = {}

    for frame in frames:
        key = (frame.get("section"), frame.get("subsection"))
        if key not in group_map:
            group_map[key] = []
            groups.append((key, group_map[key]))
        group_map[key].append(frame)

    capped_frames: list[dict] = []
    hard_cap_raw = os.getenv("PIPELINE_MAX_FRAMES_PER_SECTION_HARD", "12").strip()
    try:
        hard_cap = max(max_frames_per_section, int(hard_cap_raw))
    except ValueError:
        hard_cap = max(max_frames_per_section, 12)
    hard_cap = max(2, min(hard_cap, 20))

    for group_idx, (key, group) in enumerate(groups):
        group_text_len = 0
        group_figs = 0
        for frame in group:
            title = str(frame.get("frame_title") or "")
            bullets = frame.get("bullets") or []
            if isinstance(bullets, str):
                bullets = [bullets]
            paragraphs = frame.get("paragraphs") or []
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            group_text_len += len(title)
            group_text_len += sum(len(str(x)) for x in bullets)
            group_text_len += sum(len(str(x)) for x in paragraphs)
            slugs = frame.get("figure_slugs") or []
            if isinstance(slugs, str):
                slugs = [slugs]
            group_figs += len([s for s in slugs if s])

        adaptive_cap = max_frames_per_section
        if group_text_len > 1800:
            adaptive_cap += 1
        if group_text_len > 3000:
            adaptive_cap += 1
        if group_text_len > 4500:
            adaptive_cap += 1
        if group_figs > 4:
            adaptive_cap += 1
        if group_figs > 8:
            adaptive_cap += 1
        adaptive_cap = max(max_frames_per_section, min(adaptive_cap, hard_cap))

        # Soft linkage to global target: in standard/detailed profiles, avoid
        # over-capping early sections when the whole deck is still below the
        # desired lower bound.
        if isinstance(target_min_pages, int) and target_min_pages > 0 and detail_profile != "compact":
            remaining_groups = max(1, len(groups) - group_idx)
            remaining_need = max(0, target_min_pages - len(capped_frames))
            if remaining_need > 0:
                floor_for_this_group = max(1, (remaining_need + remaining_groups - 1) // remaining_groups)
                adaptive_cap = max(adaptive_cap, min(hard_cap, floor_for_this_group))

        if len(group) <= adaptive_cap:
            capped_frames.extend(group)
            continue

        priority = [f for f in group if _is_priority_frame(f)]
        image_frames = [f for f in group if _has_image(f) and f not in priority]
        others = [f for f in group if f not in priority and f not in image_frames]

        kept: list[dict] = []
        for bucket in (priority, image_frames, others):
            for frame in bucket:
                if frame in kept:
                    continue
                kept.append(frame)
                if len(kept) >= adaptive_cap:
                    break
            if len(kept) >= adaptive_cap:
                break

        if group and group[0] not in kept:
            if kept:
                kept[-1] = group[0]
            else:
                kept = [group[0]]

        kept_ids = {id(f) for f in kept}
        for frame in group:
            if id(frame) in kept_ids:
                capped_frames.append(frame)

        section, subsection = key
        sec_label = section or "Section"
        sub_label = f"/{subsection}" if subsection else ""
        print(
            f"[Pipeline] Capped section frames to {len(kept)} for '{sec_label}{sub_label}' "
            f"(original={len(group)}, requested={max_frames_per_section}, adaptive_cap={adaptive_cap})"
        )

    return capped_frames


def _trim_figures_per_frame(frames: list[dict], max_figs_per_frame: int) -> list[dict]:
    if not frames or not isinstance(max_figs_per_frame, int) or max_figs_per_frame <= 0:
        return frames

    trimmed: list[dict] = []
    reduced = 0

    for frame in frames:
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        deduped_slugs: list[str] = []
        seen: set[str] = set()
        for slug in slugs:
            if not slug or slug in seen:
                continue
            seen.add(slug)
            deduped_slugs.append(slug)
        if len(deduped_slugs) > max_figs_per_frame:
            new_frame = dict(frame)
            new_frame["figure_slugs"] = deduped_slugs[:max_figs_per_frame]
            trimmed.append(new_frame)
            reduced += 1
        else:
            frame["figure_slugs"] = deduped_slugs
            trimmed.append(frame)

    if reduced:
        print(
            f"[Pipeline] Limited figures per frame to {max_figs_per_frame}: trimmed {reduced} frames."
        )

    return trimmed


_MOJIBAKE_HINT_TOKENS = (
    "锟",
    "鐮",
    "寮",
    "鈥",
    "锛",
    "銆",
    "鏈€",
    "鐨",
    "鍥",
    "鐩",
    "绗",
    "缁",
    "鏂",
    "闂",
    "鎬",
    "鎻",
    "妯",
    "鍔",
    "绛",
    "鎸",
    "纭",
    "闄",
    "鏈",
    "璇",
    "\ufffd",
)


def _looks_like_mojibake_text(text: str, lang: str = "zh") -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if any(token in s for token in _MOJIBAKE_HINT_TOKENS):
        return True
    if "nnnn" in s or "IIII" in s:
        return True
    if re.search(r"(?:\bn{6,}\b|\bI{6,}\b|[nI]{10,})", s):
        return True
    if re.search(r"[\u2500-\u257F]", s):
        return True
    q = evaluate_text_quality(s, lang=lang)
    return bool(q.get("is_mojibake"))


def _split_long_text_segments(text: str, max_len: int = 180) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    if len(s) <= max_len:
        return [s]
    parts = [p.strip() for p in re.split(r"(?<=[。！？.!?；;])\s+", s) if p.strip()]
    if not parts:
        parts = [s]
    merged: list[str] = []
    buf = ""
    for part in parts:
        if not buf:
            buf = part
            continue
        if len(buf) + 1 + len(part) <= max_len:
            buf = f"{buf} {part}"
        else:
            merged.append(buf)
            buf = part
    if buf:
        merged.append(buf)
    return merged if merged else [s]


def _sanitize_line_for_render(text: str, lang: str = "zh", fallback: str = "") -> tuple[str, bool]:
    raw = (text or "").strip()
    if not raw:
        return fallback, False
    changed = False
    if _looks_like_mojibake_text(raw, lang=lang):
        changed = True
    cleaned = raw
    for token in _MOJIBAKE_HINT_TOKENS:
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("nnnn", " ").replace("IIII", " ")
    cleaned = re.sub(r"(?:\bn{6,}\b|\bI{6,}\b|[nI]{10,})", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = fallback
    elif _looks_like_mojibake_text(cleaned, lang=lang):
        cleaned = fallback

    # Preflight: keep inline math stable and avoid odd/unbalanced delimiters.
    if cleaned:
        unescaped_dollar = len(re.findall(r"(?<!\\)\$", cleaned))
        if unescaped_dollar % 2 == 1:
            cleaned = cleaned.replace("$", r"\$")
            changed = True
        open_disp = cleaned.count(r"\[")
        close_disp = cleaned.count(r"\]")
        if open_disp > close_disp:
            cleaned = cleaned + (" \\]" * (open_disp - close_disp))
            changed = True
        elif close_disp > open_disp:
            cleaned = ("\\[ " * (close_disp - open_disp)) + cleaned
            changed = True
        cleaned = cleaned.replace("\ufffd", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or fallback, changed


def _sanitize_frames_before_render(
    frames: list[dict],
    lang: str,
) -> tuple[list[dict], int]:
    if not frames:
        return frames, 0
    sanitized: list[dict] = []
    hits = 0
    title_fallback = "内容分析" if str(lang).lower().startswith("zh") else "Analysis"

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        new_frame = dict(frame)
        title, changed_title = _sanitize_line_for_render(
            str(frame.get("frame_title") or ""),
            lang=lang,
            fallback=title_fallback,
        )
        new_frame["frame_title"] = title
        if changed_title:
            hits += 1

        bullets = frame.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        sanitized_bullets: list[str] = []
        for item in bullets:
            cleaned, changed = _sanitize_line_for_render(str(item or ""), lang=lang, fallback="")
            if changed:
                hits += 1
            for seg in _split_long_text_segments(cleaned, max_len=180):
                if seg:
                    sanitized_bullets.append(seg)
        new_frame["bullets"] = sanitized_bullets[:12]

        paragraphs = frame.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        sanitized_paragraphs: list[str] = []
        for para in paragraphs:
            cleaned, changed = _sanitize_line_for_render(str(para or ""), lang=lang, fallback="")
            if changed:
                hits += 1
            for seg in _split_long_text_segments(cleaned, max_len=240):
                if seg:
                    sanitized_paragraphs.append(seg)
        new_frame["paragraphs"] = sanitized_paragraphs[:8]
        sanitized.append(new_frame)
    return sanitized, hits


def _preflight_latex_source(tex_path: str) -> dict:
    path = Path(tex_path).expanduser().resolve()
    if not path.exists():
        return {"changed": False, "error": f"tex file not found: {path}"}
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        return {"changed": False, "error": str(exc)}

    original = src
    changes = 0

    # Remove replacement/control characters that often break xelatex parsing.
    cleaned = src.replace("\ufffd", "")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    if cleaned != src:
        changes += 1
    src = cleaned

    # Balance display/inline math delimiters in a conservative way.
    for left, right in ((r"\[", r"\]"), (r"\(", r"\)")):
        l_count = src.count(left)
        r_count = src.count(right)
        if l_count > r_count:
            missing = l_count - r_count
            src = src.replace(r"\end{document}", (right + "\n") * missing + r"\end{document}")
            changes += 1

    # If standalone unescaped $ count is odd, escape them to prevent hard fail.
    dollar_count = len(re.findall(r"(?<!\\)\$", src))
    if dollar_count % 2 == 1:
        src = re.sub(r"(?<!\\)\$", r"\\$", src)
        changes += 1

    # Split overlong plain-text lines for readability/stability.
    new_lines: list[str] = []
    long_split_applied = False
    for line in src.splitlines():
        stripped = line.strip()
        if len(line) > 320 and not stripped.startswith("\\"):
            parts = re.split(r"(?<=[。！？.!?；;])\s+", line)
            if len(parts) > 1:
                new_lines.extend([p for p in parts if p])
                long_split_applied = True
                continue
        new_lines.append(line)
    if long_split_applied:
        src = "\n".join(new_lines)
        changes += 1

    if src != original:
        try:
            path.write_text(src, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return {"changed": False, "error": str(exc)}
        return {"changed": True, "error": "", "changes": changes}
    return {"changed": False, "error": "", "changes": 0}


def _expand_frames_to_target_min(
    frames: list[dict],
    aligned_figures: list[dict],
    target_min_pages: int,
    target_max_pages: int,
    lang: str,
) -> list[dict]:
    if not frames:
        return frames
    if len(frames) >= target_min_pages:
        return frames

    expanded = [dict(f) for f in frames]

    # Pass 1: split dense text frames.
    idx = 0
    while len(expanded) < target_min_pages and idx < len(expanded):
        frame = expanded[idx]
        bullets = frame.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        paragraphs = frame.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        if len(bullets) >= 5:
            half = max(2, len(bullets) // 2)
            first = dict(frame)
            second = dict(frame)
            first["bullets"] = bullets[:half]
            second["bullets"] = bullets[half:]
            title = str(frame.get("frame_title") or "")
            cont = "（续）" if str(lang).lower().startswith("zh") else " (Cont.)"
            second["frame_title"] = f"{title}{cont}" if title else ("补充分析" if str(lang).lower().startswith("zh") else "Additional Analysis")
            expanded[idx] = first
            expanded.insert(idx + 1, second)
            idx += 2
            continue
        if len(paragraphs) >= 2 and sum(len(p) for p in paragraphs) > 320:
            first = dict(frame)
            second = dict(frame)
            first["paragraphs"] = paragraphs[:1]
            second["paragraphs"] = paragraphs[1:]
            title = str(frame.get("frame_title") or "")
            cont = "（续）" if str(lang).lower().startswith("zh") else " (Cont.)"
            second["frame_title"] = f"{title}{cont}" if title else ("补充分析" if str(lang).lower().startswith("zh") else "Additional Analysis")
            expanded[idx] = first
            expanded.insert(idx + 1, second)
            idx += 2
            continue
        idx += 1

    # Pass 2: add figure-centric analysis frames for unused images.
    if len(expanded) < target_min_pages and aligned_figures:
        used_slugs: set[str] = set()
        for frame in expanded:
            slugs = frame.get("figure_slugs") or []
            if isinstance(slugs, str):
                slugs = [slugs]
            for slug in slugs:
                if slug:
                    used_slugs.add(str(slug))
        for fig in aligned_figures:
            if len(expanded) >= target_min_pages:
                break
            slug = fig.get("slug")
            path_str = fig.get("path")
            if not slug or not path_str or slug in used_slugs:
                continue
            caption = str(fig.get("caption") or "").strip()
            context = str(fig.get("context") or "").strip()
            section = str(fig.get("section") or "").strip() or (
                "研究内容分析" if str(lang).lower().startswith("zh") else "Main Analysis"
            )
            if str(lang).lower().startswith("zh"):
                frame_title = f"{section}：图表证据 {len(expanded) + 1}"
                bullets = []
                if caption:
                    bullets.append(f"图表主题：{caption[:120]}")
                if context:
                    bullets.append(f"关键解读：{context[:140]}")
                bullets.append("结合原文上下文补充解释该图在研究结论中的作用。")
            else:
                frame_title = f"{section}: Figure Evidence {len(expanded) + 1}"
                bullets = []
                if caption:
                    bullets.append(f"Figure focus: {caption[:120]}")
                if context:
                    bullets.append(f"Context: {context[:140]}")
                bullets.append("Interpret this figure in relation to the section's key findings.")
            expanded.append(
                {
                    "section": section,
                    "subsection": fig.get("subsection"),
                    "frame_title": frame_title,
                    "bullets": bullets,
                    "paragraphs": [],
                    "figure_slugs": [slug],
                    "source_pages": [fig.get("page")] if isinstance(fig.get("page"), int) else [],
                    "source_chunk_indices": [],
                }
            )
            used_slugs.add(str(slug))

    # Pass 3: if still below target, create conservative readable supplement frames.
    seed_frames = [f for f in expanded if isinstance(f, dict)]
    seed_idx = 0
    while len(expanded) < target_min_pages and seed_frames:
        base = seed_frames[seed_idx % len(seed_frames)]
        seed_idx += 1
        section = str(base.get("section") or "").strip() or ("研究分析" if str(lang).lower().startswith("zh") else "Analysis")
        title = str(base.get("frame_title") or "").strip()
        if str(lang).lower().startswith("zh"):
            frame_title = f"{title}（补充）" if title else f"{section}补充分析"
            bullets = [
                "补充说明关键变量之间的关系与变化趋势。",
                "给出方法假设、样本边界与结果可解释性的进一步讨论。",
                "结合业务/政策场景提出可执行建议。",
            ]
        else:
            frame_title = f"{title} (Supplement)" if title else f"{section} Supplement"
            bullets = [
                "Clarify variable relationships and trend behavior.",
                "Discuss assumptions, sampling boundaries, and interpretability.",
                "Provide actionable recommendations from the findings.",
            ]
        expanded.append(
            {
                "section": section,
                "subsection": base.get("subsection"),
                "frame_title": frame_title,
                "bullets": bullets,
                "paragraphs": [],
                "figure_slugs": [],
                "source_pages": base.get("source_pages") or [],
                "source_chunk_indices": base.get("source_chunk_indices") or [],
            }
        )

    if len(expanded) > target_max_pages:
        expanded = expanded[:target_max_pages]
    return expanded


def _cap_frames_to_max(
    frames: list[dict],
    aligned_figures: list[dict],
    max_frames: int,
) -> list[dict]:
    """Hard-cap the number of content frames while preserving user images.

    Frames that contain at least one concrete user image (i.e., a figure slug
    whose metadata has a non-empty ``path``) are prioritized and kept first in
    their original order. Remaining budget, if any, is filled with the earliest
    text-only frames so that the overall deck stays roughly faithful to the
    paper's flow.
    """

    if not frames or not isinstance(max_frames, int) or max_frames <= 0:
        return frames
    if len(frames) <= max_frames:
        return frames

    figures_by_slug: dict[str, dict] = {}
    for fig in aligned_figures or []:
        slug = fig.get("slug")
        if not slug:
            continue
        figures_by_slug[slug] = fig

    def _is_priority_frame(frame: dict) -> bool:
        text = " ".join(
            str(frame.get(key) or "") for key in ("frame_title", "section", "subsection")
        )
        if not text:
            return False
        cn_keywords = [
            "概述",
            "摘要",
            "贡献",
            "总结",
            "要点",
            "小结",
            "结论",
            "建议",
            "总结与展望",
            "结论与建议",
            "未来工作",
        ]
        en_keywords = [
            "overview",
            "summary",
            "contribution",
            "takeaway",
            "key result",
            "highlights",
            "conclusion",
            "recommendation",
            "future work",
        ]
        if any(kw in text for kw in cn_keywords):
            return True
        text_lower = text.lower()
        return any(kw in text_lower for kw in en_keywords)

    indexed: list[dict] = []
    for idx, frame in enumerate(frames):
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        has_image = False
        for slug in slugs:
            fig = figures_by_slug.get(slug)
            path_str = fig.get("path") if fig else None
            if path_str:
                has_image = True
                break
        indexed.append(
            {
                "index": idx,
                "frame": frame,
                "has_image": has_image,
                "is_priority": _is_priority_frame(frame),
            }
        )

    priority_frames = [item for item in indexed if item["is_priority"]]
    image_frames = [item for item in indexed if item["has_image"] and not item["is_priority"]]
    text_frames = [
        item for item in indexed if not item["has_image"] and not item["is_priority"]
    ]

    # Always keep priority frames and frames with concrete user images when possible.
    kept: list[dict] = []
    for item in priority_frames:
        kept.append(item)

    for item in image_frames:
        kept.append(item)

    if len(kept) >= max_frames:
        # Preserve all priority + image frames even if this exceeds the cap.
        kept.sort(key=lambda x: x["index"])
        if len(kept) > max_frames:
            print(
                f"[Pipeline] max_frames={max_frames} is lower than the number of priority/image frames "
                f"({len(kept)}); keeping all image-heavy frames."
            )
        return [item["frame"] for item in kept]

    remaining = max_frames - len(kept)
    if remaining > 0:
        kept.extend(text_frames[:remaining])

    # Restore original document order.
    kept.sort(key=lambda x: x["index"])
    return [item["frame"] for item in kept]


def _resolve_effective_max_frames(
    max_frames_value: int | None,
    section_blocks: list[dict],
    max_frames_per_section: int,
) -> int:
    try:
        current_max = int(max_frames_value or DEFAULT_MAX_FRAMES)
    except Exception:
        current_max = DEFAULT_MAX_FRAMES
    current_max = max(10, current_max)

    hard_raw = os.getenv("PIPELINE_MAX_FRAMES_HARD", "180").strip()
    try:
        hard_cap = int(hard_raw)
    except ValueError:
        hard_cap = 180
    hard_cap = max(20, hard_cap)

    adaptive_enabled = os.getenv("PIPELINE_ADAPTIVE_MAX_FRAMES", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not adaptive_enabled:
        return min(current_max, hard_cap)

    blocks = list(section_blocks or [])
    block_count = len(blocks)
    if block_count <= 0:
        return min(current_max, hard_cap)

    total_chars = 0
    for block in blocks:
        text = (block or {}).get("text") or ""
        total_chars += len(text)

    avg_chars = total_chars / max(1, block_count)
    per_section_target = max(2.4, min(float(max_frames_per_section) * 0.9, 6.5))
    if avg_chars > 1800:
        per_section_target += 0.8
    if avg_chars > 2600:
        per_section_target += 0.8
    if avg_chars > 3600:
        per_section_target += 0.8

    recommended = int(block_count * per_section_target)
    recommended = max(36, recommended)
    if total_chars > 90000:
        recommended += 10
    if total_chars > 140000:
        recommended += 12
    recommended = min(recommended, hard_cap)

    # Only auto-raise when the current value looks like a default-ish cap.
    if current_max <= DEFAULT_MAX_FRAMES + 2 and recommended > current_max:
        return recommended
    return min(current_max, hard_cap)


def _is_conclusion_recommendation_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    s_lower = s.lower()
    en_keywords = (
        "conclusion",
        "conclusions",
        "recommendation",
        "recommendations",
        "future work",
        "future research",
        "implication",
    )
    if any(kw in s_lower for kw in en_keywords):
        return True
    cn_keywords = (
        "结论",
        "总结",
        "建议",
        "对策",
        "政策建议",
        "展望",
        "未来工作",
        "启示",
    )
    return any(kw in s for kw in cn_keywords)


def _ensure_conclusion_frames(frames: list[dict], lang: str = "zh") -> tuple[list[dict], bool]:
    if not frames:
        return frames, False
    for frame in frames:
        text = " ".join(
            str(frame.get(key) or "") for key in ("frame_title", "section", "subsection")
        )
        if _is_conclusion_recommendation_text(text):
            return frames, False

    lang_norm = (lang or "zh").lower()
    if lang_norm.startswith("zh"):
        section_name = "结论与建议"
        frame_title = "结论与建议"
        bullets = [
            "归纳核心研究结论，明确关键影响因素与作用路径。",
            "给出分层、可执行的管理与政策建议，匹配不同人群与场景。",
            "后续工作建议：扩展样本覆盖并开展长期追踪验证。",
        ]
    else:
        section_name = "Conclusion and Recommendations"
        frame_title = section_name
        bullets = [
            "Summarize the key findings and primary impact pathways.",
            "Provide actionable, segmented recommendations for different scenarios.",
            "Future work: expand sample coverage and run longitudinal validation.",
        ]

    source_pages: list[int] = []
    source_chunk_indices: list[int] = []
    for frame in reversed(frames):
        pages = frame.get("source_pages") or []
        if isinstance(pages, list) and pages:
            source_pages = [p for p in pages if isinstance(p, int)]
            break
    for frame in reversed(frames):
        chunks = frame.get("source_chunk_indices") or []
        if isinstance(chunks, list) and chunks:
            source_chunk_indices = [c for c in chunks if isinstance(c, int)]
            break

    appended = list(frames)
    appended.append(
        {
            "section": section_name,
            "subsection": None,
            "frame_title": frame_title,
            "bullets": bullets,
            "paragraphs": [],
            "figure_slugs": [],
            "source_pages": source_pages,
            "source_chunk_indices": source_chunk_indices,
        }
    )
    print(f"[Pipeline] Injected fallback frame for '{section_name}'.")
    return appended, True


def _reorder_conclusion_recommendation_frames(frames: list[dict]) -> list[dict]:
    """Move conclusion/recommendation-related frames to the end of the deck."""

    if not frames or len(frames) == 1:
        return frames

    def _is_tail_frame(frame: dict) -> bool:
        text = " ".join(
            str(frame.get(key) or "") for key in ("frame_title", "section", "subsection")
        ).strip()
        return _is_conclusion_recommendation_text(text)

    preserved = [frames[0]]
    head: list[dict] = []
    tail: list[dict] = []
    for frame in frames[1:]:
        if _is_tail_frame(frame):
            tail.append(frame)
        else:
            head.append(frame)
    if not tail:
        return frames
    print(f"[Pipeline] Moved {len(tail)} conclusion/recommendation frames to the end.")
    return preserved + head + tail


def _build_figure_frames_from_aligned(
    aligned_figures: list[dict],
    section_blocks: list[dict],
) -> list[dict]:
    """Create dedicated figure-only frames from aligned_figures.

    Each frame shows exactly one figure together with its original caption and
    nearby context text from the paper. The frame is anchored to a
    section/subsection determined by the section_blocks that cover the
    figure's page, so that the overall slide structure still follows the
    paper's logical sections.
    """

    if not aligned_figures:
        return []

    # Build a simple mapping from page -> (section, subsection) based on the
    # heuristic section blocks constructed from PDF chunks.
    page_to_section: dict[int, tuple[str, str | None]] = {}
    for block in section_blocks or []:
        section = (block.get("section") or "图表说明").strip()
        subsection = block.get("subsection")
        pages = block.get("pages") or []
        for p in pages:
            if isinstance(p, int) and p not in page_to_section:
                page_to_section[p] = (section, subsection)

    figure_frames: list[dict] = []

    # Keep figures in the natural reading order of the paper by sorting by
    # page number when available.
    def _fig_page_key(fig: dict) -> int:
        try:
            p = fig.get("page")
        except AttributeError:
            p = None
        if isinstance(p, int) and p > 0:
            return p
        return 10**9

    for fig in sorted(aligned_figures or [], key=_fig_page_key):
        slug = fig.get("slug")
        path_str = fig.get("path")
        if not slug or not path_str:
            continue

        page = fig.get("page")
        section = None
        subsection = None
        if isinstance(page, int) and page in page_to_section:
            section, subsection = page_to_section[page]
        if not section:
            section = "图表说明"

        caption = (fig.get("caption") or "").strip()
        context = (fig.get("context") or "").strip()

        frame_title = caption or section
        if len(frame_title) > 40:
            frame_title = frame_title[:40]

        paragraphs: list[str] = []
        if context:
            paragraphs.append(context)

        frame = {
            "section": section,
            "subsection": subsection,
            "frame_title": frame_title,
            "bullets": [],
            "paragraphs": paragraphs,
            "figure_slugs": [slug],
            "source_pages": [page] if isinstance(page, int) else [],
            "source_chunk_indices": [],
        }
        figure_frames.append(frame)

    print(f"[Pipeline] Built {len(figure_frames)} dedicated figure frames from aligned figures.")
    return figure_frames


def _normalize_newlines_inside_strings(s: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if ch == "\\" and not escape:
            escape = True
            out.append(ch)
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            out.append(ch)
            continue
        if ch in "\r\n" and in_string:
            out.append("\\n")
            escape = False
            continue
        out.append(ch)
        escape = False
    return "".join(out)


def _parse_json_payload(payload_text: str) -> dict | None:
    raw = (payload_text or "").strip()
    cleaned = raw
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()
    if cleaned and not cleaned.lstrip().startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    cleaned = _normalize_newlines_inside_strings(cleaned)

    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(cleaned)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict):
            return obj
        return None


def _is_cache_text_payload_mojibake(payload, lang: str = "zh") -> bool:
    """Best-effort check for mojibake in cached structured payloads."""
    lang_norm = str(lang or "zh").lower()
    if not lang_norm.startswith("zh"):
        return False

    texts: list[str] = []

    def _collect(node) -> None:
        if isinstance(node, str):
            s = node.strip()
            if s:
                texts.append(s)
            return
        if isinstance(node, dict):
            for v in node.values():
                _collect(v)
            return
        if isinstance(node, (list, tuple, set)):
            for v in node:
                _collect(v)

    _collect(payload)
    if not texts:
        return False

    sample = texts[:300]
    suspicious = 0
    for t in sample:
        if _looks_like_mojibake_text(t, lang=lang_norm):
            suspicious += 1
    if suspicious >= 3:
        return True
    if suspicious / max(1, len(sample)) >= 0.2:
        return True
    joined = "\n".join(sample)[:12000]
    q = evaluate_text_quality(joined, lang=lang_norm)
    return bool(q.get("is_mojibake"))


def _is_cache_text_payload_toc_like(payload) -> bool:
    texts: list[str] = []

    def _collect(node) -> None:
        if isinstance(node, str):
            s = node.strip()
            if s:
                texts.append(s)
            return
        if isinstance(node, dict):
            for v in node.values():
                _collect(v)
            return
        if isinstance(node, (list, tuple, set)):
            for v in node:
                _collect(v)

    _collect(payload)
    if not texts:
        return False
    joined = "\n".join(texts[:160])
    return _is_toc_like_text(joined)


def _build_conservative_outline_chunks(lang: str = "zh") -> list[dict]:
    zh = (lang or "zh").lower().startswith("zh")
    if zh:
        texts = [
            "研究背景与问题定义：由于源文档文本质量不足，自动进入保守提纲模式，优先保证中文可读性。",
            "研究方法与数据说明：建议结合原始 PDF/DOCX 复核关键样本、变量定义和统计口径。",
            "核心发现与建议：本轮输出聚焦结构化汇报框架，避免输出潜在乱码正文。",
        ]
    else:
        texts = [
            "Background and problem definition: conservative outline mode is enabled due to low text quality.",
            "Methods and data: verify key samples, variable definitions, and statistical settings in source files.",
            "Findings and recommendations: this run prioritizes readability and structure over noisy raw text.",
        ]
    chunks: list[dict] = []
    for i, text in enumerate(texts, start=1):
        chunks.append({"page_content": text, "metadata": {"page": i}})
    return chunks


def _split_sentences(text: str, lang: str) -> list[str]:
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if lang.startswith("zh"):
        # Split by Chinese full-stop/exclamation/question and keep punctuation.
        parts = re.split(r"(?<=[。！？!?])", cleaned)
    else:
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences: list[str] = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        if len(s) > 200:
            subparts = re.split(r"[;；]", s)
            for sub in subparts:
                sub = sub.strip()
                if sub:
                    sentences.append(sub)
        else:
            sentences.append(s)
    return sentences


def _chunk_list(items: list, chunk_size: int) -> list[list]:
    if not items:
        return []
    size = max(1, int(chunk_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _estimate_section_frame_target(
    source_text: str,
    retrieved_context: str,
    section_figures: list[dict] | None,
    section_formulas: list[dict] | None,
    max_frames_per_section: int,
    lang: str,
) -> int:
    max_frames = max(2, min(int(max_frames_per_section or 6), 10))
    combined = (source_text or "").strip()
    if retrieved_context:
        combined = f"{combined}\n{retrieved_context}" if combined else retrieved_context

    char_len = len(combined)
    sentence_count = len(_split_sentences(combined, lang))
    figure_count = len(section_figures or [])
    formula_count = len(section_formulas or [])

    # Heuristic complexity score for adaptive frame count.
    score = 0.0
    score += char_len / 1200.0
    score += sentence_count / 12.0
    score += figure_count * 0.7
    score += formula_count * 0.8

    target = 2 + int(score)
    if char_len > 2600:
        target += 1
    if sentence_count > 18:
        target += 1
    if figure_count >= 5:
        target += 1
    if formula_count >= 3:
        target += 1

    target = max(2, min(target, max_frames))
    return target


def _fallback_plan_frames(
    section: str,
    subsection: str | None,
    source_text: str,
    retrieved_context: str,
    section_figures: list[dict],
    lang: str,
    max_frames_per_section: int | None = None,
) -> list[dict]:
    combined = source_text or ""
    if retrieved_context:
        combined = f"{combined}\n{retrieved_context}" if combined else retrieved_context

    sentences = _split_sentences(combined, lang)
    seen: set[str] = set()
    uniq_sentences: list[str] = []
    for s in sentences:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq_sentences.append(s)

    bullets: list[str] = []
    for s in uniq_sentences:
        if len(bullets) >= 48:
            break
        if len(s) > 180:
            bullets.append(s[:180].rstrip() + "...")
        else:
            bullets.append(s)

    paragraph_units: list[str] = []
    if not bullets and combined:
        snippet = combined.strip()
        if len(snippet) > 320:
            snippet = snippet[:320].rstrip() + "..."
        paragraph_units = [snippet]
    elif uniq_sentences:
        for idx in range(0, len(uniq_sentences), 2):
            para = " ".join(uniq_sentences[idx : idx + 2]).strip()
            if not para:
                continue
            if len(para) > 320:
                para = para[:320].rstrip() + "..."
            paragraph_units.append(para)
            if len(paragraph_units) >= 24:
                break

    fig_slugs: list[str] = []
    for fig in section_figures or []:
        slug = fig.get("slug")
        if not slug and fig.get("path"):
            slug = Path(fig.get("path")).stem
        if slug:
            fig_slugs.append(slug)
    seen_figs: set[str] = set()
    fig_slugs = [s for s in fig_slugs if not (s in seen_figs or seen_figs.add(s))]

    if not bullets and not paragraph_units:
        fallback_text = "Summary text unavailable."
        if lang.startswith("zh"):
            fallback_text = "本节暂无可用文本摘要。"
        bullets = [fallback_text]

    max_figs = _get_max_figures_per_frame()
    if isinstance(max_frames_per_section, int):
        max_frames = max(2, min(max_frames_per_section, 10))
    else:
        max_frames = _get_max_frames_per_section()
    max_frames = max(2, min(max_frames, 10))

    title_base = subsection or section
    if not title_base:
        title_base = "Summary" if not lang.startswith("zh") else "内容概览"

    bullet_chunks = _chunk_list(bullets, 4)
    para_chunks = [[p] for p in paragraph_units]
    fig_chunks = _chunk_list(fig_slugs, max_figs)

    segment_count = max(1, len(bullet_chunks), len(para_chunks), len(fig_chunks))
    if len(bullet_chunks) > 2 or len(para_chunks) > 2 or len(fig_chunks) > 1:
        segment_count = min(max_frames, max(segment_count, 3))

    frames: list[dict] = []
    for idx in range(segment_count):
        frame_bullets = bullet_chunks[idx] if idx < len(bullet_chunks) else []
        frame_paragraphs = para_chunks[idx] if idx < len(para_chunks) else []
        frame_figs = fig_chunks[idx] if idx < len(fig_chunks) else []
        if not frame_bullets and not frame_paragraphs and not frame_figs and idx > 0:
            continue

        if idx == 0:
            frame_title = str(title_base)
        elif frame_figs and not frame_bullets:
            frame_title = "Key Figures" if not lang.startswith("zh") else "关键图表"
        elif frame_bullets and not frame_figs:
            if lang.startswith("zh"):
                frame_title = f"{title_base}（要点 {idx + 1}/{segment_count}）"
            else:
                frame_title = f"{title_base} (Points {idx + 1}/{segment_count})"
        else:
            if lang.startswith("zh"):
                frame_title = f"{title_base}（续 {idx + 1}/{segment_count}）"
            else:
                frame_title = f"{title_base} (Cont. {idx + 1}/{segment_count})"

        frames.append(
            {
                "section": section,
                "subsection": subsection,
                "frame_title": frame_title,
                "bullets": frame_bullets,
                "paragraphs": frame_paragraphs,
                "figure_slugs": frame_figs,
            }
        )
        if len(frames) >= max_frames:
            break

    return frames

def _recover_frames_from_broken_json(s: str):
    frames = []
    decoder = json.JSONDecoder()
    match = re.search(r"\"frames\"\s*:\s*\[", s)
    if not match:
        return frames
    start = match.end()
    body = s[start:]
    idx = 0
    length = len(body)
    while idx < length:
        while idx < length and body[idx] in " \r\n\t,":
            idx += 1
        if idx >= length:
            break
        try:
            obj, next_idx = decoder.raw_decode(body, idx)
        except Exception:
            break
        if not isinstance(obj, dict):
            break
        frames.append(obj)
        idx = next_idx
    return frames


def _parse_frames_from_model_output(plan_json_text: str):
    """Parse frames from a model JSON string with robust fallbacks.

    This function preserves the previous behavior of trying json.loads,
    trimming corrupted tails, and finally streaming-decoding individual
    frame objects if necessary.
    """
    raw = (plan_json_text or "").strip()
    cleaned = raw
    if _contains_model_limit_message(cleaned):
        print(
            "[Pipeline] Planner output indicates model call limits were exceeded; "
            "using fallback frames for this block."
        )
        return []
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()
    if cleaned and not cleaned.lstrip().startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    cleaned = _normalize_newlines_inside_strings(cleaned)

    plan_obj = None
    parse_error: Exception | None = None
    try:
        plan_obj = json.loads(cleaned)
    except Exception as exc:
        parse_error = exc

    if plan_obj is None:
        # Try increasingly aggressive trims from the end of the string.
        trim_steps = [64, 128, 256, 512, 768, 1024, 1536, 2048]
        for trim in trim_steps:
            if len(cleaned) <= trim + 10:
                break
            candidate = cleaned[:-trim]
            last_brace = candidate.rfind("}")
            if last_brace != -1:
                candidate = candidate[: last_brace + 1]
            try:
                plan_obj = json.loads(candidate)
                print(
                    f"[Pipeline] Parsed frames JSON successfully after trimming the last {trim} characters from the model output."
                )
                cleaned = candidate
                parse_error = None
                break
            except Exception as exc:
                parse_error = exc
                continue

    if plan_obj is None:
        print(f"[Pipeline] Failed to parse frames JSON from model: {parse_error}")
        cleaned_preview = cleaned[:500].replace("\n", " ")
        print("[Pipeline] Cleaned frames JSON preview (first 500 chars): " + cleaned_preview)
        frames = _recover_frames_from_broken_json(cleaned)
        if frames:
            print(
                f"[Pipeline] Recovered {len(frames)} frames from partially valid JSON by streaming decode."
            )
        else:
            frames = []
    else:
        frames = plan_obj.get("frames") or []

    return frames


def _extract_structured_frames(response) -> list[dict] | None:
    if not isinstance(response, dict):
        return None
    sr = response.get("structured_response")
    if sr is None:
        return None
    try:
        if hasattr(sr, "model_dump"):
            obj = sr.model_dump()
        elif hasattr(sr, "dict"):
            obj = sr.dict()
        elif isinstance(sr, dict):
            obj = sr
        else:
            obj = None
    except Exception:  # noqa: BLE001
        obj = None
    if isinstance(obj, dict):
        frames = obj.get("frames") or []
        if isinstance(frames, list):
            return frames
    return None


def _extract_response_text(response) -> str:
    if isinstance(response, dict):
        messages = response.get("messages") or []
        if messages:
            last = messages[-1]
            return (getattr(last, "content", "") or "").strip()
    return (getattr(response, "content", "") or "").strip()


def _plan_frames_for_section(
    model,
    instructions: dict,
    block: dict,
    aligned_figures,
    keyword_index,
    formula_candidates,
):
    """Ask the model to plan frames for a single section block.

    The model only sees the text for this block (source_text) and optional
    figures that fall on the same pages, so that bullets stay aligned with
    the original content.
    """
    section = block.get("section") or "Section"
    subsection = block.get("subsection")
    source_text = block.get("text") or ""
    pages = block.get("pages") or []
    chunk_indices = block.get("chunk_indices") or []
    lang = (instructions.get("lang") or "zh").lower()
    global_planning_context = str(instructions.get("global_planning_context") or "").strip()
    outline_sections = [
        str(s).strip()
        for s in (instructions.get("outline_sections") or [])
        if str(s).strip()
    ]
    section_order_index = None
    section_order_total = len(outline_sections) if outline_sections else None
    previous_section = None
    next_section = None
    if outline_sections and section in outline_sections:
        pos = outline_sections.index(section)
        section_order_index = pos + 1
        if pos > 0:
            previous_section = outline_sections[pos - 1]
        if pos + 1 < len(outline_sections):
            next_section = outline_sections[pos + 1]
    max_frames_per_section = _resolve_max_frames_per_section(instructions)

    retrieved_context = ""
    if keyword_index:
        query_parts = [section or "", subsection or ""]
        query = " ".join(part for part in query_parts if part).strip()
        if not query:
            query = (source_text or "")[:200]
        results = keyword_search(
            keyword_index,
            query=query,
            pages=None,
            k=8,
            preferred_pages=pages or None,
        )
        if results:
            snippets: list[str] = []
            seen_texts: set[str] = set()
            for item in results:
                text_snippet = (item.get("page_content") or "").strip()
                if not text_snippet:
                    continue
                if text_snippet in seen_texts:
                    continue
                seen_texts.add(text_snippet)
                snippets.append(text_snippet)
            if snippets:
                retrieved_context = "\n\n".join(snippets)

    section_figures = []
    pages_set = set(pages)
    for fig in aligned_figures or []:
        try:
            fig_page = fig.get("page")
        except AttributeError:
            fig_page = None
        if fig_page and fig_page in pages_set:
            section_figures.append(fig)

    section_figures = _filter_section_figures_by_text(
        section_figures,
        source_text,
        retrieved_context,
    )

    section_formulas = []
    for formula in formula_candidates or []:
        try:
            f_page = formula.get("page")
        except AttributeError:
            f_page = None
        if f_page and f_page in pages_set:
            section_formulas.append(formula)

    target_frames = _estimate_section_frame_target(
        source_text,
        retrieved_context,
        section_figures,
        section_formulas,
        max_frames_per_section,
        lang,
    )
    print(
        f"[Pipeline] Planner target frames for section='{section}'"
        + (f", subsection='{subsection}'" if subsection else "")
        + f": {target_frames} (max_per_section={max_frames_per_section})"
    )

    planning_payload = {
        "task": "plan_frames_for_one_section",
        "instructions": instructions,
        "target_frames": target_frames,
        "section": section,
        "subsection": subsection,
        "source_text": source_text,
        "retrieved_context": retrieved_context,
        "source_pages": pages,
        "source_chunk_indices": chunk_indices,
        "section_figures": section_figures,
        "section_formulas": section_formulas,
        "global_planning_context": global_planning_context,
        "section_order_index": section_order_index,
        "section_order_total": section_order_total,
        "previous_section": previous_section,
        "next_section": next_section,
    }

    timeout_raw = os.getenv("PIPELINE_PLANNER_TIMEOUT", "180").strip()
    try:
        timeout_s = int(timeout_raw)
    except ValueError:
        timeout_s = 180
    planner_recursion_limit = _get_planner_recursion_limit()

    def _invoke_planner(force_json_only: bool = False):
        payload = dict(planning_payload)
        if force_json_only:
            payload["force_json_only"] = True
            payload["output_schema"] = {
                "frames": [
                    {
                        "section": "string",
                        "subsection": "string or null",
                        "frame_title": "string",
                        "bullets": ["string"],
                        "paragraphs": ["string"],
                        "figure_slugs": ["string"],
                    }
                ]
            }
        request = {
            "messages": [
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        }
        invoke_config = {"recursion_limit": planner_recursion_limit}
        if timeout_s > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.invoke, request, config=invoke_config)
                return future.result(timeout=timeout_s)
        return model.invoke(request, config=invoke_config)

    frames = None
    limit_hit = False
    try:
        response = _invoke_planner(force_json_only=False)
        frames = _extract_structured_frames(response)
        if frames is None:
            plan_json_text = _extract_response_text(response)
            if _contains_model_limit_message(plan_json_text):
                limit_hit = True
                frames = []
            else:
                frames = _parse_frames_from_model_output(plan_json_text)
    except TimeoutError:
        print(
            f"[Pipeline] Planner LLM call timed out after {timeout_s}s for section '{section}'; using fallback."
        )
        frames = _fallback_plan_frames(
            section,
            subsection,
            source_text,
            retrieved_context,
            section_figures,
            lang,
            target_frames,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Planner LLM call failed for section '{section}': {exc}")
        frames = _fallback_plan_frames(
            section,
            subsection,
            source_text,
            retrieved_context,
            section_figures,
            lang,
            target_frames,
        )

    if not frames and not limit_hit:
        try:
            response = _invoke_planner(force_json_only=True)
            frames = _extract_structured_frames(response)
            if frames is None:
                retry_text = _extract_response_text(response)
                if _contains_model_limit_message(retry_text):
                    limit_hit = True
                    frames = []
                else:
                    frames = _parse_frames_from_model_output(retry_text)
        except TimeoutError:
            print(
                f"[Pipeline] Planner LLM retry timed out after {timeout_s}s for section '{section}'; using fallback."
            )
            frames = _fallback_plan_frames(
                section,
                subsection,
                source_text,
                retrieved_context,
                section_figures,
                lang,
                target_frames,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Pipeline] Planner LLM retry failed for section '{section}': {exc}")
            frames = _fallback_plan_frames(
                section,
                subsection,
                source_text,
                retrieved_context,
                section_figures,
                lang,
                target_frames,
            )

    if not frames:
        frames = _fallback_plan_frames(
            section,
            subsection,
            source_text,
            retrieved_context,
            section_figures,
            lang,
            target_frames,
        )

    processed_frames = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        pf = dict(frame)
        # Enforce section/subsection from the block
        pf["section"] = section
        pf["subsection"] = subsection
        pf.setdefault("frame_title", subsection or section)

        bullets = pf.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        pf["bullets"] = bullets

        paragraphs = pf.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        pf["paragraphs"] = paragraphs

        figure_slugs = pf.get("figure_slugs") or []
        if isinstance(figure_slugs, str):
            figure_slugs = [figure_slugs]
        pf["figure_slugs"] = figure_slugs

        pf["source_pages"] = pages
        pf["source_chunk_indices"] = chunk_indices

        processed_frames.append(pf)

    print(
        f"[Pipeline] Planned {len(processed_frames)} frames for section='{section}'"
        + (f", subsection='{subsection}'" if subsection else "")
    )
    return processed_frames


def _plan_frames_for_sections_batch(
    model,
    instructions: dict,
    section_blocks: list[dict],
    aligned_figures,
    keyword_index,
    formula_candidates,
):
    """Plan frames for all section blocks, using batched LLM calls.

    This mirrors the behavior of _plan_frames_for_section but prepares a
    batched request for each section or subsection, runs them with limited
    concurrency via model.batch, and then post-processes the results into a
    single frames list.
    """
    if not section_blocks:
        print("[Pipeline] No section blocks to plan frames for.")
        return []

    lang = (instructions.get("lang") or "zh").lower()
    global_planning_context = str(instructions.get("global_planning_context") or "").strip()
    max_frames_per_section = _resolve_max_frames_per_section(instructions)
    requests: list[dict] = []
    metas: list[dict] = []
    aligned_figs_list = list(aligned_figures or [])
    formulas_list = list(formula_candidates or [])

    ordered_sections = [
        str((b or {}).get("section") or "").strip() or "Section"
        for b in section_blocks
    ]

    for block_idx, block in enumerate(section_blocks):
        section = block.get("section") or "Section"
        subsection = block.get("subsection")
        source_text = block.get("text") or ""
        pages = block.get("pages") or []
        chunk_indices = block.get("chunk_indices") or []

        # Build retrieved_context using keyword_index and section pages.
        retrieved_context = ""
        if keyword_index:
            query_parts = [section or "", subsection or ""]
            query = " ".join(part for part in query_parts if part).strip()
            if not query:
                query = (source_text or "")[:200]
            results = keyword_search(
                keyword_index,
                query=query,
                pages=None,
                k=8,
                preferred_pages=pages or None,
            )
            if results:
                snippets: list[str] = []
                seen_texts: set[str] = set()
                for item in results:
                    text_snippet = (item.get("page_content") or "").strip()
                    if not text_snippet:
                        continue
                    if text_snippet in seen_texts:
                        continue
                    seen_texts.add(text_snippet)
                    snippets.append(text_snippet)
                if snippets:
                    retrieved_context = "\n\n".join(snippets)

        pages_set = set(pages)
        section_figures = []
        for fig in aligned_figs_list:
            try:
                fig_page = fig.get("page")
            except AttributeError:
                fig_page = None
            if fig_page and fig_page in pages_set:
                section_figures.append(fig)

        section_figures = _filter_section_figures_by_text(
            section_figures,
            source_text,
            retrieved_context,
        )

        section_formulas = []
        for formula in formulas_list:
            try:
                f_page = formula.get("page")
            except AttributeError:
                f_page = None
            if f_page and f_page in pages_set:
                section_formulas.append(formula)

        target_frames = _estimate_section_frame_target(
            source_text,
            retrieved_context,
            section_figures,
            section_formulas,
            max_frames_per_section,
            lang,
        )

        planning_payload = {
            "task": "plan_frames_for_one_section",
            "instructions": instructions,
            "target_frames": target_frames,
            "section": section,
            "subsection": subsection,
            "source_text": source_text,
            "retrieved_context": retrieved_context,
            "source_pages": pages,
            "source_chunk_indices": chunk_indices,
            "section_figures": section_figures,
            "section_formulas": section_formulas,
            "global_planning_context": global_planning_context,
            "section_order_index": block_idx + 1,
            "section_order_total": len(section_blocks),
            "previous_section": ordered_sections[block_idx - 1] if block_idx > 0 else None,
            "next_section": ordered_sections[block_idx + 1] if block_idx + 1 < len(ordered_sections) else None,
        }

        messages = [
            HumanMessage(content=json.dumps(planning_payload, ensure_ascii=False)),
        ]
        requests.append({"messages": messages})
        metas.append(
            {
                "section": section,
                "subsection": subsection,
                "pages": pages,
                "chunk_indices": chunk_indices,
                "retrieved_context": retrieved_context,
                "section_figures": section_figures,
                "target_frames": target_frames,
                "block": block,
            }
        )

    if not requests:
        print("[Pipeline] Section planning: no sections to process after filtering.")
        return []

    max_concurrency = _get_planner_max_concurrency()
    planner_recursion_limit = _get_planner_recursion_limit()
    planner_batch_chunk_size = _get_planner_batch_chunk_size()
    if planner_batch_chunk_size > 16:
        print(
            f"[Pipeline] Planner batch chunk size {planner_batch_chunk_size} exceeds run_limit-safe threshold; "
            "capping to 16."
        )
        planner_batch_chunk_size = 16
    total_chunks = (len(requests) + planner_batch_chunk_size - 1) // planner_batch_chunk_size
    print(
        f"[Pipeline] Section planning: batching {len(requests)} blocks in {total_chunks} chunks "
        f"(chunk_size={planner_batch_chunk_size}, max_concurrency={max_concurrency}, recursion_limit={planner_recursion_limit})"
    )

    batch_timeout_raw = os.getenv("PIPELINE_PLANNER_BATCH_TIMEOUT", "300").strip()
    try:
        batch_timeout_s = int(batch_timeout_raw)
    except ValueError:
        batch_timeout_s = 300

    batch_config = {
        "max_concurrency": max_concurrency,
        "recursion_limit": planner_recursion_limit,
    }

    def _run_batch(reqs: list[dict]):
        if batch_timeout_s > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    model.batch,
                    reqs,
                    config=batch_config,
                )
                return future.result(timeout=batch_timeout_s)
        return model.batch(reqs, config=batch_config)

    all_frames: list[dict] = []
    total_blocks = len(section_blocks)
    processed_blocks = 0

    def _normalize_frames_for_meta(raw_frames: list[dict], meta: dict) -> list[dict]:
        normalized: list[dict] = []
        section = meta.get("section") or "Section"
        subsection = meta.get("subsection")
        pages = meta.get("pages") or []
        chunk_indices = meta.get("chunk_indices") or []
        for frame in raw_frames or []:
            if not isinstance(frame, dict):
                continue
            pf = dict(frame)
            pf["section"] = section
            pf["subsection"] = subsection
            pf.setdefault("frame_title", subsection or section)

            bullets = pf.get("bullets") or []
            if isinstance(bullets, str):
                bullets = [bullets]
            pf["bullets"] = bullets

            paragraphs = pf.get("paragraphs") or []
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            pf["paragraphs"] = paragraphs

            figure_slugs = pf.get("figure_slugs") or []
            if isinstance(figure_slugs, str):
                figure_slugs = [figure_slugs]
            pf["figure_slugs"] = figure_slugs

            pf["source_pages"] = pages
            pf["source_chunk_indices"] = chunk_indices
            normalized.append(pf)
        return normalized

    def _fallback_frames_for_meta(meta: dict) -> list[dict]:
        block = meta.get("block") or {}
        section = meta.get("section") or "Section"
        subsection = meta.get("subsection")
        source_text = block.get("text") or ""
        retrieved_context = meta.get("retrieved_context") or ""
        section_figures = meta.get("section_figures") or []
        target_frames = meta.get("target_frames")
        if not isinstance(target_frames, int):
            target_frames = max_frames_per_section
        raw_frames = _fallback_plan_frames(
            section,
            subsection,
            source_text,
            retrieved_context,
            section_figures,
            lang,
            target_frames,
        )
        return _normalize_frames_for_meta(raw_frames, meta)

    for chunk_idx in range(total_chunks):
        start = chunk_idx * planner_batch_chunk_size
        end = min(start + planner_batch_chunk_size, len(requests))
        chunk_requests = requests[start:end]
        chunk_metas = metas[start:end]
        print(
            f"[Pipeline] Section planning batch chunk {chunk_idx + 1}/{total_chunks}: "
            f"blocks {start + 1}-{end} (size={len(chunk_requests)})"
        )

        use_batch_chunk = True
        chunk_limit_hit = False
        chunk_responses = []
        try:
            chunk_responses = _run_batch(chunk_requests)
            if chunk_responses is None:
                chunk_responses = []
            elif not isinstance(chunk_responses, list):
                chunk_responses = list(chunk_responses)
        except TimeoutError:
            print(
                f"[Pipeline] Section planning batch chunk {chunk_idx + 1}/{total_chunks} "
                f"timed out after {batch_timeout_s}s; falling back to per-section planning for this chunk."
            )
            use_batch_chunk = False
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if _contains_model_limit_message(msg):
                chunk_limit_hit = True
                print(
                    f"[Pipeline] Section planning batch chunk {chunk_idx + 1}/{total_chunks} "
                    "hit model call limits; falling back to per-section planning for this chunk."
                )
            else:
                print(
                    f"[Pipeline] Section planning batch chunk {chunk_idx + 1}/{total_chunks} "
                    f"failed; falling back to per-section planning for this chunk: {exc}"
                )
            use_batch_chunk = False

        if not use_batch_chunk:
            for local_idx, meta in enumerate(chunk_metas, start=1):
                if chunk_limit_hit:
                    block_frames = _fallback_frames_for_meta(meta)
                else:
                    block = meta.get("block") or {}
                    block_frames = _plan_frames_for_section(
                        model,
                        instructions,
                        block,
                        aligned_figures,
                        keyword_index,
                        formula_candidates,
                    )
                all_frames.extend(block_frames)
                processed_blocks += 1
                if (
                    processed_blocks == 1
                    or processed_blocks % 5 == 0
                    or processed_blocks == total_blocks
                ):
                    print(
                        f"[Pipeline] Section planning progress: {processed_blocks}/{total_blocks} "
                        f"blocks processed (chunk fallback {chunk_idx + 1}/{total_chunks}, item {local_idx}/{len(chunk_metas)})."
                    )
            continue

        if len(chunk_responses) < len(chunk_metas):
            print(
                f"[Pipeline] Section planning batch chunk {chunk_idx + 1}/{total_chunks} returned "
                f"{len(chunk_responses)} responses for {len(chunk_metas)} requests; "
                "missing responses will use per-section fallback."
            )

        for local_idx, meta in enumerate(chunk_metas):
            response = chunk_responses[local_idx] if local_idx < len(chunk_responses) else None
            if response is None:
                block = meta.get("block") or {}
                retry_frames = _plan_frames_for_section(
                    model,
                    instructions,
                    block,
                    aligned_figures,
                    keyword_index,
                    formula_candidates,
                )
                all_frames.extend(retry_frames)
                processed_blocks += 1
                if (
                    processed_blocks == 1
                    or processed_blocks % 5 == 0
                    or processed_blocks == total_blocks
                ):
                    print(
                        f"[Pipeline] Section planning progress: {processed_blocks}/{total_blocks} blocks processed..."
                    )
                continue

            frames = _extract_structured_frames(response)
            limit_hit = False
            if frames is None:
                plan_json_text = _extract_response_text(response)
                if _contains_model_limit_message(plan_json_text):
                    limit_hit = True
                    frames = []
                else:
                    frames = _parse_frames_from_model_output(plan_json_text)

            if not frames:
                if limit_hit:
                    section_name = meta.get("section") or "Section"
                    print(
                        f"[Pipeline] Section planning limit hit in batch response for section '{section_name}'; using fallback."
                    )
                    retry_frames = _fallback_frames_for_meta(meta)
                else:
                    block = meta.get("block") or {}
                    retry_frames = _plan_frames_for_section(
                        model,
                        instructions,
                        block,
                        aligned_figures,
                        keyword_index,
                        formula_candidates,
                    )
                all_frames.extend(retry_frames)
                processed_blocks += 1
                if (
                    processed_blocks == 1
                    or processed_blocks % 5 == 0
                    or processed_blocks == total_blocks
                ):
                    print(
                        f"[Pipeline] Section planning progress: {processed_blocks}/{total_blocks} blocks processed..."
                    )
                continue

            all_frames.extend(_normalize_frames_for_meta(frames, meta))

            processed_blocks += 1
            if (
                processed_blocks == 1
                or processed_blocks % 5 == 0
                or processed_blocks == total_blocks
            ):
                print(
                    f"[Pipeline] Section planning progress: {processed_blocks}/{total_blocks} blocks processed..."
                )

    return all_frames


def _auto_fix_latex_with_llm(model, tex_path: str, compile_log: str, attempt: int, max_log_chars: int = 8000) -> dict:
    tex_file = Path(tex_path).expanduser().resolve()
    try:
        original_content = tex_file.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Failed to read LaTeX source for auto-fix: {exc}")
        return {"changed": False, "error": str(exc)}
    log_snippet = (compile_log or "")[-max_log_chars:]
    payload = {
        "task": "fix_beamer_latex_compile_error",
        "attempt": attempt,
        "latex_source": original_content,
        "compile_log": log_snippet,
    }
    try:
        response = model.invoke(
            {
                "messages": [
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Auto-fix LLM call failed: {exc}")
        return {"changed": False, "error": str(exc)}
    raw = ""
    if isinstance(response, dict):
        messages = response.get("messages") or []
        if messages:
            last = messages[-1]
            raw = (getattr(last, "content", None) or "").strip()
    else:
        raw = (getattr(response, "content", None) or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()
    if not raw:
        print("[Pipeline] Auto-fix LLM returned empty content; skip applying changes.")
        return {"changed": False}
    if raw == original_content:
        print("[Pipeline] Auto-fix LLM output is identical to original LaTeX; skip applying changes.")
        return {"changed": False}
    # Basic structural safety checks to avoid adopting truncated or malformed
    # LaTeX that would drop many frames or break the document.
    if "\\end{document}" not in raw:
        print("[Pipeline] Auto-fix LLM output is missing \\end{document}; skip applying changes.")
        return {"changed": False, "error": "auto-fix output missing \\end{document}"}

    begin_frames = raw.count("\\begin{frame}")
    end_frames = raw.count("\\end{frame}")
    if begin_frames and begin_frames != end_frames:
        print(
            f"[Pipeline] Auto-fix LLM output has unbalanced frame environments (begin={begin_frames}, end={end_frames}); skip applying changes."
        )
        return {"changed": False, "error": "auto-fix output has unbalanced frame environments"}

    if len(raw) < 0.7 * len(original_content):
        print("[Pipeline] Auto-fix LLM output is much shorter than original LaTeX; skip applying changes.")
        return {"changed": False, "error": "auto-fix output suspiciously short"}
    try:
        tex_file.write_text(raw, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Failed to write auto-fixed LaTeX source: {exc}")
        return {"changed": False, "error": str(exc)}
    print("[Pipeline] Applied auto-fixed LaTeX source from LLM.")
    return {"changed": True}


def _interactive_fill_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.pdf:
        while True:
            pdf_path = input("请输入论文 PDF 文件的完整路径: ").strip().strip('"')
            if pdf_path:
                args.pdf = pdf_path
                break
    # Always allow the user to override the figures directory so they can
    # point to a folder that already contains prepared figures for this
    # project. If left empty, fall back to the default ./figures.
    figures = input("请输入 figures 目录路径（可留空，回车跳过）: ").strip().strip('"')
    if figures:
        args.figures = figures
    else:
        args.figures = None
    if not args.template:
        template = input(
            f"请输入 LaTeX 模板路径（回车使用默认: {args.template or 'ppt_template.tex'}）: "
        ).strip().strip('"')
        if template:
            args.template = template
    if not args.output:
        output = input(
            f"请输入输出 LaTeX 文件路径（回车使用默认: {args.output or 'ppt论文汇报_agent.tex'}）: "
        ).strip().strip('"')
        if output:
            args.output = output
    if not args.slides_dir:
        slides_dir = input(
            f"请输入导出 PPT 页面图片目录（回车使用默认: {args.slides_dir or 'export/slides'}）: "
        ).strip().strip('"')
        if slides_dir:
            args.slides_dir = slides_dir
    if not args.export_images:
        answer = input("是否在编译后导出每一页 PPT 图片? [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            args.export_images = True
    if not getattr(args, "lang", None):
        lang = input("请选择 PPT 语言 [zh/en]（回车默认 zh）: ").strip().lower()
        if lang in {"zh", "en"}:
            args.lang = lang
        else:
            args.lang = "zh"

    # Let the user optionally override the maximum number of content frames
    # (excluding title, TOC, and closing slides) so that the final deck does
    # not become excessively long.
    try:
        current_max_frames = int(getattr(args, "max_frames", DEFAULT_MAX_FRAMES) or DEFAULT_MAX_FRAMES)
    except Exception:
        current_max_frames = DEFAULT_MAX_FRAMES
    ans_frames = input(
        f"请输入最大内容页数（不含封面和目录，回车使用默认 {current_max_frames}）: "
    ).strip()
    if ans_frames.isdigit():
        args.max_frames = int(ans_frames)
    else:
        args.max_frames = current_max_frames

    detail_profile = _get_detail_profile(default="detailed")
    _, _, default_per_section = _detail_profile_defaults(detail_profile)
    try:
        current_per_section = int(
            getattr(
                args,
                "max_frames_per_section",
                _resolve_max_frames_per_section(default=default_per_section),
            )
            or default_per_section
        )
    except Exception:
        current_per_section = _resolve_max_frames_per_section(default=default_per_section)
    ans_per_section = input(
        f"请输入每章节建议页数上限（自适应规划，回车使用默认 {current_per_section}）: "
    ).strip()
    if ans_per_section.isdigit():
        args.max_frames_per_section = int(ans_per_section)
    else:
        args.max_frames_per_section = current_per_section

    # Ask whether to enable automatic LaTeX compile error fixing. By default,
    # we turn it on so that the pipeline will keep trying to repair the
    # generated .tex file when xelatex reports errors.
    if not getattr(args, "auto_fix_latex", False):
        ans = input("是否启用自动修复 LaTeX 编译错误? [Y/n]（回车默认 Y）: ").strip().lower()
        if ans in {"n", "no"}:
            args.auto_fix_latex = False
        else:
            args.auto_fix_latex = True

    # Optionally let the user override the maximum number of auto-fix attempts.
    try:
        current_max = int(getattr(args, "max_fix_attempts", 2) or 2)
    except Exception:
        current_max = 2
    prompt = f"自动修复最大尝试次数（回车使用默认 {current_max}）: "
    ans_max = input(prompt).strip()
    if ans_max.isdigit():
        args.max_fix_attempts = int(ans_max)
    else:
        args.max_fix_attempts = current_max
    if not getattr(args, "pptx_template", None):
        candidates = [
            Path("蓝绿色_非遗赛道_template.pptx"),
            Path("template.pptx"),
        ]
        for p in candidates:
            if p.is_file():
                args.pptx_template = str(p.resolve())
                break
    return args


def main() -> None:
    # Load environment variables from env.env / .env so that APIYI_BASE_URL, APIYI_API_KEY,
    # LLM_MODEL and other settings are available to the model and tools.
    load_dotenv(dotenv_path="env.env", override=False)
    load_dotenv(override=False)

    args = parse_args()
    args = _interactive_fill_args(args)

    # CLI overrides for new quality/detail controls (all optional).
    if getattr(args, "report_type", None):
        os.environ["PIPELINE_REPORT_TYPE"] = _normalize_report_type(str(args.report_type))
    if getattr(args, "detail_profile", None):
        os.environ["PIPELINE_DETAIL_PROFILE"] = str(args.detail_profile).strip().lower()
    if getattr(args, "target_content_pages_min", None) is not None:
        os.environ["PIPELINE_TARGET_CONTENT_PAGES_MIN"] = str(int(args.target_content_pages_min))
    if getattr(args, "target_content_pages_max", None) is not None:
        os.environ["PIPELINE_TARGET_CONTENT_PAGES_MAX"] = str(int(args.target_content_pages_max))
    if getattr(args, "mojibake_policy", None):
        os.environ["PIPELINE_MOJIBAKE_POLICY"] = str(args.mojibake_policy).strip().lower()
    if getattr(args, "preferred_text_source", None):
        os.environ["PIPELINE_PREFERRED_TEXT_SOURCE"] = str(args.preferred_text_source)
    if getattr(args, "disable_ocr_fallback", False):
        os.environ["PIPELINE_ENABLE_OCR_FALLBACK"] = "0"
    elif getattr(args, "enable_ocr_fallback", False):
        os.environ["PIPELINE_ENABLE_OCR_FALLBACK"] = "1"
    if getattr(args, "text_quality_debug", False):
        os.environ["PIPELINE_TEXT_QUALITY_DEBUG"] = "1"

    base_model = build_model()

    pdf_path_obj = Path(args.pdf).expanduser().resolve()
    pdf_path = str(pdf_path_obj)

    # Derive a per-project name from the PDF stem so that each paper has its own
    # figures and slide export subdirectories, avoiding cross-project confusion.
    project_name = pdf_path_obj.stem

    # If the user did not provide a title explicitly, use the original PDF stem
    # as a reasonable default so that each project PPT reflects its source paper.
    if not args.title:
        args.title = project_name

    # For filesystem directories that will later appear in LaTeX includegraphics
    # paths, use an ASCII-safe slug to avoid problems with non-ASCII characters
    # (e.g., Chinese) in paths.
    project_dir_name = _slugify_project_name(project_name)

    # Prepare per-run output directories so that each run has an isolated
    # workspace containing figures, slides, cache, and outputs.
    output_root = Path(args.output_dir or "output").expanduser().resolve()
    run_id = args.run_id or _make_run_id()
    run_output_dir = output_root / project_dir_name / run_id
    counter = 1
    while run_output_dir.exists():
        run_output_dir = output_root / project_dir_name / f"{run_id}_{counter}"
        counter += 1

    run_output_dir.mkdir(parents=True, exist_ok=True)
    project_figures_dir = run_output_dir / "figures"
    slides_dir_path = run_output_dir / "slides"
    run_cache_dir = run_output_dir / "cache"
    project_figures_dir.mkdir(parents=True, exist_ok=True)
    slides_dir_path.mkdir(parents=True, exist_ok=True)
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    project_cache_root = output_root / project_dir_name / "cache"
    project_cache_root.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] Run output directory: {run_output_dir}")

    upload_root = None
    if args.figures:
        upload_root = Path(args.figures).expanduser().resolve()

    # Copy user-uploaded images into the project figures directory so that
    # subsequent indexing, RAG and vision analysis always use a predictable
    # relative path under ./figures/<project_slug>/.
    image_exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    if upload_root and upload_root.is_dir():
        num_copied = 0
        for src in upload_root.rglob("*"):
            if not src.is_file() or src.suffix.lower() not in image_exts:
                continue
            try:
                # Skip files that are already inside the project figures
                # directory to avoid recursive copying between runs.
                src.resolve().relative_to(project_figures_dir)
                continue
            except Exception:
                pass

            safe_name = _sanitize_filename(src.name)
            dest = project_figures_dir / safe_name
            counter = 2
            while dest.exists():
                dest = project_figures_dir / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
                counter += 1
            try:
                shutil.copy2(src, dest)
                num_copied += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[Pipeline] Warning: failed to copy image {src} -> {dest}: {exc}")
        if num_copied:
            print(
                f"[Pipeline] Copied {num_copied} user images from {upload_root} into project figures directory {project_figures_dir}"
            )

    figures_dir = str(project_figures_dir)
    slides_dir = str(slides_dir_path)

    template_path = str(Path(args.template).expanduser().resolve())
    output_name = Path(args.output).name if args.output else "ppt论文汇报_agent.tex"
    output_path = str((run_output_dir / output_name).resolve())

    # Copy the source PDF into the run output directory for traceability.
    try:
        source_copy = run_output_dir / pdf_path_obj.name
        if not source_copy.exists():
            shutil.copy2(pdf_path_obj, source_copy)
    except Exception as exc:  # noqa: BLE001
        print(f"[Pipeline] Warning: failed to copy source PDF to output dir: {exc}")

    report_type = _resolve_report_type(default=REPORT_TYPE_GENERAL)
    detail_profile = _resolve_detail_profile(default="detailed")
    target_pages_min, target_pages_max = _resolve_target_content_pages(
        detail_profile=detail_profile
    )
    default_min_pages, default_max_pages, default_per_section = _detail_profile_defaults(detail_profile)
    if target_pages_min < default_min_pages and detail_profile == "detailed":
        target_pages_min = default_min_pages
    if target_pages_max < target_pages_min:
        target_pages_max = max(default_max_pages, target_pages_min)

    selected_max_frames_per_section = _resolve_max_frames_per_section(default=default_per_section)
    try:
        cli_mfps = getattr(args, "max_frames_per_section", None)
        if isinstance(cli_mfps, int):
            selected_max_frames_per_section = _resolve_max_frames_per_section(
                {"max_frames_per_section": cli_mfps},
                default=default_per_section,
            )
    except Exception:
        pass

    mojibake_policy = _resolve_mojibake_policy(default="auto_fallback")
    enable_ocr_fallback = _enable_ocr_fallback(default=True)
    text_quality_debug = _text_quality_debug_enabled(default=True)
    preferred_text_source = _preferred_text_source_path()

    quality_report: dict = {
        "report_type": report_type,
        "detail_profile": detail_profile,
        "target_content_pages_min": target_pages_min,
        "target_content_pages_max": target_pages_max,
        "mojibake_policy": mojibake_policy,
        "enable_ocr_fallback": enable_ocr_fallback,
        "text_quality_debug": text_quality_debug,
        "preferred_text_source": preferred_text_source,
        "fallback_applied": False,
    }
    print(
        f"[Pipeline] report_type={report_type}, detail_profile={detail_profile}, "
        f"target_content_pages={target_pages_min}-{target_pages_max}, "
        f"mojibake_policy={mojibake_policy}, ocr_fallback={enable_ocr_fallback}"
    )

    user_instructions = {
        "task": "generate_ppt",
        "pdf_path": pdf_path,
        "figures_dir": figures_dir,
        "template_path": template_path,
        "output_path": output_path,
        "export_images": bool(args.export_images),
        "slides_dir": slides_dir,
        "title": args.title,
        "subtitle": args.subtitle,
        "author": args.author,
        "institute": args.institute,
        "date": args.date,
        "lang": args.lang or "zh",
        "report_type": report_type,
        "max_frames": args.max_frames or DEFAULT_MAX_FRAMES,
        "detail_profile": detail_profile,
        "target_content_pages_min": target_pages_min,
        "target_content_pages_max": target_pages_max,
        "mojibake_policy": mojibake_policy,
        "enable_ocr_fallback": enable_ocr_fallback,
        "preferred_text_source": preferred_text_source,
        "max_frames_per_section": selected_max_frames_per_section,
        "appendix_images": _resolve_appendix_images(),
        "pptx_template": args.pptx_template,
        "pptx_aspect_ratio": getattr(args, "pptx_aspect_ratio", None),
        "pptx_theme_palette": getattr(args, "pptx_theme_palette", None),
    }
    _write_cache_json(run_cache_dir / "run_config.json", user_instructions)
    cache_enabled = _use_cache()
    cache_key = ""
    figure_cache_file = None
    formula_cache_file = None
    summary_cache_file = None
    if cache_enabled:
        try:
            cache_key = _hash_file(pdf_path_obj)
            figure_cache_file = project_cache_root / f"figures_{cache_key}.json"
            formula_cache_file = project_cache_root / f"formulas_{cache_key}.json"
            summary_cache_file = project_cache_root / f"summary_{cache_key}.json"
        except Exception as exc:  # noqa: BLE001
            print(f"[Pipeline] Warning: failed to compute cache key: {exc}")
            cache_enabled = False

    mode = getattr(args, "mode", "pipeline")
    if mode == "controller":
        print("[Pipeline] 以 controller 模式运行：由 ReAct 控制器智能体通过工具完成 PDF->PPT 流程。\n")
        controller_agent = build_agent(base_model)
        controller_payload = {
            "task": "generate_ppt_with_controller",
            "instructions": user_instructions,
        }
        try:
            controller_agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=json.dumps(controller_payload, ensure_ascii=False)),
                    ]
                },
                config={"configurable": {"thread_id": f"controller-{project_dir_name}"}},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Pipeline] Controller agent execution failed: {exc}")

        tex_exists = Path(output_path).exists()
        pdf_path_summary = str(Path(output_path).with_suffix(".pdf"))
        pdf_exists = Path(pdf_path_summary).exists()
        slides_dir_exists = Path(slides_dir).exists()

        print("\n========== 生成文件路径（根据当前配置推断） ==========")
        print(
            f"LaTeX 源文件: {output_path}"
            + ("  [已找到]" if tex_exists else "  [尚未检测到，可能生成失败或位于其他目录]")
        )
        print(
            f"PPT PDF 文件: {pdf_path_summary}"
            + ("  [已找到]" if pdf_exists else "  [尚未检测到，可能编译失败]")
        )
        print(
            f"幻灯片图片目录: {slides_dir}"
            + ("  [目录存在]" if slides_dir_exists else "  [目录不存在，可能未执行 PptSlideExportTool]")
        )
        return

    pipeline_agents = build_pipeline_agents(base_model)
    model = pipeline_agents["model"]
    section_title_agent = pipeline_agents["section_titles"]
    summary_agent = pipeline_agents.get("summary")
    planner_agent = pipeline_agents["planner"]
    latex_fix_agent = pipeline_agents["latex_fix"]
    formula_agent = pipeline_agents["formula_extractor"]
    figure_metadata_agent = pipeline_agents["figure_metadata"]

    print("开始执行 PPT 自动生成任务，这可能需要一段时间，请耐心等待...\n")

    # Step 1: decide whether to reuse existing figures or extract from PDF
    figures_path_obj = Path(figures_dir)
    image_exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    existing_images = []
    small_skipped = 0
    min_side = 500
    for p in figures_path_obj.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in image_exts:
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            width, height = get_image_dimensions(p)
            if width is not None and height is not None:
                if width < min_side or height < min_side:
                    small_skipped += 1
                    continue
        existing_images.append(p)

    if existing_images:
        print(
            f"[Pipeline] Step 1/6: detected {len(existing_images)} existing images in figures directory; "
            "skip PDF image extraction and use user-provided figures."
        )
        if small_skipped:
            print(f"[Pipeline] Step 1/6: ignored {small_skipped} small images (<{min_side}x{min_side}).")
    else:
        print("[Pipeline] Step 1/6: extract raw figures from PDF into figures directory")
        pdf_figure_extract_tool.invoke({"pdf_path": pdf_path, "output_dir": figures_dir})

    # Step 2: index the figures directory
    print("[Pipeline] Step 2/6: index figures directory")
    figures_index_raw = figure_indexer_tool.invoke({"figures_dir": figures_dir})
    try:
        figures_index = json.loads(figures_index_raw)
    except Exception:
        figures_index = {"figures": []}
    num_figures = len(figures_index.get("figures", []))
    print(f"[Pipeline] Figure index loaded: {num_figures} figures")
    if num_figures:
        selected_figures_dir = run_output_dir / "figures_selected"
        selected_figures_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        name_counter = 1
        for item in figures_index.get("figures", []):
            src = Path(item.get("path", "")).expanduser()
            if not src.exists():
                continue
            dest = selected_figures_dir / src.name
            if dest.exists():
                dest = selected_figures_dir / f"{src.stem}_{name_counter}{src.suffix}"
                name_counter += 1
            try:
                shutil.copy2(src, dest)
                copied += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[Pipeline] Warning: failed to copy selected figure {src}: {exc}")
        print(f"[Pipeline] Copied {copied} selected figures into {selected_figures_dir}")

    # Step 3: (deprecated) align figure captions with extracted images using
    # a regex-based tool. We now rely on an LLM-based caption extractor over
    # pdf_chunks instead, so this step is kept as a no-op for compatibility.
    print("[Pipeline] Step 3/6: skip regex-based FigureCaptionAlignTool (using LLM-based captions instead)")
    aligned_figures: list[dict] = []

    # Step 4: load PDF text into chunks
    print("[Pipeline] Step 4/6: load PDF text into chunks")
    text_load = load_text_chunks_with_fallback(
        pdf_path=pdf_path,
        chunk_size=2000,
        chunk_overlap=200,
        lang=user_instructions.get("lang", "zh"),
        preferred_source_path=preferred_text_source,
        mojibake_policy=mojibake_policy,
        enable_ocr_fallback=enable_ocr_fallback,
    )
    pdf_chunks = text_load.get("chunks") or []
    text_quality = text_load.get("quality") or {}
    text_source_used = text_load.get("source") or "unknown"
    fallback_chain = text_load.get("fallback_chain") or []
    fallback_reason = text_load.get("fallback_reason") or ""
    load_error = text_load.get("error") or ""

    quality_report["text_source_used"] = text_source_used
    quality_report["text_quality_score"] = text_quality.get("score")
    quality_report["text_quality"] = text_quality
    quality_report["fallback_chain"] = fallback_chain
    quality_report["fallback_reason"] = fallback_reason
    quality_report["fallback_applied"] = bool(fallback_reason)

    if text_quality_debug:
        print(
            "[Pipeline][Quality] source={source}, score={score}, mojibake={mojibake}, "
            "chunks={chunks}, fallback_reason={reason}".format(
                source=text_source_used,
                score=text_quality.get("score"),
                mojibake=text_quality.get("is_mojibake"),
                chunks=len(pdf_chunks),
                reason=fallback_reason or "none",
            )
        )
        if fallback_chain:
            print("[Pipeline][Quality] fallback_chain=" + json.dumps(fallback_chain, ensure_ascii=False))

    lang_norm = str(user_instructions.get("lang") or "zh").lower()
    if lang_norm.startswith("zh") and text_quality.get("is_mojibake"):
        if mojibake_policy == "strict_fail":
            fatal_msg = (
                f"Text quality check failed under strict policy. source={text_source_used}, "
                f"reason={fallback_reason or load_error or 'unknown'}"
            )
            quality_report["fatal_error"] = fatal_msg
            _write_quality_report(run_output_dir / "quality_report.json", quality_report)
            raise RuntimeError(fatal_msg)
        if mojibake_policy == "auto_fallback":
            print(
                "[Pipeline] Text quality is still low after fallback chain; entering conservative readable outline mode."
            )
            pdf_chunks = _build_conservative_outline_chunks(lang=lang_norm)
            quality_report["fallback_applied"] = True
            if not quality_report.get("fallback_reason"):
                quality_report["fallback_reason"] = (
                    "all text sources were low quality; switched to conservative outline mode"
                )
        else:
            print(
                "[Pipeline] Warning: low text quality detected under warn_only policy; proceeding with caution."
            )

    if not pdf_chunks:
        if mojibake_policy == "strict_fail":
            fatal_msg = f"No readable text chunks loaded: {load_error or 'unknown error'}"
            quality_report["fatal_error"] = fatal_msg
            _write_quality_report(run_output_dir / "quality_report.json", quality_report)
            raise RuntimeError(fatal_msg)
        print("[Pipeline] No readable text chunks available; using conservative outline mode.")
        pdf_chunks = _build_conservative_outline_chunks(lang=lang_norm)
        quality_report["fallback_applied"] = True
        quality_report["fallback_reason"] = quality_report.get("fallback_reason") or "no readable chunks loaded"

    print(f"[Pipeline] PDF text loaded: {len(pdf_chunks)} chunks")

    image_index_items = figures_index.get("figures") or []

    # Use a specialized agent to extract figure/table captions directly from
    # the PDF text chunks. When no extracted images are available
    # (aligned_figures is empty), fall back to these metadata-only figures so
    # that planning can still see where figures/tables are and what they
    # describe.
    llm_figures = None
    formula_candidates = None
    if cache_enabled and figure_cache_file:
        cached_figures = _load_cache_json(figure_cache_file)
        if isinstance(cached_figures, list):
            if _is_cache_text_payload_mojibake(cached_figures, lang=lang_norm):
                print(
                    "[Pipeline] Cached figure metadata looks mojibake; ignoring cache and re-extracting."
                )
            else:
                llm_figures = cached_figures
                print("[Pipeline] Loaded LLM figure metadata from cache.")
    if cache_enabled and formula_cache_file:
        cached_formulas = _load_cache_json(formula_cache_file)
        if isinstance(cached_formulas, list):
            if _is_cache_text_payload_mojibake(cached_formulas, lang=lang_norm):
                print(
                    "[Pipeline] Cached formula candidates look mojibake; ignoring cache and re-extracting."
                )
            else:
                formula_candidates = cached_formulas
                print("[Pipeline] Loaded LLM formula candidates from cache.")

    parallel_extract = (
        os.getenv("PIPELINE_PARALLEL_EXTRACT", "1").strip().lower()
        not in {"0", "false", "no"}
    )
    if llm_figures is None or formula_candidates is None:
        if parallel_extract and (llm_figures is None and formula_candidates is None):
            with ThreadPoolExecutor(max_workers=2) as executor:
                fig_future = executor.submit(
                    _collect_figure_metadata_with_llm, figure_metadata_agent, pdf_chunks
                )
                formula_future = executor.submit(
                    _collect_formula_candidates_with_llm, formula_agent, pdf_chunks
                )
                llm_figures = fig_future.result()
                formula_candidates = formula_future.result()
        else:
            if llm_figures is None:
                llm_figures = _collect_figure_metadata_with_llm(
                    figure_metadata_agent, pdf_chunks
                )
            if formula_candidates is None:
                formula_candidates = _collect_formula_candidates_with_llm(
                    formula_agent, pdf_chunks
                )

    if cache_enabled and cache_key:
        if isinstance(llm_figures, list):
            _write_cache_json(run_cache_dir / f"figures_{cache_key}.json", llm_figures)
            if figure_cache_file:
                _write_cache_json(figure_cache_file, llm_figures)
        if isinstance(formula_candidates, list):
            _write_cache_json(run_cache_dir / f"formulas_{cache_key}.json", formula_candidates)
            if formula_cache_file:
                _write_cache_json(formula_cache_file, formula_candidates)

    if llm_figures and image_index_items:
        aligned_figures = _align_llm_figures_with_image_index(llm_figures, image_index_items)
        num_with_paths = sum(1 for f in aligned_figures if f.get("path"))
        print(
            f"[Pipeline] Using LLM-extracted figure/table captions as aligned_figures: "
            f"{len(aligned_figures)} items (mapped to {num_with_paths} images from {len(image_index_items)} files)"
        )

        # Some user images may still be unmatched after the initial alignment
        # (for example, hand-prepared files like "Fig. 7.jpg" that could not be
        # linked purely by filename heuristics). For those, try to reuse
        # LLM-extracted captions/context by matching on figure/table number and
        # normalized caption keys so that every image has meaningful metadata
        # (caption, context, page) before we attach them to content frames.
        aligned_figures = _backfill_unmatched_user_images_with_llm_metadata(
            image_index_items,
            aligned_figures,
        )

        # Optionally enrich figure metadata with vision-based descriptions so
        # that downstream figure/text matching can leverage both textual
        # captions/context and the actual visual content of each image.
        _enrich_figures_with_vision_descriptions(aligned_figures)

    # Build a lightweight keyword index over PDF chunks to support a simple
    # RAG-style retrieval layer used during per-section planning.
    keyword_index = None
    if pdf_chunks:
        try:
            keyword_index = build_keyword_index(pdf_chunks)
            print(f"[Pipeline] Built keyword index over {len(pdf_chunks)} chunks")
        except Exception as exc:
            print(f"[Pipeline] Failed to build keyword index for RAG: {exc}")
            keyword_index = None

    # Step 5: plan frames for all section blocks so that each frame's content is
    # tightly grounded in a specific portion of the source text. First, use the
    # LLM to normalize noisy section titles (e.g., numeric-only headings or
    # author names) into cleaner, paper-aligned section titles, then plan
    # frames with batched LLM calls.
    print("[Pipeline] Step 5/6: plan frames per section block (adaptive 2-10 frames per section)")
    section_blocks = _build_section_blocks(pdf_chunks)
    quality_report["sections_detected"] = len(section_blocks)
    skip_norm = os.getenv("PIPELINE_SKIP_SECTION_NORMALIZATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if report_type == REPORT_TYPE_ZHENGDA and not skip_norm:
        print("[Pipeline] report_type=zhengda_cup: skipping LLM section title normalization.")
        skip_norm = True
    if skip_norm:
        print("[Pipeline] Skipping section title normalization (PIPELINE_SKIP_SECTION_NORMALIZATION=1).")
    else:
        section_blocks = _normalize_section_titles_with_llm(
            section_title_agent, user_instructions, section_blocks
        )
    before_filter = len(section_blocks)
    section_blocks = _filter_out_reference_blocks(section_blocks)
    removed = before_filter - len(section_blocks)
    quality_report["sections_detected_after_filter"] = len(section_blocks)
    if removed:
        print(f"[Pipeline] Dropped {removed} reference blocks from section planning.")
    section_blocks = _refine_section_blocks_for_outline(
        section_blocks,
        lang=lang_norm,
    )
    if report_type == REPORT_TYPE_ZHENGDA:
        section_blocks = _apply_zhengda_template_to_section_blocks(section_blocks, lang=lang_norm)
    quality_report["sections_after_outline_refine"] = len(section_blocks)
    # Limit the number of distinct top-level sections in the PPT outline so
    # that the Beamer table of contents does not become overly long. We cap
    # it to 9 coarse-grained sections (e.g., 背景、文献综述、方法、数据、实验、
    # 总结与展望), and merge any later sections into the last kept one while
    # preserving document order.
    if report_type != REPORT_TYPE_ZHENGDA:
        section_blocks = _limit_top_level_sections(section_blocks, max_sections=9)
        section_blocks, appended_conclusion_block = _ensure_conclusion_section_block(
            section_blocks,
            pdf_chunks,
            lang=lang_norm,
        )
    else:
        appended_conclusion_block = False
    quality_report["conclusion_section_block_injected"] = bool(appended_conclusion_block)
    outline_sections = _build_outline_from_section_blocks(
        section_blocks,
        lang=lang_norm,
        report_type=report_type,
    )
    quality_report["outline_sections"] = outline_sections
    if outline_sections:
        print("[Pipeline] Final outline sections: " + " -> ".join(outline_sections))
    effective_max_frames = _resolve_effective_max_frames(
        user_instructions.get("max_frames"),
        section_blocks,
        selected_max_frames_per_section,
    )
    if effective_max_frames != user_instructions.get("max_frames"):
        print(
            f"[Pipeline] Adaptive max_frames applied: {user_instructions.get('max_frames')} -> "
            f"{effective_max_frames} (sections={len(section_blocks)}, max_frames_per_section={selected_max_frames_per_section})"
        )
        user_instructions["max_frames"] = effective_max_frames
    cached_summary = None
    if cache_enabled and summary_cache_file:
        cached_summary = _load_cache_json(summary_cache_file)
        if cached_summary and _is_cache_text_payload_mojibake(cached_summary, lang=lang_norm):
            print("[Pipeline] Cached summary looks mojibake; ignoring cache and regenerating.")
            cached_summary = None
        elif cached_summary and _is_cache_text_payload_toc_like(cached_summary):
            print("[Pipeline] Cached summary looks like table-of-contents text; ignoring cache and regenerating.")
            cached_summary = None
        elif cached_summary:
            print("[Pipeline] Loaded summary from cache.")
    summary_frame = _build_summary_frame(
        summary_agent,
        user_instructions,
        pdf_chunks,
        section_blocks,
        cached_summary=cached_summary if isinstance(cached_summary, dict) else None,
    )
    if summary_frame and cache_enabled and summary_cache_file and not cached_summary:
        summary_obj = {
            "title": summary_frame.get("frame_title") or "",
            "bullets": summary_frame.get("bullets") or [],
            "paragraphs": summary_frame.get("paragraphs") or [],
        }
        _write_cache_json(run_cache_dir / f"summary_{cache_key}.json", summary_obj)
        _write_cache_json(summary_cache_file, summary_obj)
    global_planning_context = _build_global_planning_context(
        section_blocks=section_blocks,
        outline_sections=outline_sections,
        summary_frame=summary_frame if isinstance(summary_frame, dict) else None,
        lang=lang_norm,
        max_chars=2600,
    )
    if global_planning_context:
        user_instructions["global_planning_context"] = global_planning_context
        user_instructions["outline_sections"] = outline_sections
        preview = global_planning_context[:220].replace("\n", " | ")
        print(f"[Pipeline] Global planning context ready ({len(global_planning_context)} chars): {preview}")
    # Plan text-centric frames per section block. We now pass aligned_figures so
    # that the planner can directly integrate figures into the content frames
    # via figure_slugs, instead of creating separate figure-only slides later.
    all_frames = _plan_frames_for_sections_batch(
        planner_agent,
        user_instructions,
        section_blocks,
        aligned_figures,
        keyword_index,
        formula_candidates,
    )

    total_frames = len(all_frames)
    print(f"[Pipeline] Aggregated frames from all sections: {total_frames} frames")
    if summary_frame:
        all_frames.insert(0, summary_frame)
    # First remove any frames that contain only figures and no text so that
    # their images can be reattached to nearby text frames.
    all_frames = _remove_figure_only_frames(all_frames)
    all_frames, outline_frames_injected = _ensure_outline_coverage_frames(
        all_frames,
        section_blocks,
        outline_sections,
        lang=lang_norm,
    )
    quality_report["outline_frames_injected"] = int(outline_frames_injected)
    quality_report["frames_before_cap"] = len(all_frames)

    # Enforce a per-section soft cap linked to global target pages.
    max_frames_per_section = _resolve_max_frames_per_section(user_instructions)
    all_frames = _cap_frames_per_section(
        all_frames,
        aligned_figures,
        max_frames_per_section,
        target_min_pages=target_pages_min,
        detail_profile=detail_profile,
    )
    quality_report["frames_after_cap"] = len(all_frames)

    # Attach any aligned figures that are still unused into existing frames
    # based on caption/context similarity and shared/nearby source pages, so
    # that images are integrated into content slides instead of occupying
    # standalone image-only slides.
    _distribute_unused_figures_into_frames(all_frames, aligned_figures, keyword_index)

    # Avoid overcrowding a single frame with too many figures; overflow images
    # will be handled later as appendix galleries.
    all_frames = _trim_figures_per_frame(all_frames, _get_max_figures_per_frame())

    # Finally, enforce that each concrete user image file from the figures
    # directory is rendered at most once across all frames.
    all_frames = _deduplicate_figure_usages_by_image_path(all_frames, aligned_figures)

    # Detailed profile gate: if still below minimum content pages and not
    # manually constrained by a low max_frames, auto-complete once.
    max_frames_hint = user_instructions.get("max_frames")
    manual_low_cap = isinstance(max_frames_hint, int) and max_frames_hint < target_pages_min
    if (
        detail_profile == "detailed"
        and len(all_frames) < target_pages_min
        and not manual_low_cap
    ):
        print(
            f"[Pipeline] Content pages below detailed minimum ({len(all_frames)} < {target_pages_min}); "
            "triggering one auto-completion pass."
        )
        all_frames = _expand_frames_to_target_min(
            all_frames,
            aligned_figures,
            target_min_pages=target_pages_min,
            target_max_pages=target_pages_max,
            lang=user_instructions.get("lang", "zh"),
        )
        _distribute_unused_figures_into_frames(all_frames, aligned_figures, keyword_index)
        all_frames = _trim_figures_per_frame(all_frames, _get_max_figures_per_frame())
        all_frames = _deduplicate_figure_usages_by_image_path(all_frames, aligned_figures)
        quality_report["replan_triggered"] = True
    else:
        quality_report["replan_triggered"] = False

    # Ensure that any remaining user images from the figures directory that
    # are still unused are given at least one dedicated frame under an
    # appendix-like section.
    if _resolve_appendix_images(user_instructions):
        extra_frames, aligned_figures = _build_extra_image_frames(
            all_frames,
            aligned_figures,
            image_index_items,
        )
        if extra_frames:
            all_frames.extend(extra_frames)
    else:
        print("[Pipeline] Appendix image frames disabled; skipping extra image-only slides.")

    if report_type != REPORT_TYPE_ZHENGDA:
        all_frames, injected_conclusion_frame = _ensure_conclusion_frames(
            all_frames,
            lang=lang_norm,
        )
    else:
        injected_conclusion_frame = False
    quality_report["conclusion_frame_injected"] = bool(injected_conclusion_frame)
    all_frames, outline_section_rewrites = _map_frames_to_outline_sections(
        all_frames,
        section_blocks,
        outline_sections,
        lang=lang_norm,
    )
    quality_report["outline_section_rewrites"] = int(outline_section_rewrites)
    all_frames = _reorder_conclusion_recommendation_frames(all_frames)
    all_frames, outline_title_rewrites = _align_frame_titles_to_outline(
        all_frames,
        outline_sections,
        lang=lang_norm,
    )
    quality_report["outline_title_rewrites"] = int(outline_title_rewrites)

    # Text gate before writing LaTeX/PPTX: rewrite suspicious mojibake strings
    # and perform lightweight compile-oriented sanitation.
    all_frames, mojibake_hits = _sanitize_frames_before_render(
        all_frames,
        lang=user_instructions.get("lang", "zh"),
    )
    quality_report["mojibake_hits"] = int(mojibake_hits)

    # After all figure post-processing, enforce the global max_frames budget.
    max_frames = user_instructions.get("max_frames")
    if detail_profile in {"standard", "detailed"}:
        if not isinstance(max_frames, int) or max_frames <= 0 or max_frames > target_pages_max:
            max_frames = target_pages_max
            user_instructions["max_frames"] = max_frames
    if isinstance(max_frames, int) and max_frames > 0:
        capped_frames = _cap_frames_to_max(all_frames, aligned_figures, max_frames)
        if len(capped_frames) < len(all_frames):
            print(
                f"[Pipeline] Capped content frames to max_frames={max_frames}: "
                f"{len(all_frames)} -> {len(capped_frames)}"
            )
        all_frames = capped_frames

    used_slugs: set[str] = set()
    for frame in all_frames:
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        for slug in slugs:
            if slug:
                used_slugs.add(str(slug))
    quality_report["figures_mapped"] = len(used_slugs)
    quality_report["frames_final"] = len(all_frames)

    plan_data = {"frames": all_frames, "figures": aligned_figures}
    plan_json_for_tool = json.dumps(plan_data, ensure_ascii=False)
    _write_cache_json(run_cache_dir / "plan.json", plan_data)
    print("[Pipeline] Model frames parsed and combined with aligned figures, calling LatexWriterTool...")

    lang_for_render = str(args.lang or user_instructions.get("lang") or "zh")
    safe_title, _ = _sanitize_line_for_render(
        str(args.title or ""),
        lang=lang_for_render,
        fallback="论文汇报" if lang_for_render.lower().startswith("zh") else "Presentation",
    )
    safe_subtitle, _ = _sanitize_line_for_render(str(args.subtitle or ""), lang=lang_for_render, fallback="")
    safe_author, _ = _sanitize_line_for_render(str(args.author or ""), lang=lang_for_render, fallback="")
    safe_institute, _ = _sanitize_line_for_render(str(args.institute or ""), lang=lang_for_render, fallback="")
    safe_date, _ = _sanitize_line_for_render(str(args.date or ""), lang=lang_for_render, fallback="")

    latex_result_raw = latex_writer_tool.invoke(
        {
            "template_path": template_path,
            "output_path": output_path,
            "plan_json": plan_json_for_tool,
            "title": safe_title,
            "subtitle": safe_subtitle,
            "author": safe_author,
            "institute": safe_institute,
            "date": safe_date,
            "lang": args.lang,
        }
    )
    try:
        latex_result = json.loads(latex_result_raw)
    except Exception:
        latex_result = {}

    if "error" in latex_result:
        print(f"[Pipeline] LatexWriterTool reported error: {latex_result['error']}")
    else:
        print(f"[Pipeline] LaTeX file generated at: {latex_result.get('output_path', output_path)}")
        preflight = _preflight_latex_source(latex_result.get("output_path", output_path))
        quality_report["latex_preflight"] = preflight
        if preflight.get("error"):
            print(f"[Pipeline] LaTeX preflight warning: {preflight.get('error')}")
        elif preflight.get("changed"):
            print(
                f"[Pipeline] LaTeX preflight applied {preflight.get('changes', 0)} auto-fix adjustments before compile."
            )

    pptx_output_path = str(Path(output_path).with_suffix(".pptx"))
    pptx_engine = os.getenv("PPTX_ENGINE", "com").strip().lower()
    if pptx_engine in {"com", "powerpoint", "pptx-com", "pptx_com"}:
        print("[Pipeline] Generating native PPTX file via PowerPoint COM engine...")
        pptx_result_raw = pptx_com_writer_tool.invoke(
            {
                "output_path": pptx_output_path,
                "plan_json": plan_json_for_tool,
                "title": safe_title,
                "subtitle": safe_subtitle,
                "author": safe_author,
                "institute": safe_institute,
                "date": safe_date,
                "lang": args.lang,
                "pptx_template": args.pptx_template,
                "aspect_ratio": getattr(args, "pptx_aspect_ratio", None),
                "theme_palette": getattr(args, "pptx_theme_palette", None),
            }
        )
        try:
            pptx_result = json.loads(pptx_result_raw)
        except Exception:
            pptx_result = {}
        if "error" in pptx_result:
            print(
                f"[Pipeline] COM PPTX engine failed, falling back to python-pptx: {pptx_result['error']}"
            )
            pptx_result_raw = pptx_writer_tool.invoke(
                {
                    "output_path": pptx_output_path,
                    "plan_json": plan_json_for_tool,
                    "title": safe_title,
                    "subtitle": safe_subtitle,
                    "author": safe_author,
                    "institute": safe_institute,
                    "date": safe_date,
                    "lang": args.lang,
                    "pptx_template": args.pptx_template,
                    "aspect_ratio": getattr(args, "pptx_aspect_ratio", None),
                    "theme_palette": getattr(args, "pptx_theme_palette", None),
                }
            )
    else:
        print("[Pipeline] Generating native PPTX file via python-pptx engine...")
        pptx_result_raw = pptx_writer_tool.invoke(
            {
                "output_path": pptx_output_path,
                "plan_json": plan_json_for_tool,
                "title": safe_title,
                "subtitle": safe_subtitle,
                "author": safe_author,
                "institute": safe_institute,
                "date": safe_date,
                "lang": args.lang,
                "pptx_template": args.pptx_template,
                "aspect_ratio": getattr(args, "pptx_aspect_ratio", None),
                "theme_palette": getattr(args, "pptx_theme_palette", None),
            }
        )
    try:
        pptx_result = json.loads(pptx_result_raw)
    except Exception:
        pptx_result = {}
    if "error" in pptx_result:
        print(f"[Pipeline] PptxWriterTool reported error: {pptx_result['error']}")
    else:
        print(f"[Pipeline] PPTX file generated at: {pptx_result.get('output_path', pptx_output_path)}")

    # Step 6: compile LaTeX to PDF
    print("[Pipeline] Step 6/6: compile LaTeX to PDF")
    pdf_path = str(Path(output_path).with_suffix(".pdf"))
    use_auto_fix = bool(getattr(args, "auto_fix_latex", False))
    max_fix_attempts = int(getattr(args, "max_fix_attempts", 2) or 0)
    attempt = 0
    compile_result: dict = {}
    while True:
        compile_result_raw = compile_check_tool.invoke(
            {"tex_path": output_path, "workdir": str(Path(output_path).parent)}
        )
        try:
            compile_result = json.loads(compile_result_raw)
        except Exception:
            compile_result = {}
        if compile_result.get("success"):
            pdf_path = compile_result.get("pdf_path", pdf_path)
            print(f"[Pipeline] PDF compiled at: {pdf_path}")
            break
        err_msg = compile_result.get("error") or ""
        if not err_msg:
            rc = compile_result.get("returncode")
            log_path = compile_result.get("log_path") or ""
            err_msg = f"xelatex failed (returncode={rc}, log_path={log_path})"
        print(f"[Pipeline] Compile failed or returned error: {err_msg}")
        if not use_auto_fix:
            # Auto-fix is disabled; do not rewrite the LaTeX file.
            break
        log_for_fix = compile_result.get("log") or ""
        if attempt >= max_fix_attempts:
            print("[Pipeline] Reached maximum number of auto-fix attempts; stop trying to fix LaTeX errors.")
            break
        attempt += 1
        print(f"[Pipeline] Attempting automatic LaTeX fix with LLM (attempt {attempt}/{max_fix_attempts})")
        fix_result = _auto_fix_latex_with_llm(latex_fix_agent, output_path, log_for_fix, attempt)
        if not fix_result.get("changed"):
            print("[Pipeline] Auto-fix did not modify LaTeX source; stop further fix attempts.")
            break

    # Optionally export each slide as an image
    if args.export_images:
        print("[Pipeline] Exporting PDF pages to slide images...")
        export_result_raw = ppt_slide_export_tool.invoke(
            {
                "pdf_path": pdf_path,
                "output_dir": slides_dir,
                "format": "png",
                "dpi": 200,
            }
        )
        try:
            export_result = json.loads(export_result_raw)
        except Exception:
            export_result = {}
        if "error" in export_result:
            print(f"[Pipeline] Slide export error: {export_result['error']}")
        else:
            print(f"[Pipeline] Slide images exported: {len(export_result.get('images', []))} files")

    quality_report["compile_success"] = bool(compile_result.get("success"))
    quality_report["compile_error"] = compile_result.get("error") or ""
    quality_report["compile_returncode"] = compile_result.get("returncode")
    quality_report["compile_log_path"] = compile_result.get("log_path")
    quality_report["output_tex_path"] = output_path
    quality_report["output_pdf_path"] = pdf_path
    quality_report["output_pptx_path"] = pptx_result.get("output_path", pptx_output_path)
    quality_report["sections_detected"] = quality_report.get("sections_detected", 0)
    quality_report["frames_before_cap"] = quality_report.get("frames_before_cap", 0)
    quality_report["frames_after_cap"] = quality_report.get("frames_after_cap", 0)
    quality_report["frames_after_postprocess"] = len(all_frames)
    quality_report["fallback_applied"] = bool(
        quality_report.get("fallback_applied")
        or quality_report.get("fallback_reason")
        or (quality_report.get("text_source_used") not in {"pdf", "preferred"})
    )

    print(
        "[Pipeline][QualitySummary] "
        + json.dumps(
            {
                "sections_detected": quality_report.get("sections_detected"),
                "sections_after_outline_refine": quality_report.get("sections_after_outline_refine"),
                "frames_before_cap": quality_report.get("frames_before_cap"),
                "frames_after_cap": quality_report.get("frames_after_cap"),
                "figures_mapped": quality_report.get("figures_mapped"),
                "mojibake_hits": quality_report.get("mojibake_hits"),
                "fallback_applied": quality_report.get("fallback_applied"),
                "conclusion_section_block_injected": quality_report.get("conclusion_section_block_injected"),
                "conclusion_frame_injected": quality_report.get("conclusion_frame_injected"),
                "outline_section_rewrites": quality_report.get("outline_section_rewrites"),
                "outline_frames_injected": quality_report.get("outline_frames_injected"),
                "outline_title_rewrites": quality_report.get("outline_title_rewrites"),
            },
            ensure_ascii=False,
        )
    )
    _write_quality_report(run_output_dir / "quality_report.json", quality_report)

    # Finally, print a summary of the key output paths so the user can easily find the results
    tex_exists = Path(output_path).exists()
    pdf_exists = Path(pdf_path).exists()
    slides_dir_exists = Path(slides_dir).exists()

    print("\n========== 生成文件路径（根据当前配置推断） ==========")
    print(f"LaTeX 源文件: {output_path}" + ("  [已找到]" if tex_exists else "  [尚未检测到，可能生成失败或位于其他目录]"))
    print(f"PPT PDF 文件: {pdf_path}" + ("  [已找到]" if pdf_exists else "  [尚未检测到，可能编译失败]"))
    print(f"幻灯片图片目录: {slides_dir}" + ("  [目录存在]" if slides_dir_exists else "  [目录不存在，可能未执行 PptSlideExportTool]"))

