from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import hashlib
import uuid
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import ast
import base64
import re

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pdf2image import convert_from_path
from openai import OpenAI

from ppt_agent.logging_utils import get_logger, make_log_print

try:  # Optional dependency for writing .pptx files directly from the plan JSON.
    from pptx import Presentation  # type: ignore[import]
    from pptx.util import Inches, Pt  # type: ignore[import]
    from pptx.dml.color import RGBColor  # type: ignore[import]
    from pptx.enum.shapes import MSO_SHAPE  # type: ignore[import]
    from pptx.enum.text import MSO_AUTO_SIZE, MSO_ANCHOR  # type: ignore[import]
except Exception:  # noqa: BLE001
    Presentation = None  # type: ignore[assignment]
    Inches = None  # type: ignore[assignment]
    Pt = None  # type: ignore[assignment]
    RGBColor = None  # type: ignore[assignment]
    MSO_SHAPE = None  # type: ignore[assignment]
    MSO_AUTO_SIZE = None  # type: ignore[assignment]
    MSO_ANCHOR = None  # type: ignore[assignment]

from ppt_agent.rag import build_keyword_index, keyword_search

try:  # Pillow is used to inspect image dimensions for safer scaling in Beamer slides.
    from PIL import Image  # type: ignore[import]
except Exception:  # noqa: BLE001
    Image = None  # type: ignore[assignment]


logger = get_logger("paper2slides.tools")
print = make_log_print(logger)


def _read_png_dimensions(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    if len(data) < 24:
        return None, None
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, None
    if data[12:16] != b"IHDR":
        return None, None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def _read_jpeg_dimensions(file_obj: io.BufferedReader) -> Tuple[Optional[int], Optional[int]]:
    file_obj.seek(0)
    if file_obj.read(2) != b"\xff\xd8":
        return None, None

    def _next_marker() -> Optional[int]:
        byte = file_obj.read(1)
        while byte and byte != b"\xff":
            byte = file_obj.read(1)
        if not byte:
            return None
        while True:
            byte = file_obj.read(1)
            if not byte:
                return None
            if byte != b"\xff":
                return byte[0]

    while True:
        marker = _next_marker()
        if marker is None:
            return None, None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            size_bytes = file_obj.read(2)
            if len(size_bytes) < 2:
                return None, None
            _size = int.from_bytes(size_bytes, "big")
            _precision = file_obj.read(1)
            height_bytes = file_obj.read(2)
            width_bytes = file_obj.read(2)
            if len(height_bytes) < 2 or len(width_bytes) < 2:
                return None, None
            height = int.from_bytes(height_bytes, "big")
            width = int.from_bytes(width_bytes, "big")
            return width, height
        if marker in {0xDA, 0xD9}:
            return None, None
        size_bytes = file_obj.read(2)
        if len(size_bytes) < 2:
            return None, None
        size = int.from_bytes(size_bytes, "big")
        if size < 2:
            return None, None
        file_obj.seek(size - 2, 1)


def get_image_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if Image is not None:
        try:
            with Image.open(path) as img:
                width, height = img.size
            return width, height
        except Exception:  # noqa: BLE001
            pass

    suffix = path.suffix.lower()
    try:
        with path.open("rb") as f:
            if suffix == ".png":
                header = f.read(24)
                return _read_png_dimensions(header)
            if suffix in {".jpg", ".jpeg"}:
                return _read_jpeg_dimensions(f)
    except Exception:  # noqa: BLE001
        return None, None
    return None, None


KNOWN_MOJIBAKE_TOKENS = (
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
    "闄",
    "鏈",
    "璇",
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
    "nnnn",
    "IIII",
    "\ufffd",
)


def evaluate_text_quality(text: str, lang: str = "zh") -> dict:
    """Evaluate whether extracted text is likely readable or mojibake."""
    s = text or ""
    total = max(1, len(s))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", s))
    replacement_char_count = s.count("\ufffd")
    known_mojibake_token_count = 0
    for token in KNOWN_MOJIBAKE_TOKENS:
        known_mojibake_token_count += s.count(token)
    box_drawing_count = len(re.findall(r"[\u2500-\u257F]", s))
    gbk_utf8_mojibake_hint_count = 0
    for token in ("锛", "銆", "鏈€", "鐨", "鍥", "鐩", "绗", "€"):
        gbk_utf8_mojibake_hint_count += s.count(token)
    mojibake_like_char_count = len(
        re.findall(r"[缁鏂闂鎬鎻妯鍔绛鎸纭鐮璇鍥鐩鍒鎴璁]", s)
    )

    placeholder_chars = 0
    for m in re.finditer(r"(?:\bn{6,}\b|\bI{6,}\b|[nI]{10,})", s):
        placeholder_chars += len(m.group(0))
    placeholder_run_ratio = placeholder_chars / total

    alpha_like = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", s))
    cjk_ratio = cjk_count / max(1, alpha_like)
    lang_norm = (lang or "").strip().lower()

    score = 1.0
    if lang_norm.startswith("zh"):
        if cjk_ratio < 0.08:
            score -= 0.45
        if cjk_ratio < 0.02:
            score -= 0.25
    score -= min(0.45, placeholder_run_ratio * 2.2)
    score -= min(0.25, known_mojibake_token_count / 40.0)
    score -= min(0.2, replacement_char_count / 8.0)
    score -= min(0.65, gbk_utf8_mojibake_hint_count / 12.0)
    score -= min(0.6, box_drawing_count / 1.0)
    score -= min(0.55, mojibake_like_char_count / 12.0)
    score = max(0.0, min(1.0, score))

    is_mojibake = score < (0.45 if lang_norm.startswith("zh") else 0.3)
    return {
        "score": round(score, 4),
        "is_mojibake": bool(is_mojibake),
        "cjk_ratio": round(cjk_ratio, 4),
        "placeholder_run_ratio": round(placeholder_run_ratio, 4),
        "replacement_char_count": int(replacement_char_count),
        "known_mojibake_token_count": int(known_mojibake_token_count),
        "gbk_utf8_mojibake_hint_count": int(gbk_utf8_mojibake_hint_count),
        "box_drawing_count": int(box_drawing_count),
        "mojibake_like_char_count": int(mojibake_like_char_count),
        "text_len": int(len(s)),
    }


def evaluate_chunks_quality(chunks: list[dict], lang: str = "zh") -> dict:
    text = "\n".join((c or {}).get("page_content") or "" for c in (chunks or []))
    result = evaluate_text_quality(text, lang=lang)
    result["num_chunks"] = len(chunks or [])
    return result


def _extract_docx_media_images(
    docx_path: Path,
    out_dir: Path,
    min_side: int,
    seen_hashes: set[str],
) -> tuple[list[dict], int, int]:
    """Extract embedded images from a DOCX package's word/media directory."""
    images: list[dict] = []
    num_small_skipped = 0
    num_duplicates = 0
    if not docx_path.exists():
        return images, num_small_skipped, num_duplicates

    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            media_names = [
                n
                for n in zf.namelist()
                if n.startswith("word/media/")
                and n.lower().split(".")[-1] in {"png", "jpg", "jpeg", "bmp", "webp"}
            ]
            media_names.sort()
            for idx, name in enumerate(media_names, start=1):
                try:
                    data = zf.read(name)
                    if not data:
                        continue
                    width = None
                    height = None
                    if Image is not None:
                        try:
                            with Image.open(io.BytesIO(data)) as img:
                                width, height = img.size
                        except Exception:  # noqa: BLE001
                            width, height = None, None
                    if width is not None and height is not None and (width < min_side or height < min_side):
                        num_small_skipped += 1
                        continue
                    img_hash = _hash_image_pixels(data)
                    if img_hash and img_hash in seen_hashes:
                        num_duplicates += 1
                        continue
                    if img_hash:
                        seen_hashes.add(img_hash)
                    ext = Path(name).suffix.lower() or ".png"
                    filename = f"page_{1:03d}_docx_img_{idx:03d}{ext}"
                    out_path = out_dir / filename
                    with out_path.open("wb") as f:
                        f.write(data)
                    images.append({"path": str(out_path), "page": 1})
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        return images, num_small_skipped, num_duplicates
    return images, num_small_skipped, num_duplicates


def _ocr_page_text(image_obj) -> str:
    """Best-effort OCR with pytesseract; fallback to tesseract CLI."""
    try:
        import pytesseract  # type: ignore

        text = pytesseract.image_to_string(image_obj, lang="chi_sim+eng")
        if text and text.strip():
            return text
    except Exception:  # noqa: BLE001
        pass

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            image_obj.save(str(tmp_path))
        proc = subprocess.run(
            ["tesseract", str(tmp_path), "stdout", "-l", "chi_sim+eng"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            encoding="utf-8",
            errors="ignore",
        )
        return proc.stdout or ""
    except Exception:  # noqa: BLE001
        return ""
    finally:
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
        except Exception:  # noqa: BLE001
            pass


def _ocr_pdf_to_chunks(path: Path, chunk_size: int = 2000, chunk_overlap: int = 200) -> list[dict]:
    pages = convert_from_path(str(path), dpi=180)
    docs: list[Document] = []
    for i, page in enumerate(pages, start=1):
        text = _ocr_page_text(page) or ""
        docs.append(Document(page_content=text, metadata={"page": i, "source": "ocr"}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return [{"page_content": d.page_content, "metadata": dict(d.metadata or {})} for d in chunks]


def load_text_chunks_with_fallback(
    pdf_path: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    lang: str = "zh",
    preferred_source_path: Optional[str] = None,
    mojibake_policy: str = "auto_fallback",
    enable_ocr_fallback: bool = True,
) -> dict:
    """Load chunks with fallback chain: PDF -> companion DOCX -> preferred -> OCR."""
    pdf_file = Path(pdf_path).expanduser().resolve()
    chain: list[dict] = []
    candidates: list[tuple[str, Path]] = []

    preferred = Path(preferred_source_path).expanduser().resolve() if preferred_source_path else None
    candidates.append(("pdf", pdf_file))

    companion_docx = pdf_file.with_suffix(".docx")
    if companion_docx.exists():
        candidates.append(("docx", companion_docx))
    if preferred and preferred.exists():
        # Keep fixed fallback priority (PDF -> companion DOCX -> OCR) and only
        # evaluate a preferred extra source afterward when it is distinct.
        candidates.append(("preferred", preferred))
    if enable_ocr_fallback:
        candidates.append(("ocr", pdf_file))

    deduped: list[tuple[str, Path]] = []
    seen_keys: set[tuple[str, str]] = set()
    for kind, src in candidates:
        key = (kind, str(src).lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((kind, src))

    best: Optional[dict] = None
    best_score = -1.0
    policy = (mojibake_policy or "auto_fallback").strip().lower()

    for kind, src in deduped:
        record = {"source": kind, "path": str(src), "status": "skipped"}
        if not src.exists():
            record["status"] = "missing"
            chain.append(record)
            continue
        try:
            if kind == "ocr":
                items = _ocr_pdf_to_chunks(src, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                items = _load_text_chunks_from_path(src, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            quality = evaluate_chunks_quality(items, lang=lang)
            record["status"] = "ok"
            record["quality"] = quality
            chain.append(record)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
            chain.append(record)
            continue

        candidate = {
            "source": kind,
            "source_path": str(src),
            "chunks": items,
            "quality": quality,
        }
        score = float(quality.get("score") or 0.0)
        if score > best_score:
            best = candidate
            best_score = score

        if policy == "warn_only":
            return {
                **candidate,
                "fallback_chain": chain,
                "fallback_reason": "",
            }

        if policy == "strict_fail":
            if quality.get("is_mojibake"):
                return {
                    "source": kind,
                    "source_path": str(src),
                    "chunks": [],
                    "quality": quality,
                    "fallback_chain": chain,
                    "fallback_reason": f"{kind} text quality below threshold under strict policy",
                    "error": "text quality failed under strict policy",
                }
            return {
                **candidate,
                "fallback_chain": chain,
                "fallback_reason": "",
            }

        if not quality.get("is_mojibake"):
            reason = ""
            if len(chain) > 1:
                reason = f"selected {kind} after fallback due to low text quality in earlier sources"
            return {
                **candidate,
                "fallback_chain": chain,
                "fallback_reason": reason,
            }

    if best is not None:
        return {
            **best,
            "fallback_chain": chain,
            "fallback_reason": "all sources were low quality; selected best-scoring source",
        }
    return {
        "source": "none",
        "source_path": "",
        "chunks": [],
        "quality": evaluate_text_quality("", lang=lang),
        "fallback_chain": chain,
        "fallback_reason": "no readable source available",
        "error": "failed to load any text source",
    }


def _hash_image_pixels(data: bytes) -> Optional[str]:
    if not data:
        return None
    if Image is not None:
        try:
            with Image.open(io.BytesIO(data)) as img:
                img = img.convert("RGBA")
                payload = img.tobytes()
                header = f"{img.size[0]}x{img.size[1]}:{img.mode}".encode("utf-8")
                return hashlib.sha256(header + payload).hexdigest()
        except Exception:  # noqa: BLE001
            pass
    return hashlib.sha256(data).hexdigest()


def _hash_pixmap_pixels(pix) -> Optional[str]:
    try:
        samples = pix.samples
        if not samples:
            return None
        header = f"{pix.width}x{pix.height}:{pix.n}:{int(getattr(pix, 'alpha', 0))}".encode(
            "utf-8"
        )
        return hashlib.sha256(header + samples).hexdigest()
    except Exception:  # noqa: BLE001
        return None


# Shared OpenAI client for image understanding (vision) calls.
_VISION_CLIENT: OpenAI | None = None
# Lightweight in-memory cache so that each image path is analyzed at most
# once per process.
_VISION_DESC_CACHE: dict[str, str] = {}
# Disable repeated image-generation calls for endpoints/models that already
# returned permanent client-side errors (e.g., 404 unsupported route).
_IMAGE_GEN_DISABLED_KEYS: dict[str, str] = {}
_IMAGE_GEN_DISABLED_LOGGED: set[str] = set()


def _get_vision_client() -> OpenAI:
    """Lazily construct a shared OpenAI-compatible client for vision models.

    Configuration uses environment variables instead of hardcoded keys:
      - API key: VISION_API_KEY or APIYI_API_KEY or OPENAI_API_KEY
      - Base URL: VISION_BASE_URL or APIYI_BASE_URL or OPENAI_BASE_URL
        (default https://api.apiyi.com/v1)

    This helper is used both by the LangChain ImageUnderstandingTool and by
    any internal pipeline steps that want to analyze images.
    """

    global _VISION_CLIENT
    if _VISION_CLIENT is not None:
        return _VISION_CLIENT

    api_key = (
        os.getenv("VISION_API_KEY")
        or os.getenv("IMAGE_API_KEY")
        or os.getenv("image_api_key")
        or os.getenv("APIYI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        os.getenv("VISION_BASE_URL")
        or os.getenv("IMAGE_BASE_URL")
        or os.getenv("image_base_url")
        or os.getenv("APIYI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.apiyi.com/v1"
    )

    if not api_key:
        raise RuntimeError(
            "No vision API key configured; set VISION_API_KEY / IMAGE_API_KEY / image_api_key "
            "or APIYI_API_KEY / OPENAI_API_KEY."
        )

    _VISION_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _VISION_CLIENT


def analyze_image_with_vision(image_path: str, prompt: str | None = None) -> str:
    """Analyze a local image with a multimodal model and return a description.

    This is a low-level helper used by both the LangChain tool and the
    deterministic CLI pipeline to enrich figure metadata for better
    figure/text matching.
    """

    path = Path(image_path).expanduser().resolve()
    cache_key = str(path)
    cached = _VISION_DESC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        print(f"[Vision] Image not found for analysis: {path}")
        return ""

    if prompt is None:
        prompt = "请用中文简要描述并分析这张图的内容，用于论文汇报 PPT 配图匹配。"

    try:
        data = path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        print(f"[Vision] Failed to read image bytes from {path}: {exc}")
        return ""

    if not data:
        return ""

    b64 = base64.b64encode(data).decode("utf-8")
    suffix = path.suffix.lower()
    if suffix in {".png", ".apng"}:
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp"}:
        mime = "image/jpeg"
    else:
        mime = "image/png"

    try:
        client = _get_vision_client()
    except Exception as exc:  # noqa: BLE001
        print(f"[Vision] No usable vision client configured: {exc}")
        return ""

    model = (
        os.getenv("VISION_MODEL")
        or os.getenv("LLM_MODEL")
        or "gpt-4o"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                            },
                        },
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Vision] Vision API call failed for {path}: {exc}")
        return ""

    try:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        message = choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            return ""
        # content is usually a plain string; fall back to joining pieces when it
        # is returned as a structured list.
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = str(part.get("text") or "").strip()
                    if txt:
                        parts.append(txt)
                elif isinstance(part, str):
                    txt = part.strip()
                    if txt:
                        parts.append(txt)
            desc = " ".join(parts)
        else:
            desc = str(content).strip()

        desc = desc.strip()
        _VISION_DESC_CACHE[cache_key] = desc
        return desc
    except Exception as exc:  # noqa: BLE001
        print(f"[Vision] Failed to parse vision response for {path}: {exc}")
        return ""


class ImageUnderstandingInput(BaseModel):
    image_path: str = Field(description="Path to the local image file to analyze.")
    prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional natural-language instruction for the image analysis. "
            "If omitted, a Chinese default prompt for PPT figure matching is used."
        ),
    )


@tool(
    "ImageUnderstandingTool",
    args_schema=ImageUnderstandingInput,
    description=(
        "Use a multimodal OpenAI-compatible model to analyze a local image file "
        "and return a natural-language description. This is useful for "
        "understanding figures when planning PPT slides or improving "
        "figure/text alignment."
    ),
)
def image_understanding_tool(image_path: str, prompt: Optional[str] = None) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return json.dumps(
            {
                "image_path": str(path),
                "description": "",
                "error": f"image not found: {path}",
            }
        )

    prompt_text = prompt or "请用中文简要描述并分析这张图的内容，用于论文汇报 PPT 配图匹配。"
    try:
        desc = analyze_image_with_vision(str(path), prompt_text)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "image_path": str(path),
                "description": "",
                "error": f"vision analysis failed: {exc}",
            }
        )

    return json.dumps({"image_path": str(path), "description": desc})


def _load_text_chunks_from_path(path: Path, chunk_size: int = 2000, chunk_overlap: int = 200) -> list[dict]:
    suffix = path.suffix.lower()

    docs: list[Document] = []
    if suffix == ".pdf":
        try:
            import pdfplumber  # type: ignore[import]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "pdfplumber is not installed; please install it with 'pip install pdfplumber'"
            ) from exc

        with pdfplumber.open(str(path)) as pdf:  # type: ignore[arg-type]
            for page_index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_index},
                    )
                )
    elif suffix in {".md", ".markdown", ".mkd", ".txt"}:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to read text file: {exc}") from exc
        docs.append(Document(page_content=text, metadata={"page": 1}))
    elif suffix == ".docx":
        try:
            import docx  # type: ignore[import]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "python-docx is not installed; please install it with 'pip install python-docx'"
            ) from exc

        try:
            doc_obj = docx.Document(str(path))  # type: ignore[call-arg]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to parse docx file: {exc}") from exc

        paragraphs = [p.text for p in getattr(doc_obj, "paragraphs", []) if getattr(p, "text", "")]  # type: ignore[attr-defined]
        text = "\n".join(paragraphs)
        docs.append(Document(page_content=text, metadata={"page": 1}))
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"failed to read document as text: {exc}") from exc
        docs.append(Document(page_content=text, metadata={"page": 1}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    items: list[dict] = []
    for doc in chunks:
        items.append({"page_content": doc.page_content, "metadata": dict(doc.metadata or {})})
    return items


class PDFLoaderInput(BaseModel):
    pdf_path: str
    chunk_size: int = 2000
    chunk_overlap: int = 200


@tool(
    "PDFLoaderTool",
    args_schema=PDFLoaderInput,
    description="Load a PDF file and split it into text chunks with metadata (page, section, etc.)",
)
def pdf_loader_tool(pdf_path: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> str:
    path = Path(pdf_path).expanduser().resolve()
    print(f"[PDFLoaderTool] Start loading PDF: {path}")
    if not path.exists():
        return json.dumps({"error": f"document not found: {path}", "chunks": []})
    try:
        items = _load_text_chunks_from_path(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"failed to load document: {exc}", "chunks": []})
    print(f"[PDFLoaderTool] Finished loading document, got {len(items)} text chunks")
    return json.dumps({"chunks": items})


class FigureIndexerInput(BaseModel):
    figures_dir: str


@tool(
    "FigureIndexerTool",
    args_schema=FigureIndexerInput,
    description=(
        "Scan a figures directory and build an index of image files with simple filename-based hints. "
        "The guessed_caption field is only a rough guess from the filename and MUST NOT be used as the "
        "final figure title in the PPT; when constructing plan_json.figures you must replace it with the "
        "true caption text extracted from the original PDF (e.g., 'Figure 1. ...')."
    ),
)
def figure_indexer_tool(figures_dir: str) -> str:
    base = Path(figures_dir).expanduser().resolve()
    if not base.exists():
        return json.dumps({"figures": [], "warning": f"figures_dir not found: {base}"})
    exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    figures = []
    num_small_skipped = 0
    min_side = 500
    print(f"[FigureIndexerTool] Start indexing figures directory: {base}")
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                width, height = get_image_dimensions(path)
                if width is not None and height is not None:
                    if width < min_side or height < min_side:
                        num_small_skipped += 1
                        continue
            slug = path.stem
            guessed_caption = slug.replace("_", " ")
            figures.append({"slug": slug, "path": str(path), "guessed_caption": guessed_caption})
    if num_small_skipped:
        print(f"[FigureIndexerTool] Skipped {num_small_skipped} small images (<{min_side}x{min_side})")
    print(f"[FigureIndexerTool] Finished indexing, found {len(figures)} images")
    return json.dumps({"figures": figures})


class KeywordSearchInput(BaseModel):
    pdf_path: str = Field(description="Path to the source PDF file for sentence/phrase-based retrieval.")
    query: str = Field(description="Search query string; should be one or more short natural-language sentences instead of bare keywords.")
    pages: Optional[List[int]] = Field(
        default=None,
        description="Optional list of page numbers to restrict the search to.",
    )
    k: int = Field(default=8, description="Maximum number of relevant chunks to return.")


_KEYWORD_INDEX_CACHE: dict[str, List[dict]] = {}


@tool(
    "KeywordSearchTool",
    args_schema=KeywordSearchInput,
    description=(
        "Perform a lightweight sentence/phrase-based RAG search over a paper PDF. "
        "This tool internally uses PDFLoaderTool to split the PDF into text chunks, "
        "builds a simple text index (via build_keyword_index), and then returns up to k "
        "relevant chunks whose text best matches the query sentences as substrings. Use this "
        "when you need additional context from the original paper beyond the current section text."
    ),
)
def keyword_search_tool(pdf_path: str, query: str, pages: Optional[List[int]] = None, k: int = 8) -> str:
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        return json.dumps({"results": [], "error": f"document not found: {path}"})

    cache_key = str(path)
    index = _KEYWORD_INDEX_CACHE.get(cache_key)

    if index is None:
        raw = pdf_loader_tool.invoke({"pdf_path": str(path)})
        try:
            obj = json.loads(raw)
            chunks = obj.get("chunks", [])
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {"results": [], "error": f"failed to load document chunks for RAG: {exc}"}
            )

        index = build_keyword_index(chunks)
        _KEYWORD_INDEX_CACHE[cache_key] = index

    try:
        results = keyword_search(
            index,
            query=query,
            pages=None,
            k=k,
            preferred_pages=pages or None,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"results": [], "error": f"keyword search failed: {exc}"})

    items: List[dict] = []
    for item in results:
        meta = item.get("metadata") or {}
        items.append(
            {
                "page": meta.get("page"),
                "index": item.get("index"),
                "page_content": (item.get("page_content") or "")[:500],
            }
        )

    return json.dumps({"results": items})


class FigureCaptionAlignInput(BaseModel):
    pdf_path: str
    figures_dir: str


@tool(
    "FigureCaptionAlignTool",
    args_schema=FigureCaptionAlignInput,
    description=(
        "Extract numbered figure captions from the PDF text (e.g., 'Figure 1. ...') and align them "
        "with images in the figures directory. The alignment is heuristic: images are matched to captions "
        "primarily by page index encoded in filenames like 'page_002_img_01.png'. The result ensures there "
        "is at most one image per numbered figure in the final PPT plan."
    ),
)
def figure_caption_align_tool(pdf_path: str, figures_dir: str) -> str:
    """Extract figure captions from PDF and align them with page_*_img_* images.

    This is a heuristic alignment: for each numbered figure caption found on a page, we assign the next
    available extracted image from that same page (based on filename ordering). If no image is available
    on that page, we fall back to any remaining unassigned images.
    """
    import re

    pdf_file = Path(pdf_path).expanduser().resolve()
    base = Path(figures_dir).expanduser().resolve()

    if not pdf_file.exists():
        return json.dumps({"figures": [], "error": f"PDF not found: {pdf_file}"})
    if not base.exists():
        return json.dumps({"figures": [], "error": f"figures_dir not found: {base}"})

    try:
        import pdfplumber
    except Exception:
        return json.dumps(
            {
                "figures": [],
                "error": "pdfplumber is not installed; please install it with 'pip install pdfplumber'",
            }
        )

    # Collect extracted images grouped by page from filenames like 'page_002_img_01.*'
    images_by_page = {}
    image_pattern = re.compile(r"page_(\d+)_img_(\d+)")
    exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        m = image_pattern.search(path.name)
        if not m:
            continue
        page_index = int(m.group(1))
        img_index = int(m.group(2))
        images_by_page.setdefault(page_index, []).append({"path": str(path), "img_index": img_index})
    for page_index in images_by_page:
        images_by_page[page_index].sort(key=lambda x: x["img_index"])

    print(
        f"[FigureCaptionAlignTool] Start aligning figure captions: pdf={pdf_file}, figures_dir={base}, "
        f"pages_with_images={len(images_by_page)}"
    )

    captions = []
    caption_pattern = re.compile(r"((?:Figure|Fig\.)\s*\d+[^\n]*)", re.IGNORECASE)

    try:
        with pdfplumber.open(str(pdf_file)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                for match in caption_pattern.finditer(text):
                    full = match.group(1).strip()
                    num_match = re.search(r"\d+", full)
                    fig_num = int(num_match.group(0)) if num_match else None
                    # Capture a local textual context around the caption so that downstream
                    # planning can use the original description and analysis from the paper.
                    start_idx = max(0, match.start() - 400)
                    end_idx = min(len(text), match.end() + 400)
                    context = text[start_idx:end_idx].strip()
                    captions.append(
                        {
                            "page": page_index,
                            "figure_number": fig_num,
                            "caption": full,
                            "context": context,
                        }
                    )
    except Exception as exc:
        return json.dumps({"figures": [], "error": f"failed to parse pdf for captions: {exc}"})

    print(f"[FigureCaptionAlignTool] Found {len(captions)} figure captions in PDF text")

    # Align captions to images by page
    figures_output = []
    unassigned_images = []
    for page_index, imgs in images_by_page.items():
        unassigned_images.extend([{**img, "page": page_index} for img in imgs])

    # Track per-page image cursor
    image_cursor = {page: 0 for page in images_by_page}

    for idx, cap in enumerate(captions, start=1):
        page = cap["page"]
        fig_num = cap["figure_number"]
        caption_text = cap["caption"]
        context_text = cap.get("context")

        chosen = None
        if page in images_by_page:
            cursor = image_cursor.get(page, 0)
            imgs = images_by_page[page]
            if cursor < len(imgs):
                chosen = imgs[cursor]
                image_cursor[page] = cursor + 1

        if chosen is None and unassigned_images:
            chosen = unassigned_images.pop(0)

        if chosen is None:
            # No available image; skip this caption
            continue

        if fig_num is not None:
            slug = f"figure_{fig_num:02d}"
        else:
            slug = f"figure_{idx:03d}"

        figures_output.append(
            {
                "slug": slug,
                "path": chosen["path"],
                "caption": caption_text,
                "context": context_text,
                "page": chosen.get("page", page),
                "figure_number": fig_num,
            }
        )

    print(f"[FigureCaptionAlignTool] Aligned {len(figures_output)} figures with images")
    return json.dumps({"figures": figures_output})


class PdfFigureExtractInput(BaseModel):
    pdf_path: str
    output_dir: str = "figures/auto_from_pdf"


class DocxFigureExtractInput(BaseModel):
    docx_path: str
    output_dir: str = "figures/auto_from_docx"


def _extract_page_render_candidates(
    pdf_path: Path,
    out_dir: Path,
    min_side: int,
    seen_hashes: set[str],
    max_pages: int = 12,
) -> tuple[list[dict], int, int]:
    """Render PDF pages as fallback figure candidates when embedded images are scarce."""
    images: list[dict] = []
    num_small_skipped = 0
    num_duplicates = 0
    if not pdf_path.exists():
        return images, num_small_skipped, num_duplicates
    try:
        pages = convert_from_path(str(pdf_path), dpi=170, first_page=1, last_page=max_pages)
    except Exception:
        return images, num_small_skipped, num_duplicates

    for idx, page in enumerate(pages, start=1):
        try:
            width, height = page.size
            if width < min_side or height < min_side:
                num_small_skipped += 1
                continue
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            data = buf.getvalue()
            img_hash = _hash_image_pixels(data)
            if img_hash and img_hash in seen_hashes:
                num_duplicates += 1
                continue
            if img_hash:
                seen_hashes.add(img_hash)
            filename = f"page_{idx:03d}_render_01.png"
            out_path = out_dir / filename
            with out_path.open("wb") as f:
                f.write(data)
            images.append({"path": str(out_path), "page": idx})
        except Exception:  # noqa: BLE001
            continue
    return images, num_small_skipped, num_duplicates


@tool(
    "PdfFigureExtractTool",
    args_schema=PdfFigureExtractInput,
    description=(
        "Extract embedded images from a paper PDF into a local figures directory. "
        "This tool only dumps raw images (possibly more than true figures). Later, when you "
        "construct plan_json.figures, you must align these images with the real numbered figure "
        "captions found in the PDF text (e.g., 'Figure 1', 'Fig. 2') and keep exactly one image "
        "per numbered figure in the final PPT."
    ),
)
def pdf_figure_extract_tool(pdf_path: str, output_dir: str = "figures/auto_from_pdf") -> str:
    path = Path(pdf_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return json.dumps({"images": [], "error": f"PDF not found: {path}"})
    try:
        import fitz  # type: ignore
    except Exception:
        return json.dumps({"images": [], "error": "PyMuPDF (fitz) is not installed"})
    try:
        import pdfplumber  # type: ignore
    except Exception:
        pdfplumber = None  # type: ignore[assignment]

    doc = fitz.open(str(path))
    pdf_for_tables = None
    if pdfplumber is not None:
        try:
            pdf_for_tables = pdfplumber.open(str(path))  # type: ignore[call-arg]
        except Exception as exc:  # noqa: BLE001
            print(
                f"[PdfFigureExtractTool] Warning: failed to open PDF with pdfplumber for table detection: {exc}"
            )
            pdf_for_tables = None

    images: list[dict] = []
    num_table_screenshots = 0
    num_docx_images = 0
    num_render_candidates = 0
    num_small_skipped = 0
    num_duplicates = 0
    seen_hashes: set[str] = set()
    print(f"[PdfFigureExtractTool] Start extracting images from PDF: {path} -> {out_dir}")
    min_side = 500

    def _large_enough(width: Optional[int], height: Optional[int], data: Optional[bytes] = None) -> bool:
        if width is None or height is None:
            if data and Image is not None:
                try:
                    with Image.open(io.BytesIO(data)) as img:
                        width, height = img.size
                except Exception:  # noqa: BLE001
                    return True
            else:
                return True
        return width >= min_side and height >= min_side

    try:
        page_count = doc.page_count  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        page_count = len(doc)

    for page_index in range(1, page_count + 1):
        page = doc[page_index - 1]
        table_page = None
        if pdf_for_tables is not None and 0 <= page_index - 1 < len(pdf_for_tables.pages):  # type: ignore[operator]
            table_page = pdf_for_tables.pages[page_index - 1]  # type: ignore[index]

        # First, extract embedded raster images as before.
        embedded_images = page.get_images(full=True)
        for image_index, img in enumerate(embedded_images, start=1):
            xref = img[0]
            try:
                # Use extract_image to obtain raw image bytes and a suitable file
                # extension. This avoids colorspace issues when saving Pixmaps
                # directly to PNG for exotic or unsupported colorspaces.
                info = doc.extract_image(xref)
                image_bytes = info.get("image")
                ext = info.get("ext", "png")
                if not image_bytes:
                    continue
                width = info.get("width")
                height = info.get("height")
                if not _large_enough(width, height, image_bytes):
                    num_small_skipped += 1
                    continue
                img_hash = _hash_image_pixels(image_bytes)
                if img_hash and img_hash in seen_hashes:
                    num_duplicates += 1
                    continue
                if img_hash:
                    seen_hashes.add(img_hash)
                filename = f"page_{page_index:03d}_img_{image_index:02d}.{ext}"
                output_path = out_dir / filename
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                images.append({"path": str(output_path), "page": page_index})
            except Exception as exc:  # noqa: BLE001
                # Do not crash the whole pipeline because of a single problematic image.
                print(
                    f"[PdfFigureExtractTool] Skipping image xref={xref} on page {page_index} due to error: {exc}"
                )
                continue

        # Additionally, when pdfplumber is available, detect tabular regions and
        # save them as high-resolution screenshots so that tables in the running
        # text can also appear as figures in the PPT.
        if table_page is not None:
            try:
                tables = table_page.find_tables() or []  # type: ignore[call-arg]
            except Exception as exc:  # noqa: BLE001
                print(f"[PdfFigureExtractTool] Warning: failed to detect tables on page {page_index}: {exc}")
                tables = []

            next_index = len(embedded_images)
            for t_idx, tbl in enumerate(tables, start=1):
                try:
                    bbox = getattr(tbl, "bbox", None)
                    if not bbox or len(bbox) != 4:
                        continue
                    x0, top, x1, bottom = bbox
                    rect = fitz.Rect(x0, top, x1, bottom)
                    # Use a modest zoom factor so that cell text remains readable.
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
                    if pix.width < min_side or pix.height < min_side:
                        num_small_skipped += 1
                        continue
                    pix_hash = None
                    try:
                        pix_bytes = pix.tobytes("png")
                        pix_hash = _hash_image_pixels(pix_bytes)
                    except Exception:  # noqa: BLE001
                        pix_hash = None
                    if pix_hash and pix_hash in seen_hashes:
                        num_duplicates += 1
                        continue
                    if pix_hash:
                        seen_hashes.add(pix_hash)
                    img_index = next_index + t_idx
                    filename = f"page_{page_index:03d}_img_{img_index:02d}.png"
                    output_path = out_dir / filename
                    pix.save(str(output_path))
                    images.append({"path": str(output_path), "page": page_index})
                    num_table_screenshots += 1
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[PdfFigureExtractTool] Skipping table screenshot on page {page_index} due to error: {exc}"
                    )
                    continue

    if pdf_for_tables is not None:
        try:
            pdf_for_tables.close()  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001
            pass
    doc.close()

    # Fallback 1: companion DOCX media assets.
    companion_docx = path.with_suffix(".docx")
    if companion_docx.exists():
        docx_images, docx_small, docx_dedup = _extract_docx_media_images(
            companion_docx, out_dir, min_side, seen_hashes
        )
        if docx_images:
            images.extend(docx_images)
            num_docx_images = len(docx_images)
        num_small_skipped += docx_small
        num_duplicates += docx_dedup

    # Fallback 2: page render candidates (useful for vector-only PDFs).
    enable_render_fallback = (os.getenv("PIPELINE_ENABLE_PAGE_RENDER_FIGURES", "1").strip().lower() not in {"0", "false", "no"})
    if enable_render_fallback and len(images) < 8:
        render_images, render_small, render_dedup = _extract_page_render_candidates(
            path,
            out_dir,
            min_side=min_side,
            seen_hashes=seen_hashes,
            max_pages=12,
        )
        if render_images:
            images.extend(render_images)
            num_render_candidates = len(render_images)
        num_small_skipped += render_small
        num_duplicates += render_dedup

    print(
        f"[PdfFigureExtractTool] Finished extraction: {len(images)} images saved to {out_dir} "
        f"(including {num_table_screenshots} table screenshots, "
        f"{num_docx_images} DOCX images, {num_render_candidates} rendered page candidates, "
        f"skipped {num_small_skipped} small images, "
        f"deduped {num_duplicates} duplicates)"
    )
    return json.dumps({"images": images})


@tool(
    "DocxFigureExtractTool",
    args_schema=DocxFigureExtractInput,
    description="Extract images from DOCX word/media into a figures directory.",
)
def docx_figure_extract_tool(docx_path: str, output_dir: str = "figures/auto_from_docx") -> str:
    path = Path(docx_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return json.dumps({"images": [], "error": f"DOCX not found: {path}"})
    seen_hashes: set[str] = set()
    images, num_small_skipped, num_duplicates = _extract_docx_media_images(
        path, out_dir, min_side=500, seen_hashes=seen_hashes
    )
    print(
        f"[DocxFigureExtractTool] Finished extraction: {len(images)} images saved to {out_dir} "
        f"(skipped {num_small_skipped} small images, deduped {num_duplicates} duplicates)"
    )
    return json.dumps({"images": images})


class SectionPlannerInput(BaseModel):
    sections: List[str] = Field(default_factory=list)
    max_frames: int = 40


@tool(
    "SectionPlannerTool",
    args_schema=SectionPlannerInput,
    description=(
        "Create an initial PPT frame outline from a list of section titles. "
        "This tool is only for a coarse outline (about 1 frame per section); "
        "you must then expand it into a detailed plan_json with adaptive 2-8 frames "
        "per section (respect max_frames_per_section, hard cap 10), covering the whole paper, including separate frames "
        "for methods, equations, experiments, ablations, and conclusions."
    ),
)
def section_planner_tool(sections: List[str], max_frames: int = 40) -> str:
    frames = []
    for index, section in enumerate(sections[:max_frames]):
        frames.append(
            {
                "section": section,
                "subsection": None,
                "frame_title": section,
                "bullets": [],
                "figure_slugs": [],
            }
        )
    print(
        f"[SectionPlannerTool] Planned coarse frames: {len(sections)} input sections -> {len(frames)} initial frames "
        f"(should later be expanded adaptively to around 2-8 frames per section in the detailed plan)"
    )
    return json.dumps({"frames": frames})


class LatexWriterInput(BaseModel):
    template_path: str
    output_path: str
    plan_json: str
    title: Optional[str] = None
    subtitle: Optional[str] = None
    author: Optional[str] = None
    institute: Optional[str] = None
    date: Optional[str] = None
    lang: Optional[str] = None


@tool(
    "LatexWriterTool",
    args_schema=LatexWriterInput,
    description=(
        "Render a new LaTeX Beamer file from a template and a structured PPT plan. "
        "The bullets in plan_json.frames[*].bullets are inserted into the LaTeX frames "
        "as raw LaTeX content (without escaping), so they may contain math environments "
        "and formulas; you must ensure the LaTeX syntax is valid."
    ),
)
def latex_writer_tool(
    template_path: str,
    output_path: str,
    plan_json: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    author: Optional[str] = None,
    institute: Optional[str] = None,
    date: Optional[str] = None,
    lang: Optional[str] = None,
) -> str:
    import re

    template = Path(template_path).expanduser().resolve()
    if not template.exists():
        print(f"[LatexWriterTool] Template not found: {template}")
        return json.dumps({"error": f"template not found: {template}"})
    content = template.read_text(encoding="utf-8")

    def _inject_after_documentclass(text: str, snippet: str) -> str:
        m_doc = re.search(r"\\documentclass[^\n]*\n", text)
        if m_doc:
            insert_pos = m_doc.end()
            return text[:insert_pos] + snippet + text[insert_pos:]
        return snippet + text

    def _ensure_package(text: str, package: str, options: str | None = None) -> str:
        pattern = rf"\\usepackage(?:\[[^\]]+\])?\{{{re.escape(package)}\}}"
        if re.search(pattern, text):
            return text
        opt = f"[{options}]" if options else ""
        return _inject_after_documentclass(text, f"\\usepackage{opt}{{{package}}}\n")

    # Remove explicit frame-number footline so that slide page numbers are not shown.
    content = re.sub(r"\\setbeamertemplate\{footline\}\[[^\]]*\]", "", content)

    # Ensure essential packages are available for figures and math.
    content = _ensure_package(content, "graphicx")
    content = _ensure_package(content, "grffile")
    content = _ensure_package(content, "amsmath")
    content = _ensure_package(content, "amssymb")
    content = _ensure_package(content, "booktabs")

    # Globally shrink figure/table caption font so that long captions are less
    # likely to overflow the visual frame. If the template has not customized
    # caption font yet, set it to \scriptsize.
    if "\\setbeamerfont{caption}" not in content:
        m_cap = re.search(r"\\documentclass[^\n]*\n", content)
        if m_cap:
            insert_pos = m_cap.end()
            content = (
                content[:insert_pos]
                + "\\setbeamerfont{caption}{size=\\scriptsize}\n"
                + content[insert_pos:]
            )
        else:
            content = "\\setbeamerfont{caption}{size=\\scriptsize}\n" + content

    # If language is Chinese and the template does not already include ctex,
    # automatically inject a basic Chinese support package into the preamble.
    if lang and str(lang).lower().startswith("zh") and "ctex" not in content:
        m = re.search(r"\\documentclass[^\n]*\n", content)
        if m:
            insert_pos = m.end()
            content = content[:insert_pos] + "\\usepackage[UTF8]{ctex}\n" + content[insert_pos:]
        else:
            # Fallback: prepend to the very beginning
            content = "\\usepackage[UTF8]{ctex}\n" + content
    if lang and str(lang).lower().startswith("zh") and "setCJKmainfont" not in content:
        font_block = (
            "\\ifdefined\\setCJKmainfont\n"
            "\\IfFontExistsTF{Microsoft YaHei}{\\setCJKmainfont{Microsoft YaHei}}{\n"
            "\\IfFontExistsTF{SimSun}{\\setCJKmainfont{SimSun}}{\n"
            "\\IfFontExistsTF{FandolSong-Regular}{\\setCJKmainfont{FandolSong-Regular}}{}\n"
            "}}\n"
            "\\fi\n"
        )
        m_ctex = re.search(r"\\usepackage(?:\[[^\]]+\])?\{ctex\}[^\n]*\n", content)
        if m_ctex:
            insert_pos = m_ctex.end()
            content = content[:insert_pos] + font_block + content[insert_pos:]
        else:
            content = _inject_after_documentclass(content, font_block)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    def _escape_latex_value(value: str) -> str:
        # Escape LaTeX-sensitive characters in structural fields such as
        # titles, section names, and captions. These fields are not expected
        # to contain complex math, so it is safe to escape special symbols.
        escaped = value.replace("\\", r"\\")
        escaped = escaped.replace("{", r"\{").replace("}", r"\}")
        escaped = escaped.replace("&", r"\&")
        escaped = escaped.replace("%", r"\%")
        escaped = escaped.replace("#", r"\#")
        escaped = escaped.replace("_", r"\_")
        escaped = escaped.replace("$", r"\$")
        return escaped

    def _sanitize_mojibake_text(value: str, fallback: str = "") -> str:
        raw = (value or "").strip()
        if not raw:
            return fallback
        q = evaluate_text_quality(raw, lang=(lang or "zh"))
        has_placeholder = bool(re.search(r"(?:\bn{6,}\b|\bI{6,}\b|[nI]{10,})", raw))
        if q.get("is_mojibake") or has_placeholder:
            cleaned = raw
            for token in KNOWN_MOJIBAKE_TOKENS:
                if token in {"nnnn", "IIII"}:
                    continue
                cleaned = cleaned.replace(token, "")
            cleaned = re.sub(r"(?:\bn{6,}\b|\bI{6,}\b|[nI]{10,})", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned and not evaluate_text_quality(cleaned, lang=(lang or "zh")).get("is_mojibake"):
                return cleaned
            return fallback
        return raw

    def _sanitize_inline_latex_text(value: str) -> str:
        """Lightly sanitize inline LaTeX text coming from bullets/paragraphs.

        We keep the text mostly raw so that math environments still work, but
        we do escape special characters in *non-math* segments of the text so
        that plain strings like ``Bearing1_3`` do not break compilation.

        Additionally, for definition-style sentences such as ``...：公式`` we
        heuristically wrap only the suffix after the colon in inline math and
        normalize common Unicode/math-like characters into LaTeX so that
        copied formulas (e.g., UAV dynamics or neighborhood definitions) render
        properly.
        """

        if not value:
            return value

        # Normalize common mis-typed display math delimiters like "\\[ ... \\]"
        # or "\\( ... \\)" into their canonical single-backslash forms. When
        # left as-is, sequences such as "\\[" are parsed by LaTeX as a linebreak
        # with an optional length argument ("\\[<len>]") and will trigger
        # "Missing number, treated as zero" errors at the end of the frame.
        value = (
            value.replace("\\\\[", "\\[")
            .replace("\\\\]", "\\]")
            .replace("\\\\(", "\\(")
            .replace("\\\\)", "\\)")
        )

        # First, handle complete patterns of the form "$\\[ ... \\]$" or
        # "$\\[ ... \\](17)$" by stripping the outer inline-math dollars and
        # keeping only the display math "\\[ ... \\]" plus any trailing
        # equation-style number such as "(17)". This avoids nested math
        # environments like "$\\[ ... \\]$" that lead to "Bad math
        # environment delimiter" and frame-level beamer errors.
        value = re.sub(
            r"(?s)\$\s*(\\\[.*?\\\])(\s*\(\d+\))?\s*\$",
            r"\1\2",
            value,
        )

        # As a secondary safeguard, remove stray leading/trailing dollars that
        # immediately wrap a display math delimiter.
        value = re.sub(r"\$\s*(\\\\\[)", r"\\[", value)
        value = re.sub(r"(\\\\\])\s*\$", r"\\]", value)

        # Heuristic: for sentences of the form "...：公式" or "...: formula", when
        # the suffix after the first colon clearly looks like a formula, wrap
        # that suffix in inline math and normalize common Unicode/math
        # characters so that the resulting expression compiles and renders as
        # expected.
        if "$" not in value:
            colon_pos = value.find("：")
            if colon_pos == -1:
                colon_pos = value.find(":")
            if colon_pos != -1:
                prefix = value[: colon_pos + 1]
                rest = value[colon_pos + 1 :]
                rest_stripped = rest.strip()
                if rest_stripped:
                    math_like = False
                    # Typical LaTeX-style subscripts or set-like notation.
                    if re.search(r"[A-Za-z0-9]\s*_[A-Za-z0-9{]", rest_stripped):
                        math_like = True
                    # Common Unicode math / set symbols.
                    if not math_like and re.search(r"[\u2208\u2209\u2205\u2282\u2286\u2284\u2287\u222a\u2229\u2211\u220f\u221a\u221e\u2248\u2260\u2264\u2265\u00b1\u00d7\u00f7]", rest_stripped):
                        math_like = True
                    # Unicode bold/italic math letters frequently seen when
                    # copying formulas from PDFs (e.g., 饾拺, 饾懐, 饾惖, 饾懢, 饾湈, 饾湉).
                    if not math_like and re.search(r"[饾拺饾挅饾拻饾挊饾懐饾懆饾懢饾懡饾懍饾懏饾懙饾惔饾惖饾惗饾惙饾惛饾湈饾湉饾渷饾潈]", rest_stripped):
                        math_like = True

                    if math_like:
                        # If the suffix already starts with a display-math
                        # delimiter like '\[', do NOT wrap it in '$...$'.
                        # Instead, only normalize Unicode/math-like characters
                        # inside that display expression and leave it as
                        # '\[ ... \]' so that we do not create nested math
                        # environments such as '$\[ ... \]$'.
                        if rest_stripped.lstrip().startswith(r"\["):

                            def _normalize_formula(s: str) -> str:
                                # Undo previous escaping that turned subscripts and
                                # superscripts into plain characters.
                                s = s.replace(r"\_", "_").replace(r"\^", "^")

                                # Map common Unicode operators and delimiters.
                                replacements = {
                                    "≤": r"\leq",
                                    "≥": r"\geq",
                                    "≠": r"\ne",
                                    "∈": r"\in",
                                    "‖": r"\|",
                                    "⊙": r"\odot",
                                }
                                for k, v in replacements.items():
                                    s = s.replace(k, v)

                                # Handle common transpose markers.
                                s = s.replace("^⊤", r"^{\top}")
                                s = s.replace("⊤", r"\top")

                                # Greek letters that often appear as Unicode.
                                s = s.replace("蟿", r"\tau").replace("饾湉", r"\tau")
                                s = s.replace("蟽", r"\sigma").replace("饾湈", r"\sigma")
                                s = s.replace("味", r"\zeta").replace("饾渷", r"\zeta")

                                # Xi variants (state vectors, etc.).
                                s = s.replace("尉", r"\xi").replace("饾湁", r"\xi").replace("饾潈", r"\xi")

                                # Bold/blackboard letters from Unicode math alphabets.
                                bold_map = {
                                    "饾拺": r"\boldsymbol{p}",
                                    "饾挅": r"\boldsymbol{u}",
                                    "饾拻": r"\boldsymbol{q}",
                                    "饾挊": r"\boldsymbol{w}",
                                    "饾懐": r"\mathbf{H}",
                                    "饾懆": r"\mathbf{A}",
                                    "饾懢": r"\mathbf{W}",
                                    "饾懡": r"\mathbf{V}",
                                    "饾懍": r"\mathbf{E}",
                                    "饾懏": r"\mathbf{G}",
                                    "饾懙": r"\mathbf{N}",
                                    "饾惖": r"\mathbf{B}",
                                }
                                for k, v in bold_map.items():
                                    s = s.replace(k, v)

                                # Unicode math italic letters that often appear
                                # as plain variables in copied formulas.
                                italic_map = {
                                    "饾憳": "k",
                                    "饾憴": "l",
                                }
                                for k, v in italic_map.items():
                                    s = s.replace(k, v)

                                return s

                            norm_display = _normalize_formula(rest_stripped)
                            value = prefix + norm_display
                        else:
                            formula_text = rest_stripped
                            suffix = ""
                            # Try to keep trailing Chinese punctuation outside math.
                            m_punct = re.search(r"[，。；;]", rest_stripped)
                            if m_punct:
                                formula_text = rest_stripped[: m_punct.start()].rstrip()
                                suffix = rest_stripped[m_punct.start() :]

                            def _normalize_formula(s: str) -> str:
                                # Undo previous escaping that turned subscripts and
                                # superscripts into plain characters.
                                s = s.replace(r"\_", "_").replace(r"\^", "^")

                                # Map common Unicode operators and delimiters.
                                replacements = {
                                    "≤": r"\leq",
                                    "≥": r"\geq",
                                    "≠": r"\ne",
                                    "∈": r"\in",
                                    "‖": r"\|",
                                    "⊙": r"\odot",
                                }
                                for k, v in replacements.items():
                                    s = s.replace(k, v)

                                # Handle common transpose markers.
                                s = s.replace("^⊤", r"^{\top}")
                                s = s.replace("⊤", r"\top")

                                # Greek letters that often appear as Unicode.
                                s = s.replace("蟿", r"\tau").replace("饾湉", r"\tau")
                                s = s.replace("蟽", r"\sigma").replace("饾湈", r"\sigma")
                                s = s.replace("味", r"\zeta").replace("饾渷", r"\zeta")

                                # Xi variants (state vectors, etc.).
                                s = s.replace("尉", r"\xi").replace("饾湁", r"\xi").replace("饾潈", r"\xi")

                                # Bold/blackboard letters from Unicode math alphabets.
                                bold_map = {
                                    "饾拺": r"\boldsymbol{p}",
                                    "饾挅": r"\boldsymbol{u}",
                                    "饾拻": r"\boldsymbol{q}",
                                    "饾挊": r"\boldsymbol{w}",
                                    "饾懐": r"\mathbf{H}",
                                    "饾懆": r"\mathbf{A}",
                                    "饾懢": r"\mathbf{W}",
                                    "饾懡": r"\mathbf{V}",
                                    "饾懍": r"\mathbf{E}",
                                    "饾懏": r"\mathbf{G}",
                                    "饾懙": r"\mathbf{N}",
                                    "饾惖": r"\mathbf{B}",
                                }
                                for k, v in bold_map.items():
                                    s = s.replace(k, v)

                                # Unicode math italic letters that often appear
                                # as plain variables in copied formulas.
                                italic_map = {
                                    "饾憳": "k",
                                    "饾憴": "l",
                                }
                                for k, v in italic_map.items():
                                    s = s.replace(k, v)

                                return s

                            if not rest_stripped.lstrip().startswith(r"\[") and formula_text:
                                norm = _normalize_formula(formula_text)
                                value = prefix + "$" + norm + "$" + suffix

        # If the string does not already contain any math delimiters but
        # clearly looks like LaTeX math (e.g., uses \\frac or \\sqrt), wrap the
        # whole string in inline math so that commands and subscripts are
        # interpreted correctly instead of causing "Missing $ inserted".
        if "$" not in value:
            math_markers = [
                "\\frac",
                "\\sqrt",
                "\\sum",
                "\\int",
                "\\begin{cases}",
                "\\end{cases}",
            ]
            has_display_math = any(
                marker in value
                for marker in ("\\[", "\\]", "\\begin{equation}", "\\end{equation}")
            )
            if not has_display_math and any(marker in value for marker in math_markers):
                value = "$" + value + "$"

        out_chars: list[str] = []
        i = 0
        in_math = False
        last_math_delim: str | None = None  # '$' or 'paren'
        length = len(value)

        while i < length:
            ch = value[i]

            # Toggle math mode on unescaped '$' and handle '$$...$$' display math.
            if ch == "$" and (i == 0 or value[i - 1] != "\\"):
                # Treat a pair of consecutive '$$' as a single display-math
                # delimiter so that the entire '$$...$$' region is considered
                # to be in math mode for the purposes of escaping.
                if value.startswith("$$", i):
                    out_chars.append("$$")
                    if not in_math:
                        in_math = True
                        last_math_delim = "dollars"
                    else:
                        in_math = False
                        last_math_delim = None
                    i += 2
                    continue

                out_chars.append("$")
                if not in_math:
                    in_math = True
                    last_math_delim = "dollar"
                else:
                    in_math = False
                    last_math_delim = None
                i += 1
                continue

            # Enter math mode with '\[' (display math)
            if not in_math and value.startswith(r"\[", i):
                out_chars.append(r"\[")
                in_math = True
                last_math_delim = "bracket"
                i += 2
                continue

            # Exit math mode with '\]' (display math)
            if in_math and value.startswith(r"\]", i):
                out_chars.append(r"\]")
                in_math = False
                last_math_delim = None
                i += 2
                continue

            # Enter math mode with '\(' (inline math)
            if not in_math and value.startswith(r"\(", i):
                out_chars.append(r"\(")
                in_math = True
                last_math_delim = "paren"
                i += 2
                continue

            # Exit math mode with '\)' (inline math)
            if in_math and value.startswith(r"\)", i):
                out_chars.append(r"\)")
                in_math = False
                last_math_delim = None
                i += 2
                continue

            if not in_math:
                # Escape special characters in normal text, but respect
                # already-escaped forms like '\&', '\_' etc.
                if ch == "&" and (i == 0 or value[i - 1] != "\\"):
                    out_chars.append(r"\&")
                elif ch == "_" and (i == 0 or value[i - 1] != "\\"):
                    out_chars.append(r"\_")
                elif ch == "%" and (i == 0 or value[i - 1] != "\\"):
                    out_chars.append(r"\%")
                elif ch == "#" and (i == 0 or value[i - 1] != "\\"):
                    out_chars.append(r"\#")
                elif ch == "$" and (i == 0 or value[i - 1] != "\\"):
                    out_chars.append(r"\$")
                elif ch == "^" and (i == 0 or value[i - 1] != "\\"):
                    # Caret outside of math mode acts as a superscript trigger
                    # and leads to 'Missing $ inserted' errors (e.g., in
                    # sequences like ']^鈯? coming from copied formulas). When
                    # we see an unescaped '^' in normal text, render it as a
                    # literal caret symbol instead of starting math.
                    out_chars.append(r"\^")
                else:
                    out_chars.append(ch)
            else:
                # Inside math: keep content as-is so that subscripts and other
                # constructs continue to work (e.g., E_{r_i}).
                out_chars.append(ch)

            i += 1

        # If we reach the end of the string still in math mode, close the
        # last opened math environment to avoid 'Missing $ inserted' errors
        # from stray delimiters coming from the planner.
        if in_math:
            if last_math_delim == "paren":
                out_chars.append(r"\)")
            elif last_math_delim == "bracket":
                out_chars.append(r"\]")
            else:
                # Default to '$' when we cannot confidently infer the opener.
                out_chars.append("$")

        return "".join(out_chars)

    if title is not None:
        safe_title = _escape_latex_value(title)
        content = re.sub(r"\\title\{.*?\}", lambda m: f"\\title{{{safe_title}}}", content)
    if subtitle is not None:
        safe_subtitle = _escape_latex_value(subtitle)
        content = re.sub(r"\\subtitle\{.*?\}", lambda m: f"\\subtitle{{{safe_subtitle}}}", content)
    if author is not None:
        safe_author = _escape_latex_value(author)
        content = re.sub(r"\\author\{.*?\}", lambda m: f"\\author{{{safe_author}}}", content)
    if institute is not None:
        safe_institute = _escape_latex_value(institute)
        content = re.sub(r"\\institute\{.*?\}", lambda m: f"\\institute{{{safe_institute}}}", content)
    if date is not None:
        safe_date = _escape_latex_value(date)
        content = re.sub(r"\\date\{.*?\}", lambda m: f"\\date{{{safe_date}}}", content)
    # Try to robustly parse plan_json, which may be wrapped in code fences or contain extra text.
    raw = (plan_json or "").strip()
    cleaned = raw
    # Strip simple Markdown code fences if present
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        # parts format: ["", "json\n{...}", ""] or similar
        if len(parts) >= 2:
            cleaned = parts[1].strip()
    # If there is leading/trailing text, try to keep only the JSON object body
    if cleaned and not cleaned.lstrip().startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    plan = None
    parse_error: Optional[Exception] = None
    for parser_name in ("json", "ast"):
        try:
            if parser_name == "json":
                plan = json.loads(cleaned)
            else:
                plan = ast.literal_eval(cleaned)
            break
        except Exception as exc:
            parse_error = exc

    if plan is None:
        # Log a short preview to help debug invalid JSON coming from the model
        raw_preview = raw[:500].replace("\n", " ")
        cleaned_preview = cleaned[:500].replace("\n", " ")
        print(f"[LatexWriterTool] Failed to parse plan_json: {parse_error}")
        print(f"[LatexWriterTool] Raw plan_json preview (first 500 chars): {raw_preview}")
        print(f"[LatexWriterTool] Cleaned plan_json preview (first 500 chars): {cleaned_preview}")
        return json.dumps({"error": f"invalid plan_json: {parse_error}"})

    frames_data = plan.get("frames") or []
    figures_meta = plan.get("figures") or plan.get("figure_index") or []
    lang_norm = str(lang or "").lower() if isinstance(lang, str) else ""
    lang_norm = str(lang or "").lower() if isinstance(lang, str) else ""
    print(
        f"[LatexWriterTool] Start writing LaTeX: template={template}, output={output}, "
        f"frames={len(frames_data)}, figures_meta={len(figures_meta or [])}"
    )

    # First, collapse extremely sparse text-only frames and drop exact textual
    # duplicates within the same (section, subsection). This avoids one-sentence
    # slides and repeated content across neighboring frames.
    condensed_frames: List[dict] = []
    seen_text_by_section: dict[tuple[str, str | None], set[str]] = {}

    for frame in frames_data:
        if not isinstance(frame, dict):
            continue

        section = _sanitize_mojibake_text(str(frame.get("section") or "").strip(), fallback="内容分析")
        subsection_val = frame.get("subsection")
        subsection = (
            _sanitize_mojibake_text(subsection_val.strip(), fallback="")
            if isinstance(subsection_val, str)
            else subsection_val
        )
        key = (section, subsection)

        paragraphs = frame.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        sanitized_paragraphs: list[str] = []
        for para in paragraphs:
            cleaned_para = _sanitize_mojibake_text(str(para), fallback="")
            if cleaned_para:
                sanitized_paragraphs.append(cleaned_para)
        paragraphs = sanitized_paragraphs

        bullets = frame.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        sanitized_bullets: list[str] = []
        for bullet in bullets:
            cleaned_bullet = _sanitize_mojibake_text(str(bullet), fallback="")
            if cleaned_bullet:
                sanitized_bullets.append(cleaned_bullet)
        bullets = sanitized_bullets

        figure_slugs_val = frame.get("figure_slugs") or []
        if isinstance(figure_slugs_val, str):
            figure_slugs_val = [figure_slugs_val]

        text_body = "\n".join(str(p) for p in paragraphs) + "\n" + "\n".join(str(b) for b in bullets)
        norm_text = " ".join(text_body.split()).lower()

        # Drop exact duplicate pure-text frames within the same section/subsection
        # (we keep duplicates when they also carry figures, since those are
        # intentionally split across slides).
        has_fig = bool(figure_slugs_val)
        if norm_text:
            seen_set = seen_text_by_section.setdefault(key, set())
            if norm_text in seen_set and not has_fig:
                continue
            seen_set.add(norm_text)

        text_len = len(norm_text)
        num_para = len(paragraphs)
        num_bullets = len(bullets)
        # Extremely sparse pure-text frames (single short paragraph, no bullets).
        is_sparse_text_only = (
            not has_fig
            and norm_text
            and text_len < 80
            and num_para <= 1
            and num_bullets == 0
        )
        # Moderately short pure-text frames (few paragraphs/bullets, short overall
        # length). These can also be merged into the previous frame to avoid
        # single-sentence or very light slides.
        is_moderately_sparse = (
            not has_fig
            and norm_text
            and text_len < 160
            and (num_para + num_bullets) <= 3
        )

        if (is_sparse_text_only or is_moderately_sparse) and condensed_frames:
            prev = condensed_frames[-1]
            prev_section = (prev.get("section") or "").strip()
            prev_sub_val = prev.get("subsection")
            prev_sub = prev_sub_val.strip() if isinstance(prev_sub_val, str) else prev_sub_val
            prev_figs = prev.get("figure_slugs") or []
            if isinstance(prev_figs, str):
                prev_figs = [prev_figs]

            # Only merge into previous if it is from the same section/subsection
            # and also text-only. Additionally, avoid creating an overlong slide
            # by capping the combined text length.
            if prev_section == section and prev_sub == subsection and not prev_figs:
                prev_paras = prev.get("paragraphs") or []
                if isinstance(prev_paras, str):
                    prev_paras = [prev_paras]
                prev_bullets = prev.get("bullets") or []
                if isinstance(prev_bullets, str):
                    prev_bullets = [prev_bullets]

                prev_text_body = "\n".join(str(p) for p in prev_paras) + "\n" + "\n".join(
                    str(b) for b in prev_bullets
                )
                prev_norm_text = " ".join(prev_text_body.split()).lower()
                combined_len = len(prev_norm_text) + text_len

                if combined_len <= 320:
                    prev_paras.extend(paragraphs)
                    prev_bullets.extend(bullets)
                    prev["paragraphs"] = prev_paras
                    prev["bullets"] = prev_bullets
                    continue

        frame["paragraphs"] = paragraphs
        frame["bullets"] = bullets
        frame["figure_slugs"] = figure_slugs_val
        condensed_frames.append(frame)

    frames_data = condensed_frames

    # Normalize figure_slugs into lists without forcing single-figure slides.
    for frame in frames_data:
        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        frame["figure_slugs"] = [s for s in slugs if s]

    figures_by_slug = {}
    for fig in figures_meta or []:
        slug = fig.get("slug")
        path_str = fig.get("path")
        if not slug and path_str:
            slug = Path(path_str).stem
        if not slug:
            continue
        figures_by_slug[slug] = fig

    def _rel_path_for_tex(path_str: str) -> str:
        p = Path(path_str)
        try:
            # Prefer a normal relative path from the generated .tex file so that
            # figures still resolve when the .tex file lives under artifacts/.
            rel = Path(os.path.relpath(p, start=output.parent))
        except Exception:
            rel = p
        return str(rel).replace("\\", "/")

    def _get_image_size(path_str: str) -> tuple[Optional[int], Optional[int]]:
        """Return (width, height) in pixels for a local image file, if possible.

        If Pillow is unavailable or the image cannot be opened, return (None, None)
        so that callers can gracefully fall back to default LaTeX scalings.
        """

        if Image is None:
            return None, None
        try:
            with Image.open(path_str) as img:  # type: ignore[call-arg]
                return img.width, img.height
        except Exception:  # noqa: BLE001
            return None, None

    def _choose_figure_size_for_tex(
        path_str: str,
        num_figs: int,
        kind: Optional[str] = None,
    ) -> tuple[str, str]:
        """Choose LaTeX width/height specifiers based on image size and layout.

        We start from conservative base fractions that depend on how many figures
        share the same frame, then gently shrink very large images or avoid
        over-enlarging tiny ones. Tables are treated slightly differently so that
        they do not occupy excessive vertical space.
        """

        # Base fractions depending on how many figures share this frame.
        if num_figs <= 1:
            base_w = 0.7
            base_h = 0.6
        elif num_figs == 2:
            base_w = 0.65
            base_h = 0.42
        else:
            base_w = 0.6
            base_h = 0.3

        kind_lower = (kind or "").lower()
        if kind_lower.startswith("table"):
            # Tables are often wide and dense; keep them a bit smaller vertically
            # so that headers and surrounding text still have room.
            base_w = min(base_w + 0.05, 0.85)
            base_h = min(base_h, 0.5)

        width_px, height_px = _get_image_size(path_str)
        if not width_px or not height_px:
            return f"{base_w:.2f}\\textwidth", f"{base_h:.2f}\\textheight"

        max_dim = max(width_px, height_px)
        aspect = height_px / width_px if width_px else 1.0

        scale = 1.0

        # Very large source images: shrink slightly so they don't dominate.
        if max_dim >= 3500:
            scale *= 0.7
        elif max_dim >= 2500:
            scale *= 0.82
        elif max_dim <= 800:
            # Avoid over-enlarging very small icons or low-res images.
            scale *= 0.8

        # Adjust for extreme aspect ratios.
        if aspect > 1.3:
            # Tall images: restrict height somewhat more.
            base_h = min(base_h, 0.5 if num_figs <= 1 else 0.4)
        elif aspect < 0.7:
            # Very wide images: keep width a bit smaller to leave side margins.
            base_w = min(base_w, 0.8)

        width_frac = min(base_w * scale, 0.9)
        height_frac = min(base_h * scale, 0.8)

        return f"{width_frac:.2f}\\textwidth", f"{height_frac:.2f}\\textheight"

    def _truncate_caption(caption: str, max_len: int = 80) -> str:
        caption = " ".join((caption or "").split())
        if len(caption) <= max_len:
            return caption
        cut = max_len - 3 if max_len > 3 else max_len
        return caption[:cut].rstrip() + "..."

    def _render_figures_block(figure_slugs: list[str], has_text: bool) -> list[str]:
        fig_items: list[dict] = []
        for slug in figure_slugs:
            fig = figures_by_slug.get(slug)
            if not fig:
                continue
            fig_path = fig.get("path")
            if not fig_path:
                continue
            if not Path(fig_path).expanduser().resolve().exists():
                print(f"[LatexWriterTool] Warning: figure file not found: {fig_path}")
                continue
            fig_items.append(fig)

        if not fig_items:
            return []

        if len(fig_items) > 4:
            print(
                f"[LatexWriterTool] Too many figures for one frame ({len(fig_items)}); "
                "showing the first 4."
            )
            fig_items = fig_items[:4]

        num_figs = len(fig_items)
        lines: list[str] = []

        if num_figs == 1:
            fig = fig_items[0]
            fig_path = fig.get("path")
            tex_path = _rel_path_for_tex(str(fig_path))
            lines.append("\\begin{figure}[h]")
            lines.append("  \\centering")
            fig_kind = fig.get("kind")
            width_spec, height_spec = _choose_figure_size_for_tex(str(fig_path), num_figs, kind=fig_kind)
            lines.append(
                f"  \\includegraphics[width={width_spec},height={height_spec},keepaspectratio]{{{tex_path}}}"
            )
            caption_val = fig.get("caption")
            if caption_val:
                cleaned_caption = _sanitize_mojibake_text(str(caption_val), fallback="")
                if cleaned_caption:
                    safe_caption = _escape_latex_value(cleaned_caption)
                    lines.append(f"  \\caption{{{safe_caption}}}")
            lines.append("\\end{figure}")
            return lines

        # Multi-image grid layout (up to 4 images).
        rows: list[list[dict]] = []
        for i in range(0, num_figs, 2):
            rows.append(fig_items[i : i + 2])
        rows = rows[:2]

        if len(rows) == 1:
            height_spec = "0.45\\textheight" if has_text else "0.55\\textheight"
        else:
            height_spec = "0.30\\textheight" if has_text else "0.38\\textheight"

        lines.append("\\begin{figure}[h]")
        lines.append("  \\centering")
        for row_idx, row in enumerate(rows):
            for col_idx, fig in enumerate(row):
                fig_path = fig.get("path")
                tex_path = _rel_path_for_tex(str(fig_path))
                width_spec = "0.48\\textwidth" if len(row) > 1 else "0.7\\textwidth"
                lines.append(f"  \\begin{{minipage}}[b]{{{width_spec}}}")
                lines.append("    \\centering")
                lines.append(
                    f"    \\includegraphics[width=\\linewidth,height={height_spec},keepaspectratio]{{{tex_path}}}"
                )
                caption_val = fig.get("caption")
                if caption_val:
                    cleaned_caption = _sanitize_mojibake_text(str(caption_val), fallback="")
                    if cleaned_caption:
                        safe_caption = _escape_latex_value(_truncate_caption(cleaned_caption))
                        lines.append(f"    \\\\{{\\scriptsize {safe_caption}}}")
                lines.append("  \\end{minipage}")
                if col_idx < len(row) - 1:
                    lines.append("  \\hfill")
            if row_idx < len(rows) - 1:
                lines.append("  \\\\")
                lines.append("  \\vspace{0.4em}")
        lines.append("\\end{figure}")
        return lines

    frames_tex_parts: List[str] = []

    # Title page and table of contents
    frames_tex_parts.append("\\begin{frame}\n  \\titlepage\n\\end{frame}")
    toc_title = "目录" if str(lang or "").lower().startswith("zh") else "Contents"
    frames_tex_parts.append(f"\\begin{{frame}}{{{_escape_latex_value(toc_title)}}}\n  \\tableofcontents\n\\end{{frame}}")

    current_section: Optional[str] = None
    current_subsection: Optional[str] = None

    for frame in frames_data:
        section = frame.get("section")
        subsection = frame.get("subsection")
        if section:
            if section != current_section:
                frames_tex_parts.append(f"\\section{{{_escape_latex_value(str(section))}}}")
                current_section = section
                current_subsection = None
        if subsection:
            if subsection != current_subsection:
                frames_tex_parts.append(f"\\subsection{{{_escape_latex_value(str(subsection))}}}")
                current_subsection = subsection

        frame_title_raw = frame.get("frame_title") or section or ""
        frame_title = _sanitize_mojibake_text(str(frame_title_raw), fallback="内容分析" if str(lang or "").lower().startswith("zh") else "Analysis")
        bullets = frame.get("bullets") or []
        paragraphs = frame.get("paragraphs") or []
        figure_slugs = frame.get("figure_slugs") or []
        if isinstance(figure_slugs, str):
            figure_slugs = [figure_slugs]

        # Escape the frame title so that special characters and braces do not
        # break the surrounding LaTeX command. Bullets and paragraphs remain
        # raw LaTeX to allow inline math. Render the frame content in a slightly
        # smaller font to help keep dense technical text within the visual
        # bounds of the slide.
        safe_frame_title = _escape_latex_value(str(frame_title)) if frame_title else ""
        frame_lines = [f"\\begin{{frame}}{{{safe_frame_title}}}", "\\small"]

        # Optional natural paragraphs to describe content in a more narrative
        # style instead of only bullet points.
        if paragraphs:
            for para in paragraphs:
                if para:
                    sanitized_para = _sanitize_inline_latex_text(str(para))
                    frame_lines.append(sanitized_para)
                    frame_lines.append("")

        if bullets:
            frame_lines.append("\\begin{itemize}")
            for bullet in bullets:
                sanitized_bullet = _sanitize_inline_latex_text(str(bullet))
                frame_lines.append(f"  \\item {sanitized_bullet}")
            frame_lines.append("\\end{itemize}")

        has_text = bool(paragraphs or bullets)
        figure_lines = _render_figures_block(figure_slugs, has_text)
        if figure_lines:
            frame_lines.extend(figure_lines)

        frame_lines.append("\\end{frame}")
        frames_tex_parts.append("\n".join(frame_lines))

    # Append a closing "thank you" slide at the end of the deck.
    lang_norm = (lang or "").lower() if isinstance(lang, str) else ""
    if lang_norm.startswith("zh"):
        closing_title = "谢谢观看！"
        closing_body = "谢谢观看！"
    else:
        closing_title = "Thank you!"
        closing_body = "Thank you!"
    closing_frame_lines = [
        f"\\begin{{frame}}{{{_escape_latex_value(closing_title)}}}",
        "\\centering",
        "\\LARGE " + _escape_latex_value(closing_body),
        "\\end{frame}",
    ]
    frames_tex_parts.append("\n".join(closing_frame_lines))

    frames_tex = "\n\n".join(frames_tex_parts)

    # Rebuild document body: keep preamble, replace everything from \begin{document} onwards
    begin_match = re.search(r"\\begin{document}", content)
    if begin_match:
        preamble = content[: begin_match.end()]
    else:
        preamble = content + "\n\\begin{document}"  # fallback, should not normally happen

    new_content = preamble + "\n\n" + frames_tex + "\n\n\\end{document}\n"
    output.write_text(new_content, encoding="utf-8")
    print(f"[LatexWriterTool] Finished writing LaTeX to: {output}")
    return json.dumps({"output_path": str(output)})


class PptxWriterInput(BaseModel):
    output_path: str
    plan_json: str
    aspect_ratio: Optional[str] = None
    title: Optional[str] = None
    subtitle: Optional[str] = None
    author: Optional[str] = None
    institute: Optional[str] = None
    date: Optional[str] = None
    lang: Optional[str] = None
    pptx_template: Optional[str] = None
    theme_palette: Optional[str] = None


def _get_pptx_background_config(output_path: Path) -> tuple[str, Optional[Path], Optional[str]]:
    mode = (os.getenv("PPTX_BACKGROUND_MODE") or "inherit").strip().lower()
    if mode not in {"inherit", "upload", "ai", "none"}:
        mode = "inherit"
    image_path_raw = (os.getenv("PPTX_BACKGROUND_IMAGE") or "").strip()
    prompt = (os.getenv("PPTX_BACKGROUND_PROMPT") or "").strip()

    image_path = None
    if image_path_raw:
        try:
            image_path = Path(image_path_raw).expanduser().resolve()
        except Exception:
            image_path = None

    if mode == "ai" and not image_path and prompt:
        image_path = _generate_background_image(prompt, output_path)

    if mode == "none":
        return mode, None, None
    return mode, image_path, prompt


def _extract_error_status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    msg = str(exc)
    patterns = [
        r"\bstatus code\b\s*[:=]\s*(\d{3})",
        r"\berror code\b\s*[:=]\s*(\d{3})",
        r"\bHTTP\s*(\d{3})\b",
    ]
    for pat in patterns:
        m = re.search(pat, msg, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            continue
    return None


def _is_permanent_image_generation_error(exc: Exception) -> bool:
    code = _extract_error_status_code(exc)
    if code in {400, 401, 403, 404, 405}:
        return True
    msg = str(exc).lower()
    permanent_markers = [
        "model not found",
        "unsupported model",
        "not supported",
        "no longer supported",
        "does not support image",
        "invalid api key",
        "incorrect api key",
    ]
    return any(marker in msg for marker in permanent_markers)


def _generate_background_image(
    prompt: str,
    output_path: Path,
    filename: Optional[str] = None,
) -> Optional[Path]:
    global _IMAGE_GEN_DISABLED_KEYS
    global _IMAGE_GEN_DISABLED_LOGGED

    try:
        from PIL import Image  # type: ignore[import]
        from io import BytesIO
    except Exception:
        print("[PptxComWriterTool] Missing image generation dependencies (openai/Pillow).")
        return None

    try:
        import requests  # type: ignore[import]
    except Exception:
        requests = None  # type: ignore[assignment]
    try:
        from urllib import request as urllib_request
    except Exception:
        urllib_request = None  # type: ignore[assignment]

    api_key = (
        os.getenv("PPTX_IMAGE_API_KEY")
        or os.getenv("IMAGE_API_KEY")
        or os.getenv("image_api_key")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("APIYI_API_KEY")
    )
    raw_base_url = (
        os.getenv("PPTX_IMAGE_BASE_URL")
        or os.getenv("IMAGE_BASE_URL")
        or os.getenv("image_base_url")
        or os.getenv("OPENAI_IMAGE_BASE_URL")
        or "https://api.apiyi.com/v1"
    )
    base_url = (raw_base_url or "").strip()
    if not base_url:
        base_url = "https://api.apiyi.com/v1"
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    if "api.apiyi.com" in base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    # DeepSeek's OpenAI-compatible endpoint does not serve image generation.
    # When users share a single APIYI_BASE_URL for chat, force image calls to
    # the dedicated image gateway unless IMAGE_BASE_URL is explicitly set.
    if "deepseek.com" in base_url.lower():
        print(
            "[PptxComWriterTool] IMAGE base URL points to DeepSeek; "
            "switching image generation endpoint to https://api.apiyi.com/v1."
        )
        base_url = "https://api.apiyi.com/v1"
    model = os.getenv("IMAGE_MODEL") or "gpt-image-1"
    size = os.getenv("IMAGE_SIZE") or "1536x1024"
    quality = os.getenv("IMAGE_QUALITY") or "low"
    output_format = (os.getenv("IMAGE_OUTPUT_FORMAT") or "png").strip().lower()
    disable_key = f"{base_url}|{model}"
    disabled_reason = _IMAGE_GEN_DISABLED_KEYS.get(disable_key)
    if disabled_reason is not None:
        if disable_key not in _IMAGE_GEN_DISABLED_LOGGED:
            print(
                f"[PptxComWriterTool] Skipping AI image generation for this run because "
                f"the endpoint/model already returned a permanent error ({disabled_reason})."
            )
            _IMAGE_GEN_DISABLED_LOGGED.add(disable_key)
        return None

    image_timeout_raw = (os.getenv("IMAGE_API_TIMEOUT") or "90").strip()
    try:
        image_timeout = float(image_timeout_raw)
    except ValueError:
        image_timeout = 90.0
    if image_timeout <= 0:
        image_timeout = 90.0

    if not api_key:
        print(
            "[PptxComWriterTool] No image API key configured; "
            "set PPTX_IMAGE_API_KEY / IMAGE_API_KEY / image_api_key (or OPENAI_API_KEY)."
        )
        return None

    try:
        print(
            f"[PptxComWriterTool] Generating background image via {base_url} "
            f"(model={model}, timeout={int(image_timeout)}s)."
        )
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=image_timeout)
        payload_candidates = [
            {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": size,
                "quality": quality,
                "output_format": output_format,
            },
            {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": size,
                "quality": quality,
            },
            {"model": model, "prompt": prompt, "n": 1, "size": size},
            {"model": model, "prompt": prompt, "n": 1},
        ]
        response = None
        last_error: Optional[Exception] = None
        for idx, payload in enumerate(payload_candidates):
            try:
                response = client.images.generate(**payload)
                break
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                msg = str(exc).lower()
                is_param_issue = (
                    "output_format" in msg
                    or "quality" in msg
                    or "unknown parameter" in msg
                    or "unrecognized request argument" in msg
                    or "extra_forbidden" in msg
                )
                if idx < len(payload_candidates) - 1 and is_param_issue:
                    continue
                raise
        if response is None:
            if last_error is not None:
                raise last_error
            return None

        if isinstance(response, dict):
            data = response.get("data") or []
        else:
            data = getattr(response, "data", None) or []
        if not data:
            return None
        first = data[0]
        b64_json = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        if isinstance(first, dict):
            b64_json = b64_json or first.get("b64_json")
            url = url or first.get("url")
        if b64_json:
            img_bytes = base64.b64decode(b64_json)
            img = Image.open(BytesIO(img_bytes))
        elif url:
            if requests is not None:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
            elif urllib_request is not None:
                with urllib_request.urlopen(url, timeout=60) as resp:  # noqa: S310
                    img = Image.open(BytesIO(resp.read()))
            else:
                print(
                    "[PptxComWriterTool] Missing HTTP client dependency for URL-based image download."
                )
                return None
        else:
            return None
        if filename:
            out_path = output_path.parent / filename
        else:
            out_path = output_path.parent / f"pptx_background.{output_format}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_target = out_path
        if save_target.suffix.lower() in {".jpg", ".jpeg"} and img.mode in {"RGBA", "LA", "P"}:
            img = img.convert("RGB")
        try:
            img.save(save_target)
        except Exception:
            # Fallback to PNG when requested format is unsupported in current Pillow build.
            save_target = out_path.with_suffix(".png")
            img.save(save_target)
        out_path = save_target
        print(f"[PptxComWriterTool] Generated background image at: {out_path}")
        return out_path
    except Exception as exc:  # noqa: BLE001
        print(
            f"[PptxComWriterTool] Failed to generate background image "
            f"(base_url={base_url}, model={model}): {exc}"
        )
        if _is_permanent_image_generation_error(exc):
            _IMAGE_GEN_DISABLED_KEYS[disable_key] = str(exc)
            print(
                "[PptxComWriterTool] Detected permanent image API error; "
                "disabling further AI image generation for this endpoint/model in current process."
            )
        if "404" in str(exc):
            print(
                "[PptxComWriterTool] Hint: ensure image calls use "
                "https://api.apiyi.com/v1 and configure IMAGE_API_KEY/PPTX_IMAGE_API_KEY."
            )
        return None


def _get_animation_config() -> tuple[str, str, str]:
    mode = (os.getenv("PPTX_ANIMATION_MODE") or "by_paragraph").strip().lower()
    if mode not in {"none", "enter", "by_paragraph"}:
        mode = "by_paragraph"
    effect = (os.getenv("PPTX_ANIMATION_EFFECT") or "appear").strip().lower()
    if effect not in {"appear", "fade"}:
        effect = "appear"
    image_effect = (os.getenv("PPTX_IMAGE_ANIMATION_EFFECT") or "fly_from_right").strip().lower()
    if image_effect not in {"appear", "fade", "fly_from_right", "fly_from_left", "fly_from_bottom"}:
        image_effect = "fly_from_right"
    return mode, effect, image_effect


def _apply_shape_animation(slide, shape, constants, mode: str, effect: str) -> None:
    try:
        if mode == "none":
            return
        if shape is None:
            return
        effect_id = getattr(constants, "msoAnimEffectAppear", 1)
        if effect == "fade":
            effect_id = getattr(constants, "msoAnimEffectFade", effect_id)
        elif effect == "fly_from_right":
            effect_id = getattr(constants, "msoAnimEffectFlyFromRight", effect_id)
        elif effect == "fly_from_left":
            effect_id = getattr(constants, "msoAnimEffectFlyFromLeft", effect_id)
        elif effect == "fly_from_bottom":
            effect_id = getattr(constants, "msoAnimEffectFlyFromBottom", effect_id)
        seq = slide.TimeLine.MainSequence
        anim = seq.AddEffect(
            Shape=shape,
            effectId=effect_id,
            trigger=getattr(constants, "msoAnimTriggerOnPageClick", 1),
        )
        if mode == "by_paragraph":
            try:
                anim.TextLevelEffect = getattr(constants, "msoAnimTextLevelEffectByParagraph", 1)
            except Exception:
                pass
    except Exception:
        return


def _parse_safe_area_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()
    if cleaned and not cleaned.lstrip().startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def _render_formula_image(formula: str, output_dir: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except Exception:
        return None
    if not formula:
        return None
    text = formula.strip()
    if not (text.startswith("$") and text.endswith("$")):
        text = f"${text}$"
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.axis("off")
        t = ax.text(0, 0, text, fontsize=24, color="black")
        fig.canvas.draw()
        bbox = t.get_window_extent().expanded(1.15, 1.25)
        width, height = bbox.width / fig.dpi, bbox.height / fig.dpi
        plt.close(fig)
        fig = plt.figure(figsize=(max(0.5, width), max(0.3, height)))
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0, 0, text, fontsize=24, color="black")
        out_path = output_dir / f"formula_{uuid.uuid4().hex[:8]}.png"
        fig.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return out_path
    except Exception:
        return None

# Render simple charts from extracted table data (if available).
# This is best-effort and will skip when data is missing.
def _render_chart_from_table(meta: dict, output_dir: Path) -> Optional[Path]:
    if not meta or not isinstance(meta, dict):
        return None
    data = meta.get("table_data") or meta.get("data") or meta.get("rows")
    if not data:
        return None
    title = str(meta.get("caption") or meta.get("title") or "").strip()
    chart_type = str(meta.get("chart") or meta.get("chart_type") or "").strip().lower()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        fig = plt.figure(figsize=(5.8, 3.3), dpi=180)
        ax = fig.add_subplot(111)
        ax.grid(True, alpha=0.2)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            if len(keys) >= 2:
                x_key, y_key = keys[0], keys[1]
                xs = [row.get(x_key) for row in data]
                ys = [row.get(y_key) for row in data]
                if chart_type == "line":
                    ax.plot(xs, ys, marker="o")
                else:
                    ax.bar(xs, ys)
        elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
            # Assume first row is header
            header = list(data[0])
            body = data[1:] if len(data) > 1 else []
            if len(header) >= 2 and body:
                xs = [row[0] for row in body]
                ys = [row[1] for row in body]
                if chart_type == "line":
                    ax.plot(xs, ys, marker="o")
                else:
                    ax.bar(xs, ys)
        else:
            plt.close(fig)
            return None
        if title:
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        out = output_dir / f"chart_{uuid.uuid4().hex}.png"
        fig.savefig(out, transparent=True)
        plt.close(fig)
        return out
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None



@tool(
    "PptxComWriterTool",
    args_schema=PptxWriterInput,
    description=(
        "Render a .pptx PowerPoint file directly from a structured PPT plan using "
        "PowerPoint COM automation (Windows only). Supports templates, backgrounds, "
        "and richer layout fidelity than python-pptx."
    ),
)
def pptx_com_writer_tool(
    output_path: str,
    plan_json: str,
    aspect_ratio: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    author: Optional[str] = None,
    institute: Optional[str] = None,
    date: Optional[str] = None,
    lang: Optional[str] = None,
    pptx_template: Optional[str] = None,
    theme_palette: Optional[str] = None,
) -> str:
    try:
        import win32com.client  # type: ignore[import]
    except Exception:
        return json.dumps({"error": "pywin32 is not installed or COM is unavailable on this system."})
    try:
        import pythoncom  # type: ignore[import]
    except Exception:
        pythoncom = None  # type: ignore[assignment]

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".pptx":
        output = output.with_suffix(".pptx")

    raw = (plan_json or "").strip()
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

    plan = None
    parse_error: Optional[Exception] = None
    for parser_name in ("json", "ast"):
        try:
            if parser_name == "json":
                plan = json.loads(cleaned)
            else:
                plan = ast.literal_eval(cleaned)
            break
        except Exception as exc:  # noqa: BLE001
            parse_error = exc

    if plan is None:
        raw_preview = raw[:500].replace("\n", " ")
        cleaned_preview = cleaned[:500].replace("\n", " ")
        print(f"[PptxComWriterTool] Failed to parse plan_json: {parse_error}")
        print(f"[PptxComWriterTool] Raw plan_json preview (first 500 chars): {raw_preview}")
        print(f"[PptxComWriterTool] Cleaned plan_json preview (first 500 chars): {cleaned_preview}")
        return json.dumps({"error": "Failed to parse plan_json; see logs for details."})

    frames_data = plan.get("frames") or []
    figures_meta = plan.get("figures") or []
    frames_data = frames_data if isinstance(frames_data, list) else []
    figures_meta = figures_meta if isinstance(figures_meta, list) else []

    print(
        f"[PptxComWriterTool] Start writing PPTX via COM: output={output}, frames={len(frames_data)}, "
        f"figures_meta={len(figures_meta)}"
    )

    com_initialized = False
    if pythoncom is not None:
        try:
            pythoncom.CoInitialize()
            com_initialized = True
            print("[PptxComWriterTool] COM apartment initialized for current thread.")
        except Exception as exc:
            print(f"[PptxComWriterTool] Failed to initialize COM apartment: {exc}")

    app = None
    ppt = None
    try:
        try:
            app = win32com.client.DispatchEx("PowerPoint.Application")
            print("[PptxComWriterTool] PowerPoint COM dispatch acquired via DispatchEx.")
        except Exception as exc:
            print(f"[PptxComWriterTool] DispatchEx failed, retrying Dispatch: {exc}")
            app = win32com.client.Dispatch("PowerPoint.Application")
            print("[PptxComWriterTool] PowerPoint COM dispatch acquired via Dispatch.")

        try:
            app.Visible = True if os.getenv("PPTX_COM_VISIBLE", "0").strip() == "1" else False
        except Exception:
            pass
        try:
            app.DisplayAlerts = 0
        except Exception:
            pass

        template_replace_only_mode = False
        template_start_slide = 1
        template_content_base_slide = 1
        template_trim_unused = True

        if pptx_template:
            tpl = Path(pptx_template).expanduser().resolve()
            if tpl.exists():
                template_replace_only_mode = os.getenv("PPTX_TEMPLATE_REPLACE_ONLY", "1").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                try:
                    template_start_slide = max(
                        1,
                        int(os.getenv("PPTX_TEMPLATE_START_SLIDE", "1")),
                    )
                except Exception:
                    template_start_slide = 1
                try:
                    template_content_base_slide = max(
                        1,
                        int(
                            os.getenv(
                                "PPTX_TEMPLATE_CONTENT_BASE_SLIDE",
                                str(template_start_slide),
                            )
                        ),
                    )
                except Exception:
                    template_content_base_slide = template_start_slide
                template_trim_unused = os.getenv("PPTX_TEMPLATE_TRIM_UNUSED", "1").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                print(f"[PptxComWriterTool] Opening template: {tpl}")
                ppt = app.Presentations.Open(str(tpl), WithWindow=False)
                if template_replace_only_mode:
                    print(
                        "[PptxComWriterTool] Template replace-only mode enabled; keeping template slides."
                    )
                else:
                    try:
                        # Remove existing slides while keeping template masters/layouts.
                        for idx in range(ppt.Slides.Count, 0, -1):
                            ppt.Slides(idx).Delete()
                    except Exception:
                        pass
                    print("[PptxComWriterTool] Template opened and existing slides cleared.")
        if ppt is None:
            ppt = app.Presentations.Add()
            print("[PptxComWriterTool] Created new presentation without template.")

        bg_mode, bg_path, _ = _get_pptx_background_config(output)
        anim_mode, anim_effect, image_anim_effect = _get_animation_config()
        safe_margin_left = float(os.getenv("PPTX_SAFE_MARGIN_LEFT") or 50)
        safe_margin_right = float(os.getenv("PPTX_SAFE_MARGIN_RIGHT") or 50)
        safe_margin_top = float(os.getenv("PPTX_SAFE_MARGIN_TOP") or 40)
        safe_margin_bottom = float(os.getenv("PPTX_SAFE_MARGIN_BOTTOM") or 40)
        auto_layout_from_template = os.getenv("PPTX_LAYOUT_FROM_TEMPLATE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        render_latex = os.getenv("PPTX_RENDER_LATEX", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        fill_image_on_empty = os.getenv("PPTX_FILL_IMAGE_ON_EMPTY", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        header_footer = os.getenv("PPTX_HEADER_FOOTER", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        header_logo = os.getenv("PPTX_HEADER_LOGO") or ""
        header_author = os.getenv("PPTX_HEADER_AUTHOR") or ""
        header_date = os.getenv("PPTX_HEADER_DATE") or ""
        header_right = os.getenv("PPTX_HEADER_RIGHT") or ""
        card_layout = os.getenv("PPTX_CARD_LAYOUT", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            from win32com.client import constants  # type: ignore[import]
        except Exception:
            class _Constants:
                msoAnimEffectAppear = 1
                msoAnimEffectFade = 1
                msoAnimEffectFlyFromRight = 1
                msoAnimEffectFlyFromLeft = 1
                msoAnimEffectFlyFromBottom = 1
                msoAnimTriggerOnPageClick = 1
                msoAnimTextLevelEffectByParagraph = 1
            constants = _Constants()

        def _apply_background(slide):
            if bg_mode == "inherit":
                try:
                    slide.FollowMasterBackground = True
                except Exception:
                    pass
                return
            if bg_mode == "none":
                try:
                    slide.FollowMasterBackground = False
                    slide.Background.Fill.Visible = False
                except Exception:
                    pass
                return
            if bg_path and bg_path.exists():
                try:
                    slide.FollowMasterBackground = False
                except Exception:
                    pass
                try:
                    pic = slide.Shapes.AddPicture(
                        FileName=str(bg_path),
                        LinkToFile=False,
                        SaveWithDocument=True,
                        Left=0,
                        Top=0,
                        Width=ppt.PageSetup.SlideWidth,
                        Height=ppt.PageSetup.SlideHeight,
                    )
                    pic.ZOrder(1)  # Send to back
                except Exception:
                    pass

        def _resolve_safe_area() -> tuple[float, float, float, float]:
            left = safe_margin_left
            right = safe_margin_right
            top = safe_margin_top
            bottom = safe_margin_bottom
            if not auto_layout_from_template:
                return left, top, right, bottom
            try:
                layout = ppt.SlideMaster.CustomLayouts(1) if ppt.SlideMaster.CustomLayouts.Count else ppt.SlideMaster.CustomLayouts(1)
                tmp = ppt.Slides.AddSlide(ppt.Slides.Count + 1, layout)
                _apply_background(tmp)
                tmp_path = output.parent / "template_bg_preview.png"
                tmp.Export(str(tmp_path), "PNG", 1280, 720)
                tmp.Delete()
                if tmp_path.exists():
                    prompt = (
                        "请识别这张 PPT 模板背景图中的可用安全内容区域，"
                        "避开顶部 logo/标题区和底部装饰。"
                        "仅返回 JSON，例如 {\"safe_area\": {\"left\":0.1, \"top\":0.15, \"right\":0.95, \"bottom\":0.88}}。"
                        "数值为相对百分比(0-1)。"
                    )
                    desc = analyze_image_with_vision(str(tmp_path), prompt)
                    obj = _parse_safe_area_json(desc)
                    safe = (obj or {}).get("safe_area") if isinstance(obj, dict) else None
                    if isinstance(safe, dict):
                        l = float(safe.get("left", 0.06))
                        t = float(safe.get("top", 0.12))
                        r = float(safe.get("right", 0.94))
                        b = float(safe.get("bottom", 0.9))
                        sw = float(ppt.PageSetup.SlideWidth)
                        sh = float(ppt.PageSetup.SlideHeight)
                        return sw * l, sh * t, sw * (1 - r), sh * (1 - b)
            except Exception:
                pass
            return left, top, right, bottom

        safe_left, safe_top, safe_right, safe_bottom = _resolve_safe_area()
        try:
            sw = float(ppt.PageSetup.SlideWidth)
            sh = float(ppt.PageSetup.SlideHeight)
            safe_left = max(10.0, min(safe_left, sw * 0.4))
            safe_right = max(10.0, min(safe_right, sw * 0.4))
            safe_top = max(10.0, min(safe_top, sh * 0.4))
            safe_bottom = max(10.0, min(safe_bottom, sh * 0.4))
        except Exception:
            pass

        def _estimate_title_font(text: str, max_width: float) -> int:
            if not text:
                return 28
            length = len(text)
            if length <= 18:
                return 32
            if length <= 30:
                return 28
            if length <= 45:
                return 24
            return 20

        def _fit_image_box(img_w: int, img_h: int, box_w: float, box_h: float) -> tuple[float, float]:
            if img_w <= 0 or img_h <= 0 or box_w <= 0 or box_h <= 0:
                return box_w, box_h
            img_ratio = img_w / img_h
            box_ratio = box_w / box_h
            if img_ratio >= box_ratio:
                # Fit to width
                w = box_w
                h = box_w / img_ratio
            else:
                # Fit to height
                h = box_h
                w = box_h * img_ratio
            return w, h

        def _parse_palette(palette: Optional[str]) -> list[tuple[int, int, int]]:
            if not palette:
                return [(52, 152, 219), (52, 219, 203), (24, 72, 120)]
            colors: list[tuple[int, int, int]] = []
            for raw in palette.split(","):
                val = raw.strip().lstrip("#")
                if len(val) != 6:
                    continue
                try:
                    r = int(val[0:2], 16)
                    g = int(val[2:4], 16)
                    b = int(val[4:6], 16)
                except Exception:
                    continue
                colors.append((r, g, b))
            return colors

        def _rgb_to_int(rgb: tuple[int, int, int]) -> int:
            r, g, b = rgb
            return r + (g << 8) + (b << 16)

        def _tint(rgb: tuple[int, int, int], factor: float = 0.85) -> tuple[int, int, int]:
            r, g, b = rgb
            return (
                int(r + (255 - r) * factor),
                int(g + (255 - g) * factor),
                int(b + (255 - b) * factor),
            )

        def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
            return (
                int(a[0] + (b[0] - a[0]) * t),
                int(a[1] + (b[1] - a[1]) * t),
                int(a[2] + (b[2] - a[2]) * t),
            )

        def _luminance(rgb: tuple[int, int, int]) -> float:
            r, g, b = [x / 255.0 for x in rgb]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def _contrast_text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
            return (15, 15, 15) if _luminance(bg) > 0.6 else (245, 245, 245)

        def _build_theme_colors(palette: list[tuple[int, int, int]]) -> dict:
            primary = palette[0] if palette else (52, 152, 219)
            accent = palette[1] if len(palette) > 1 else _blend(primary, (255, 255, 255), 0.2)
            deep = palette[2] if len(palette) > 2 else _blend(primary, (0, 0, 0), 0.35)
            neutral = _blend(deep, (0, 0, 0), 0.35)
            bg_light = _blend(primary, (255, 255, 255), 0.9)
            divider = _blend(primary, (0, 0, 0), 0.25)
            text = _contrast_text_color(bg_light)
            return {
                "primary": primary,
                "accent": accent,
                "deep": deep,
                "neutral": neutral,
                "bg_light": bg_light,
                "divider": divider,
                "text": text,
                "text_on_primary": _contrast_text_color(primary),
            }

        def _clamp_dim(value: float, min_val: float = 50.0) -> float:
            try:
                return max(min_val, float(value))
            except Exception:
                return min_val

        def _get_fonts() -> tuple[str, str, str]:
            title_font = os.getenv("PPTX_FONT_TITLE") or "Microsoft YaHei"
            body_font = os.getenv("PPTX_FONT_BODY") or "Microsoft YaHei"
            note_font = os.getenv("PPTX_FONT_NOTE") or "Microsoft YaHei"
            return title_font, body_font, note_font

        def _preset_config(name: str) -> tuple[list[tuple[int, int, int]], tuple[str, str, str]]:
            preset = (name or "").strip().lower()
            if preset in {"tech", "technology"}:
                palette = [(28, 104, 242), (41, 189, 226), (11, 27, 43), (240, 244, 252), (191, 205, 224)]
                fonts = ("Microsoft YaHei", "Microsoft YaHei", "Microsoft YaHei")
            elif preset in {"academic", "study", "research"}:
                palette = [(29, 64, 120), (64, 118, 219), (24, 43, 77), (241, 244, 249), (190, 202, 222)]
                fonts = ("Times New Roman", "Calibri", "Calibri")
            elif preset in {"business", "corporate"}:
                palette = [(31, 78, 121), (0, 120, 215), (24, 52, 82), (240, 244, 248), (200, 210, 223)]
                fonts = ("Aptos Display", "Aptos", "Aptos")
            elif preset in {"blackgold", "luxury"}:
                palette = [(203, 168, 88), (255, 214, 118), (20, 20, 20), (28, 28, 28), (60, 60, 60)]
                fonts = ("Microsoft YaHei", "Microsoft YaHei", "Microsoft YaHei")
            else:
                palette = []
                fonts = ("", "", "")
            return palette, fonts

        def _get_font_sizes() -> tuple[int, int, int]:
            try:
                title_size = int(os.getenv("PPTX_FONT_TITLE_SIZE") or 32)
            except ValueError:
                title_size = 32
            try:
                body_size = int(os.getenv("PPTX_FONT_BODY_SIZE") or 20)
            except ValueError:
                body_size = 20
            try:
                note_size = int(os.getenv("PPTX_FONT_NOTE_SIZE") or 14)
            except ValueError:
                note_size = 14
            return title_size, body_size, note_size

        def _estimate_lines(text: str, chars_per_line: int) -> int:
            if not text:
                return 0
            total = 0
            for line in text.splitlines():
                ln = max(1, int(len(line) / max(1, chars_per_line)) + 1)
                total += ln
            return total

        def _split_bullets_for_pages(bullets: list[str], chars_per_line: int, max_lines: int) -> list[list[str]]:
            pages: list[list[str]] = []
            current: list[str] = []
            current_lines = 0
            for b in bullets:
                if not b:
                    continue
                lines = max(1, int(len(b) / max(1, chars_per_line)) + 1)
                if current and current_lines + lines > max_lines:
                    pages.append(current)
                    current = []
                    current_lines = 0
                current.append(b)
                current_lines += lines
            if current:
                pages.append(current)
            return pages

        def _split_paragraphs_for_pages(paragraphs: list[str], chars_per_line: int, max_lines: int) -> list[list[str]]:
            pages: list[list[str]] = []
            current: list[str] = []
            current_lines = 0
            for p in paragraphs:
                if not p:
                    continue
                lines = max(1, int(len(p) / max(1, chars_per_line)) + 1)
                if current and current_lines + lines > max_lines:
                    pages.append(current)
                    current = []
                    current_lines = 0
                current.append(p)
                current_lines += lines
            if current:
                pages.append(current)
            return pages

        def _style_bullet_icons(shape, text: str) -> None:
            if shape is None or not text:
                return
            try:
                tr = shape.TextFrame.TextRange
                lines = text.splitlines()
                pos = 1
                for line in lines:
                    ln = len(line)
                    if line.startswith("l "):
                        try:
                            bullet = tr.Characters(pos, 1)
                            bullet.Font.Name = "Wingdings"
                            bullet.Font.Size = tr.Font.Size
                            bullet.Font.Color.RGB = _rgb_to_int(accent)
                            rest = tr.Characters(pos + 1, max(1, ln - 1))
                            rest.Font.Name = body_font
                            rest.Font.Color.RGB = _rgb_to_int(text_color)
                        except Exception:
                            pass
                    pos += ln + 1
            except Exception:
                pass

        def _add_header_footer(slide, section_name: str, page_no: int) -> None:
            if not header_footer or slide is None:
                return
            try:
                sw = float(ppt.PageSetup.SlideWidth)
                sh = float(ppt.PageSetup.SlideHeight)
                left = safe_left
                right = safe_right
                y = sh - safe_bottom + 8
                line = slide.Shapes.AddShape(
                    1,
                    left,
                    y - 6,
                    _clamp_dim(sw - left - right),
                    1.6,
                )
                line.Fill.Solid()
                line.Fill.ForeColor.RGB = _rgb_to_int(divider)
                line.Line.Visible = False
            except Exception:
                pass
            try:
                # Optional logo at left
                if header_logo and Path(header_logo).expanduser().exists():
                    logo = slide.Shapes.AddPicture(
                        FileName=str(Path(header_logo).expanduser().resolve()),
                        LinkToFile=False,
                        SaveWithDocument=True,
                        Left=safe_left,
                        Top=sh - safe_bottom - 2,
                        Width=24,
                        Height=24,
                    )
                left_x = safe_left + (28 if header_logo else 0)
                left_box = slide.Shapes.AddTextbox(
                    1,
                    left_x,
                    sh - safe_bottom,
                    _clamp_dim((sw - safe_left - safe_right) * 0.55),
                    20,
                )
                tr_left = left_box.TextFrame.TextRange
                sec = section_name.strip() if section_name else ""
                author_txt = header_author or (author if author else "")
                date_txt = header_date or (date if date else "")
                left_text = sec
                if author_txt:
                    left_text = f"{left_text}  |  {author_txt}" if left_text else author_txt
                if date_txt:
                    left_text = f"{left_text}  |  {date_txt}" if left_text else date_txt
                tr_left.Text = left_text
                tr_left.Font.Name = note_font
                tr_left.Font.Size = max(10, note_base_size - 2)
                tr_left.Font.Color.RGB = _rgb_to_int(neutral)
                tr_left.ParagraphFormat.Alignment = 1
                right_box = slide.Shapes.AddTextbox(
                    1,
                    sw - safe_right - 140,
                    sh - safe_bottom,
                    140,
                    20,
                )
                tr_right = right_box.TextFrame.TextRange
                right_text = header_right or str(page_no)
                tr_right.Text = right_text
                tr_right.Font.Name = note_font
                tr_right.Font.Size = max(10, note_base_size - 2)
                tr_right.Font.Color.RGB = _rgb_to_int(neutral)
                tr_right.ParagraphFormat.Alignment = 3
            except Exception:
                pass
                return
            try:
                sw = float(ppt.PageSetup.SlideWidth)
                sh = float(ppt.PageSetup.SlideHeight)
                left = safe_left
                right = safe_right
                y = sh - safe_bottom + 8
                line = slide.Shapes.AddShape(
                    1,
                    left,
                    y - 6,
                    _clamp_dim(sw - left - right),
                    1.6,
                )
                line.Fill.Solid()
                line.Fill.ForeColor.RGB = _rgb_to_int(divider)
                line.Line.Visible = False
            except Exception:
                pass
            try:
                sec = section_name.strip() if section_name else ""
                left_box = slide.Shapes.AddTextbox(
                    1,
                    safe_left,
                    sh - safe_bottom,
                    _clamp_dim((sw - safe_left - safe_right) * 0.7),
                    20,
                )
                tr_left = left_box.TextFrame.TextRange
                tr_left.Text = sec
                tr_left.Font.Name = note_font
                tr_left.Font.Size = max(10, note_base_size - 2)
                tr_left.Font.Color.RGB = _rgb_to_int(neutral)
                tr_left.ParagraphFormat.Alignment = 1
                right_box = slide.Shapes.AddTextbox(
                    1,
                    sw - safe_right - 60,
                    sh - safe_bottom,
                    60,
                    20,
                )
                tr_right = right_box.TextFrame.TextRange
                tr_right.Text = str(page_no)
                tr_right.Font.Name = note_font
                tr_right.Font.Size = max(10, note_base_size - 2)
                tr_right.Font.Color.RGB = _rgb_to_int(neutral)
                tr_right.ParagraphFormat.Alignment = 3
            except Exception:
                pass

        def _extract_keywords(bullets: list[str], paragraphs: list[str], max_items: int = 5) -> list[str]:
            candidates: list[str] = []
            for text in (bullets or []) + (paragraphs or []):
                for part in re.split(r"[，。；;\n]+", str(text)):
                    p = part.strip()
                    if len(p) >= 2:
                        candidates.append(p)
            seen: set[str] = set()
            keywords: list[str] = []
            for c in candidates:
                if c in seen:
                    continue
                seen.add(c)
                keywords.append(c)
                if len(keywords) >= max_items:
                    break
            return keywords

        preset_palette, preset_fonts = _preset_config(os.getenv("PPTX_STYLE_PRESET") or "academic")
        title_font, body_font, note_font = _get_fonts()
        if preset_fonts[0] and not os.getenv("PPTX_FONT_TITLE"):
            title_font = preset_fonts[0]
        if preset_fonts[1] and not os.getenv("PPTX_FONT_BODY"):
            body_font = preset_fonts[1]
        if preset_fonts[2] and not os.getenv("PPTX_FONT_NOTE"):
            note_font = preset_fonts[2]
        title_base_size, body_base_size, note_base_size = _get_font_sizes()
        palette = _parse_palette(theme_palette or "")
        if not (theme_palette or "").strip() and preset_palette:
            palette = preset_palette
        theme_colors = _build_theme_colors(palette)
        primary = theme_colors["primary"]
        accent = theme_colors["accent"]
        neutral = theme_colors["neutral"]
        bg_light = theme_colors["bg_light"]
        divider = theme_colors["divider"]
        text_color = neutral if _luminance(bg_light) > 0.5 else theme_colors["text"]
        section_last = None
        fig_meta_by_slug = (
            {str(m.get("slug")): m for m in figures_meta if isinstance(m, dict) and m.get("slug")}
            if figures_meta
            else {}
        )

        def _shape_text(shape) -> str:
            try:
                if not getattr(shape, "HasTextFrame", False):
                    return ""
                if not shape.TextFrame.HasText:
                    return ""
                return str(shape.TextFrame.TextRange.Text or "")
            except Exception:
                return ""

        def _set_shape_text_keep_style(shape, text: str) -> bool:
            try:
                if not getattr(shape, "HasTextFrame", False):
                    return False
                shape.TextFrame.TextRange.Text = text or ""
                try:
                    shape.TextFrame.WordWrap = True
                except Exception:
                    pass
                # Prefer shrinking text-to-fit inside template boxes to avoid overlap.
                try:
                    shape.TextFrame2.AutoSize = 2  # msoAutoSizeTextToFitShape
                except Exception:
                    pass
                return True
            except Exception:
                return False

        def _collect_text_shapes(slide) -> list:
            out = []
            try:
                for s in slide.Shapes:
                    try:
                        if getattr(s, "HasTextFrame", False):
                            out.append(s)
                    except Exception:
                        continue
            except Exception:
                return []
            return out

        def _shape_sort_key(shape) -> tuple[float, float, float]:
            try:
                top = float(getattr(shape, "Top", 0.0))
                left = float(getattr(shape, "Left", 0.0))
                area = float(getattr(shape, "Width", 0.0)) * float(getattr(shape, "Height", 0.0))
            except Exception:
                return (10**9, 10**9, 10**9)
            return (top, left, -area)

        def _body_shape_rank(shape) -> tuple[float, float, float]:
            try:
                area = float(getattr(shape, "Width", 0.0)) * float(getattr(shape, "Height", 0.0))
                top = float(getattr(shape, "Top", 0.0))
                left = float(getattr(shape, "Left", 0.0))
            except Exception:
                return (10**9, 10**9, 10**9)
            return (-area, top, left)

        def _find_title_shape(slide, text_shapes: list):
            try:
                title_shape = slide.Shapes.Title
                if title_shape is not None:
                    return title_shape
            except Exception:
                pass
            ordered = sorted(text_shapes, key=_shape_sort_key)
            return ordered[0] if ordered else None

        def _collect_picture_shapes(slide, slide_width: float, slide_height: float) -> list:
            pics = []
            try:
                for s in slide.Shapes:
                    try:
                        stype = int(getattr(s, "Type", 0))
                    except Exception:
                        stype = 0
                    try:
                        w = float(getattr(s, "Width", 0.0))
                        h = float(getattr(s, "Height", 0.0))
                        left = float(getattr(s, "Left", 0.0))
                        top = float(getattr(s, "Top", 0.0))
                    except Exception:
                        w, h, left, top = 0.0, 0.0, 0.0, 0.0
                    name = str(getattr(s, "Name", "") or "").lower()
                    if w < 60 or h < 60:
                        continue
                    area = w * h
                    # Avoid replacing slide-sized background/decorative images.
                    if slide_width > 0 and slide_height > 0 and area > (slide_width * slide_height * 0.72):
                        continue
                    # Prefer content-area image containers, not header decorations.
                    if slide_height > 0 and top < slide_height * 0.12:
                        continue
                    if slide_width > 0 and left > slide_width * 0.85 and w < slide_width * 0.12:
                        continue
                    if stype in {11, 13} or any(k in name for k in ("image", "img", "picture", "photo", "图片")):
                        pics.append(s)
            except Exception:
                return []
            pics.sort(
                key=lambda sh: (
                    -float(getattr(sh, "Width", 0.0)) * float(getattr(sh, "Height", 0.0)),
                    float(getattr(sh, "Top", 0.0)),
                    float(getattr(sh, "Left", 0.0)),
                )
            )
            return pics

        def _replace_picture_shape(slide, shape, path: str) -> bool:
            try:
                left = float(shape.Left)
                top = float(shape.Top)
                width = float(shape.Width)
                height = float(shape.Height)
            except Exception:
                return False
            try:
                shape.Delete()
            except Exception:
                pass
            try:
                slide.Shapes.AddPicture(
                    FileName=str(path),
                    LinkToFile=False,
                    SaveWithDocument=True,
                    Left=left,
                    Top=top,
                    Width=width,
                    Height=height,
                )
                return True
            except Exception:
                return False

        _template_protected_keywords = {
            "项目概况",
            "市场分析",
            "项目规划",
            "运营管理",
            "项目成果",
            "目录",
            "contents",
            "table of contents",
        }
        _template_placeholder_markers = (
            "点击此处",
            "请输入",
            "占位",
            "示例",
            "样例",
            "xxxx",
            "xxxxx",
            "xxxxxxxx",
            "xxxxxxxxx",
            "文本",
            "内容",
            "小标题",
            "联合创始人",
            "项目经验",
            "研发历程一",
            "观小风",
        )

        def _normalize_inline_text(text: str) -> str:
            return re.sub(r"\s+", " ", str(text or "")).strip()

        def _is_protected_nav_text(text: str) -> bool:
            t = _normalize_inline_text(text)
            if not t:
                return False
            if t in _template_protected_keywords:
                return True
            low = t.lower()
            return low in _template_protected_keywords

        def _looks_like_placeholder_text(text: str) -> bool:
            t = _normalize_inline_text(text)
            if not t:
                return True
            low = t.lower()
            if any(m in low for m in _template_placeholder_markers):
                return True
            if re.search(r"x{4,}", low):
                return True
            # Repeated duplicated text is often a template filler.
            if len(t) >= 4 and t[: len(t) // 2] == t[len(t) // 2 :] and len(t) % 2 == 0:
                return True
            return False

        def _estimate_shape_char_capacity(shape) -> int:
            try:
                area = float(getattr(shape, "Width", 0.0)) * float(getattr(shape, "Height", 0.0))
            except Exception:
                area = 0.0
            if area <= 0:
                return 140
            # Empirical mapping for East-Asian templates.
            cap = int(area / 1800.0)
            return max(40, min(cap, 560))

        def _split_text_to_shape_caps(text: str, caps: list[int]) -> list[str]:
            if not caps:
                return []
            plain = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not plain:
                return ["" for _ in caps]
            lines = [ln.strip() for ln in plain.split("\n") if ln.strip()]
            if not lines:
                lines = [plain]
            chunks: list[str] = []
            idx_line = 0
            for cap in caps:
                if idx_line >= len(lines):
                    chunks.append("")
                    continue
                cur = ""
                while idx_line < len(lines):
                    candidate = lines[idx_line]
                    trial = f"{cur}\r\n{candidate}".strip() if cur else candidate
                    if cur and len(trial) > cap:
                        break
                    if not cur and len(candidate) > cap:
                        cur = candidate[:cap].rstrip() + "..."
                        idx_line += 1
                        break
                    cur = trial
                    idx_line += 1
                chunks.append(cur)
            if idx_line < len(lines) and chunks:
                tail = "\r\n".join(lines[idx_line:])
                last = chunks[-1]
                merged = f"{last}\r\n{tail}".strip() if last else tail
                max_last = max(60, caps[-1])
                if len(merged) > max_last:
                    merged = merged[:max_last].rstrip() + "..."
                chunks[-1] = merged
            return chunks

        def _select_text_slots(slide, max_body_slots: int):
            try:
                sw = float(getattr(ppt.PageSetup, "SlideWidth", 0.0))
                sh = float(getattr(ppt.PageSetup, "SlideHeight", 0.0))
            except Exception:
                sw, sh = 0.0, 0.0
            text_shapes = _collect_text_shapes(slide)
            title_shape = None
            nav_shapes: list = []
            replace_candidates: list = []

            for s in text_shapes:
                txt = _normalize_inline_text(_shape_text(s))
                try:
                    top = float(getattr(s, "Top", 0.0))
                    left = float(getattr(s, "Left", 0.0))
                    w = float(getattr(s, "Width", 0.0))
                    h = float(getattr(s, "Height", 0.0))
                    area = w * h
                except Exception:
                    continue
                if area < 1200:
                    continue
                is_top_band = sh > 0 and top < sh * 0.14
                if _is_protected_nav_text(txt) or (is_top_band and len(txt) <= 12 and left > (sw * 0.35 if sw else 1200)):
                    nav_shapes.append(s)
                    continue
                replace_candidates.append((s, txt, top, left, area, w, h))

            # Strong preference for left-top main heading box inside template.
            title_pool = [
                item
                for item in replace_candidates
                if (sh <= 0 or item[2] < sh * 0.2) and (sw <= 0 or item[3] < sw * 0.45)
            ]
            if not title_pool:
                title_pool = replace_candidates
            if title_pool:
                title_shape = sorted(title_pool, key=lambda it: (it[2], it[3], -it[4]))[0][0]
            else:
                title_shape = _find_title_shape(slide, text_shapes)

            title_id = getattr(title_shape, "Id", None) if title_shape is not None else None
            body_pool = []
            for s, txt, top, left, area, w, h in replace_candidates:
                if title_id is not None and getattr(s, "Id", None) == title_id:
                    continue
                is_placeholder = _looks_like_placeholder_text(txt)
                if sh > 0 and top < sh * 0.11 and not is_placeholder:
                    continue
                score = 0
                if is_placeholder:
                    score += 5
                if sh > 0 and top > sh * 0.18:
                    score += 2
                if len(txt) >= 20:
                    score += 2
                if sw > 0 and w > sw * 0.22:
                    score += 1
                if not is_placeholder and score < 3:
                    continue
                body_pool.append((s, txt, top, left, area, w, h, score))
            body_pool.sort(key=lambda it: (-it[7], it[2], it[3], -it[4]))
            body_shapes = [it[0] for it in body_pool[:max_body_slots]]

            return title_shape, body_shapes, nav_shapes

        def _match_template_token(text: str) -> tuple[str, int] | None:
            t = _normalize_inline_text(text)
            if not t:
                return None
            up = t.upper()
            zh = t
            if "{{TITLE}}" in up or "【标题】" in zh:
                return ("title", 0)
            if "{{SUBTITLE}}" in up or "{{SECTION}}" in up or "【副标题】" in zh or "【章节】" in zh:
                return ("subtitle", 0)
            m_en = re.search(r"\{\{\s*BODY\s*([1-9]?)\s*\}\}", up)
            if m_en:
                idx_raw = m_en.group(1) or "1"
                try:
                    idx = max(1, min(9, int(idx_raw)))
                except Exception:
                    idx = 1
                return ("body", idx)
            m_zh = re.search(r"【\s*正文\s*([1-9]?)\s*】", zh)
            if m_zh:
                idx_raw = m_zh.group(1) or "1"
                try:
                    idx = max(1, min(9, int(idx_raw)))
                except Exception:
                    idx = 1
                return ("body", idx)
            return None

        if template_replace_only_mode and pptx_template and ppt is not None and getattr(ppt.Slides, "Count", 0) > 0:
            try:
                template_slide_count = int(ppt.Slides.Count)
            except Exception:
                template_slide_count = 0
            if template_slide_count > 0:
                try:
                    body_max_chars = max(
                        120,
                        int(os.getenv("PPTX_TEMPLATE_BODY_MAX_CHARS", "640")),
                    )
                except Exception:
                    body_max_chars = 640
                try:
                    max_body_slots = max(
                        1,
                        int(os.getenv("PPTX_TEMPLATE_BODY_SLOTS", "3")),
                    )
                except Exception:
                    max_body_slots = 3
                clear_unfilled_slots = os.getenv("PPTX_TEMPLATE_CLEAR_UNFILLED", "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                start_idx = max(1, min(template_start_slide, template_slide_count))
                base_idx = max(1, min(template_content_base_slide, template_slide_count))
                print(
                    "[PptxComWriterTool] Replace-only fill: "
                    f"start_slide={start_idx}, base_slide={base_idx}, frames={len(frames_data)}"
                )

                def _get_or_clone_slide(target_index: int):
                    try:
                        current_count = int(ppt.Slides.Count)
                    except Exception:
                        current_count = 0
                    if target_index <= current_count and target_index >= 1:
                        return ppt.Slides(target_index)
                    try:
                        dup = ppt.Slides(base_idx).Duplicate()
                        try:
                            slide = dup.Item(1)
                        except Exception:
                            slide = ppt.Slides(ppt.Slides.Count)
                        try:
                            slide.MoveTo(ppt.Slides.Count)
                        except Exception:
                            pass
                        return slide
                    except Exception:
                        return ppt.Slides(ppt.Slides.Count)

                for frame_idx, frame in enumerate(frames_data, start=1):
                    if not isinstance(frame, dict):
                        continue
                    target_slide = _get_or_clone_slide(start_idx + frame_idx - 1)
                    section = (frame.get("section") or "").strip()
                    subsection = (frame.get("subsection") or "").strip()
                    title_text = (frame.get("frame_title") or "").strip()
                    bullets = frame.get("bullets") or []
                    paragraphs = frame.get("paragraphs") or []
                    fig_slugs = frame.get("figure_slugs") or []
                    if isinstance(bullets, str):
                        bullets = [bullets]
                    if isinstance(paragraphs, str):
                        paragraphs = [paragraphs]
                    if isinstance(fig_slugs, str):
                        fig_slugs = [fig_slugs]

                    title_final = title_text or section or f"Slide {frame_idx}"
                    subtitle_line = section if not subsection else f"{section} - {subsection}"
                    para_lines = [str(p).strip() for p in paragraphs if str(p).strip()]
                    bullet_lines = [str(b).strip() for b in bullets if str(b).strip()]

                    token_title_shape = None
                    token_subtitle_shape = None
                    token_body_shapes: list[tuple[int, object]] = []
                    for s in _collect_text_shapes(target_slide):
                        token = _match_template_token(_shape_text(s))
                        if token is None:
                            continue
                        kind, idx = token
                        if kind == "title":
                            token_title_shape = s
                        elif kind == "subtitle":
                            token_subtitle_shape = s
                        elif kind == "body":
                            token_body_shapes.append((idx, s))
                    token_body_shapes.sort(key=lambda it: it[0])

                    if token_title_shape is not None:
                        title_shape = token_title_shape
                        body_shapes = [s for _, s in token_body_shapes][:max_body_slots]
                        if token_subtitle_shape is not None and len(body_shapes) < max_body_slots:
                            body_shapes = [token_subtitle_shape] + body_shapes
                    else:
                        title_shape, body_shapes, _nav_shapes = _select_text_slots(
                            target_slide,
                            max_body_slots=max_body_slots,
                        )

                    if title_shape is not None:
                        _set_shape_text_keep_style(title_shape, title_final)

                    body_payload_source = ""
                    if para_lines:
                        body_payload_source = "\r\n".join(para_lines)
                    elif bullet_lines:
                        body_payload_source = "\r\n".join([f"• {b}" for b in bullet_lines])
                    if subtitle_line:
                        body_payload_source = (
                            f"{subtitle_line}\r\n{body_payload_source}".strip()
                            if body_payload_source
                            else subtitle_line
                        )
                    if len(body_payload_source) > body_max_chars:
                        body_payload_source = body_payload_source[:body_max_chars].rstrip() + "..."
                    caps = [_estimate_shape_char_capacity(s) for s in body_shapes]
                    body_payloads = _split_text_to_shape_caps(body_payload_source, caps)

                    for idx_shape, shape in enumerate(body_shapes):
                        if idx_shape < len(body_payloads):
                            _set_shape_text_keep_style(shape, body_payloads[idx_shape])
                        elif clear_unfilled_slots and _shape_text(shape).strip():
                            _set_shape_text_keep_style(shape, "")

                    fig_paths = []
                    for slug in fig_slugs:
                        meta = fig_meta_by_slug.get(str(slug))
                        if isinstance(meta, dict) and meta.get("path"):
                            fig_paths.append(str(meta.get("path")))
                    if fig_paths:
                        try:
                            sw = float(getattr(ppt.PageSetup, "SlideWidth", 0.0))
                            sh = float(getattr(ppt.PageSetup, "SlideHeight", 0.0))
                        except Exception:
                            sw, sh = 0.0, 0.0
                        pic_shapes = _collect_picture_shapes(target_slide, sw, sh)
                        for i, path in enumerate(fig_paths):
                            if i >= len(pic_shapes):
                                break
                            _replace_picture_shape(target_slide, pic_shapes[i], path)

                if template_trim_unused and frames_data:
                    last_keep = start_idx + len(frames_data) - 1
                    try:
                        for idx in range(int(ppt.Slides.Count), last_keep, -1):
                            if idx >= 1:
                                ppt.Slides(idx).Delete()
                    except Exception:
                        pass

                print(f"[PptxComWriterTool] Saving presentation to: {output}")
                ppt.SaveAs(str(output))
                print(f"[PptxComWriterTool] Finished writing PPTX to: {output}")
                return json.dumps({"output_path": str(output)})

        def _add_section_cover(section_name: str, keywords: Optional[list[str]] = None) -> None:
            if not section_name:
                return
            layout = ppt.SlideMaster.CustomLayouts(1) if ppt.SlideMaster.CustomLayouts.Count else ppt.SlideMaster.CustomLayouts(1)
            slide = ppt.Slides.AddSlide(ppt.Slides.Count + 1, layout)
            _apply_background(slide)
            sw = float(ppt.PageSetup.SlideWidth)
            sh = float(ppt.PageSetup.SlideHeight)
            mx = safe_left
            try:
                overlay = slide.Shapes.AddShape(
                    1,
                    0,
                    sh * 0.25,
                    sw,
                    sh * 0.5,
                )
                overlay.Fill.Solid()
                overlay.Fill.ForeColor.RGB = _rgb_to_int(bg_light)
                overlay.Fill.Transparency = 0.15
                overlay.Line.Visible = False
            except Exception:
                pass
            try:
                bar = slide.Shapes.AddShape(
                    1,
                    mx,
                    sh * 0.33,
                    8,
                    sh * 0.26,
                )
                bar.Fill.Solid()
                bar.Fill.ForeColor.RGB = _rgb_to_int(primary)
                bar.Line.Visible = False
                _apply_shape_animation(slide, bar, constants, "enter", anim_effect)
            except Exception:
                pass
            try:
                box = slide.Shapes.AddTextbox(
                    1,
                    mx + 14,
                    sh * 0.35,
                    _clamp_dim(sw - mx - safe_right),
                    120,
                )
                tr = box.TextFrame.TextRange
                tr.Text = section_name
                tr.Font.Name = title_font
                tr.Font.Size = max(36, title_base_size + 6)
                tr.Font.Bold = True
                tr.Font.Color.RGB = _rgb_to_int(primary)
                _apply_shape_animation(slide, box, constants, "enter", anim_effect)
            except Exception:
                pass
            try:
                line = slide.Shapes.AddShape(
                    1,
                    mx + 14,
                    sh * 0.35 + 90,
                    _clamp_dim(sw - mx - safe_right),
                    2,
                )
                line.Fill.Solid()
                line.Fill.ForeColor.RGB = _rgb_to_int(divider)
                line.Line.Visible = False
            except Exception:
                pass
            if keywords:
                try:
                    start_x = mx + 14
                    y = sh * 0.35 + 104
                    pill_h = 28
                    gap = 8
                    for word in keywords[:5]:
                        w = max(60, len(word) * (note_base_size * 0.7) + 24)
                        if start_x + w > sw - safe_right:
                            break
                        pill = slide.Shapes.AddShape(
                            1,
                            start_x,
                            y,
                            w,
                            pill_h,
                        )
                        pill.Fill.Solid()
                        pill.Fill.ForeColor.RGB = _rgb_to_int(_tint(accent, 0.85))
                        pill.Line.Visible = True
                        pill.Line.ForeColor.RGB = _rgb_to_int(divider)
                        tr = pill.TextFrame.TextRange
                        tr.Text = word
                        tr.Font.Name = note_font
                        tr.Font.Size = note_base_size
                        tr.Font.Color.RGB = _rgb_to_int(_contrast_text_color(_tint(accent, 0.85)))
                        _apply_shape_animation(slide, pill, constants, "enter", anim_effect)
                        start_x += w + gap
                except Exception:
                    pass

        # Create slides
        empty_text_streak = 0
        ai_image_count = 0
        ai_image_failure_count = 0
        try:
            max_ai_images = int(os.getenv("PPTX_MAX_AI_IMAGES") or 6)
        except ValueError:
            max_ai_images = 6
        try:
            max_ai_failures = int(os.getenv("PPTX_MAX_AI_IMAGE_FAILURES") or 2)
        except ValueError:
            max_ai_failures = 2
        max_ai_failures = max(1, max_ai_failures)
        ai_image_enabled = max_ai_images > 0
        page_no = 0
        total_frames = len(frames_data)
        fig_meta_by_slug = (
            {str(m.get("slug")): m for m in figures_meta if isinstance(m, dict) and m.get("slug")}
            if figures_meta
            else {}
        )
        for frame_idx, frame in enumerate(frames_data, start=1):
            if frame_idx == 1 or frame_idx % 5 == 0 or frame_idx == total_frames:
                print(
                    f"[PptxComWriterTool] Slide compose progress: {frame_idx}/{total_frames} frames..."
                )
            if not isinstance(frame, dict):
                continue
            section = (frame.get("section") or "").strip()
            subsection = (frame.get("subsection") or "").strip()
            title_text = (frame.get("frame_title") or "").strip()
            bullets = frame.get("bullets") or []
            paragraphs = frame.get("paragraphs") or []
            fig_slugs = frame.get("figure_slugs") or []

            if section and section != section_last:
                keywords = _extract_keywords(bullets, paragraphs, max_items=5)
                _add_section_cover(section, keywords=keywords)
                section_last = section

            # Build pagination for text
            fig_paths = []
            fig_info = []
            if fig_slugs:
                for slug in fig_slugs:
                    meta = fig_meta_by_slug.get(str(slug))
                    if meta and meta.get("path"):
                        fig_paths.append(meta.get("path"))
                        fig_info.append((meta.get("path"), meta.get("caption") or "", meta))
            # Try to replace table figures with rendered charts when data is available.
            chart_map = {}
            if fig_info:
                for p, cap, meta in fig_info:
                    if not isinstance(meta, dict):
                        continue
                    chart = _render_chart_from_table(meta, output.parent)
                    if chart:
                        chart_map[str(p)] = str(chart)

            layout = ppt.SlideMaster.CustomLayouts(1) if ppt.SlideMaster.CustomLayouts.Count else ppt.SlideMaster.CustomLayouts(1)
            slide_width = float(ppt.PageSetup.SlideWidth)
            slide_height = float(ppt.PageSetup.SlideHeight)
            margin_x = safe_left
            margin_top = safe_top
            margin_bottom = safe_bottom
            gap_y = 18

            # Prepare per-page text chunks
            rough_text = " ".join(str(x) for x in (paragraphs or []) + (bullets or []))
            rough_chars = len(rough_text)
            body_font_size = body_base_size
            if rough_chars > 2600:
                body_font_size -= 7
            elif rough_chars > 1800:
                body_font_size -= 5
            elif rough_chars > 1200:
                body_font_size -= 3
            elif rough_chars > 700:
                body_font_size -= 2
            if len(fig_paths) >= 2:
                body_font_size -= 1
            if len(fig_paths) >= 4:
                body_font_size -= 1
            body_font_size = max(12, min(body_font_size, body_base_size + 2))
            char_factor = 0.56
            if isinstance(lang, str) and lang.lower().startswith("zh"):
                char_factor = 0.62
            chars_per_line = max(
                14,
                int(_clamp_dim(slide_width - margin_x - safe_right, 220) / (body_font_size * char_factor)),
            )
            text_ratio = 0.58 if not fig_paths else (0.48 if len(fig_paths) == 1 else 0.42)
            max_lines = max(5, int((slide_height * text_ratio) / (body_font_size * 1.16)))
            bullet_pages = _split_bullets_for_pages(bullets, chars_per_line, max_lines) if bullets else [[]]
            para_pages = _split_paragraphs_for_pages(paragraphs, chars_per_line, max_lines) if paragraphs else [[]]
            max_pages = max(len(bullet_pages), len(para_pages))

            for page_idx in range(max_pages):
                page_bullets = bullet_pages[page_idx] if page_idx < len(bullet_pages) else []
                page_paras = para_pages[page_idx] if page_idx < len(para_pages) else []
                show_images = True
                if max_pages > 1:
                    show_images = page_idx == max_pages - 1
                display_title = title_text
                if max_pages > 1 and page_idx > 0:
                    if isinstance(lang, str) and lang.lower().startswith("zh"):
                        display_title = f"{title_text}（续 {page_idx + 1}/{max_pages}）"
                    else:
                        display_title = f"{title_text} (cont. {page_idx + 1}/{max_pages})"
                layout = ppt.SlideMaster.CustomLayouts(1) if ppt.SlideMaster.CustomLayouts.Count else ppt.SlideMaster.CustomLayouts(1)
                slide = ppt.Slides.AddSlide(ppt.Slides.Count + 1, layout)
                _apply_background(slide)
                # Clear template placeholder texts on the slide (keep master visuals).
                try:
                    for s in slide.Shapes:
                        if getattr(s, "HasTextFrame", False):
                            try:
                                if s.TextFrame.HasText:
                                    s.TextFrame.TextRange.Text = ""
                            except Exception:
                                pass
                except Exception:
                    pass

                # Title
                title_shape = None
                try:
                    if slide.Shapes.Title is not None:
                        slide.Shapes.Title.TextFrame.TextRange.Text = display_title
                        title_shape = slide.Shapes.Title
                except Exception:
                    title_shape = None
                if title_shape is None and display_title:
                    try:
                        title_shape = slide.Shapes.AddTextbox(
                            1,
                            margin_x,
                            margin_top,
                            _clamp_dim(slide_width - margin_x - safe_right),
                            60,
                        )
                        title_shape.TextFrame.TextRange.Text = display_title
                    except Exception:
                        title_shape = None

                if title_shape is not None and display_title:
                    try:
                        font_size = _estimate_title_font(display_title, _clamp_dim(slide_width - margin_x - safe_right))
                        title_shape.TextFrame.TextRange.Font.Size = font_size
                        title_shape.TextFrame.TextRange.Font.Bold = True
                        title_shape.TextFrame.TextRange.Font.Name = title_font
                        # Increase height if multi-line
                        lines = display_title.count("\n") + 1
                        title_shape.Height = max(50, lines * font_size * 1.3)
                        if float(title_shape.Left) <= margin_x + 2:
                            title_shape.Left = margin_x + 14
                            title_shape.Width = _clamp_dim(slide_width - title_shape.Left - safe_right)
                        bar = slide.Shapes.AddShape(
                            1,
                            margin_x,
                            float(title_shape.Top),
                            6,
                            float(title_shape.Height),
                        )
                        bar.Fill.Solid()
                        bar.Fill.ForeColor.RGB = _rgb_to_int(primary)
                        bar.Line.Visible = False
                        _apply_shape_animation(slide, bar, constants, "enter", anim_effect)
                    except Exception:
                        pass

                title_bottom = margin_top
                if title_shape is not None:
                    try:
                        title_bottom = float(title_shape.Top) + float(title_shape.Height)
                    except Exception:
                        title_bottom = margin_top + 50

                # Section subtitle line
                if section:
                    try:
                        subtitle = slide.Shapes.AddTextbox(
                            1,
                            margin_x,
                            title_bottom + 4,
                            _clamp_dim(slide_width - margin_x - safe_right),
                            22,
                        )
                        tr = subtitle.TextFrame.TextRange
                        tr.Text = section if not subsection else f"{section} - {subsection}"
                        tr.Font.Name = note_font
                        tr.Font.Size = note_base_size
                        tr.Font.Color.RGB = _rgb_to_int(accent)
                        title_bottom = float(subtitle.Top) + float(subtitle.Height)
                        _apply_shape_animation(slide, subtitle, constants, "enter", anim_effect)
                    except Exception:
                        pass

                # Divider line under title/subtitle
                try:
                    line = slide.Shapes.AddShape(
                        1,
                        margin_x,
                        title_bottom + 4,
                        _clamp_dim(slide_width - margin_x - safe_right),
                        2,
                    )
                    line.Fill.Solid()
                    line.Fill.ForeColor.RGB = _rgb_to_int(divider)
                    line.Line.Visible = False
                except Exception:
                    pass

                # Pre-scan for formula-only lines to reserve image space.
                has_formula = False
                if render_latex:
                    for _line in (page_paras or []) + (page_bullets or []):
                        raw = str(_line).strip()
                        if raw.startswith("$") and raw.endswith("$") and len(raw) > 2:
                            has_formula = True
                            break
                        if "\\(" in raw and "\\)" in raw:
                            has_formula = True
                            break

                body_text = ""
                if page_paras:
                    body_text += "\n".join([str(p) for p in page_paras if str(p).strip()])
                if page_bullets:
                    if body_text:
                        body_text += "\n"
                    body_text += "\n".join([f"l {str(b)}" for b in page_bullets if str(b).strip()])

                if not body_text:
                    empty_text_streak += 1
                else:
                    empty_text_streak = 0

                formulas: list[Path] = []
                if render_latex and body_text:
                    kept_lines: list[str] = []
                    for line in body_text.splitlines():
                        raw = line.strip()
                        if raw.startswith("$") and raw.endswith("$") and len(raw) > 2:
                            img = _render_formula_image(raw, output.parent)
                            if img:
                                formulas.append(img)
                                continue
                        if "\\(" in raw and "\\)" in raw:
                            start = raw.find("\\(")
                            end = raw.find("\\)")
                            if start != -1 and end != -1 and end > start:
                                formula = raw[start + 2 : end]
                                img = _render_formula_image(formula, output.parent)
                                if img:
                                    formulas.append(img)
                                    continue
                        kept_lines.append(line)
                    body_text = "\n".join(kept_lines)

                if fill_image_on_empty and ai_image_enabled:
                    if empty_text_streak >= 2 and fig_paths:
                        try:
                            prompt = ""
                            try:
                                prompt = analyze_image_with_vision(
                                    str(fig_paths[0]),
                                    "Briefly describe this image to generate a stylistically similar illustration.",
                                )
                            except Exception:
                                prompt = "Generate an abstract illustration matching the current theme."
                            img_ext = (os.getenv("IMAGE_OUTPUT_FORMAT") or "png").strip().lower()
                            gen_path = _generate_background_image(
                                prompt,
                                output,
                                filename=f"pptx_ai_fill_{page_no + 1}.{img_ext}",
                            )
                            if gen_path:
                                fig_paths = [str(gen_path)]
                                empty_text_streak = 0
                            else:
                                ai_image_failure_count += 1
                        except Exception:
                            ai_image_failure_count += 1
                    elif body_text and not fig_paths and ai_image_count < max_ai_images:
                        try:
                            keywords = _extract_keywords(page_bullets, page_paras, max_items=6)
                            topic = title_text or section or "presentation slide"
                            hex_palette = [
                                f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in (primary, accent, neutral)
                            ]
                            prompt = (
                                f"Create a clean, minimal illustration for a presentation slide about: {topic}. "
                                f"Keywords: {', '.join(keywords) if keywords else 'research, analysis'}. "
                                f"Style: flat vector, soft gradients, modern, no text, no labels. "
                                f"Palette: {', '.join(hex_palette)}."
                            )
                            img_ext = (os.getenv("IMAGE_OUTPUT_FORMAT") or "png").strip().lower()
                            gen_path = _generate_background_image(
                                prompt,
                                output,
                                filename=f"pptx_ai_{ai_image_count + 1}.{img_ext}",
                            )
                            if gen_path:
                                fig_paths = [str(gen_path)]
                                ai_image_count += 1
                            else:
                                ai_image_failure_count += 1
                        except Exception:
                            ai_image_failure_count += 1
                    if ai_image_failure_count >= max_ai_failures:
                        ai_image_enabled = False
                        print(
                            "[PptxComWriterTool] Reached AI image failure threshold; "
                            "disabling further AI image generation for remaining slides."
                        )

                available_top = title_bottom + gap_y
                available_height = max(0, slide_height - margin_bottom - available_top)
                image_area_height = 0.0
                text_area_height = available_height
                text_len = len(body_text)
                fig_count = len(fig_paths)
                prefer_image = fig_count >= 2 and text_len < 120
                prefer_text = text_len > 260 and fig_count <= 1
                if (has_formula or (show_images and fig_paths)) and available_height > 0:
                    image_area_height = min(available_height * 0.4, max(80.0, available_height * 0.3))
                    text_area_height = available_height - image_area_height - gap_y
                    if text_area_height < 80:
                        deficit = 80 - text_area_height
                        image_area_height = max(0.0, image_area_height - deficit)
                        text_area_height = max(0.0, available_height - image_area_height - gap_y)
                    if text_area_height <= 0:
                        text_area_height = available_height
                        image_area_height = 0.0
                if not body_text and (has_formula or (show_images and fig_paths)):
                    text_area_height = 0.0
                    image_area_height = available_height
                if prefer_image and image_area_height > 0:
                    image_area_height = min(available_height * 0.55, max(120.0, available_height * 0.45))
                    text_area_height = max(0.0, available_height - image_area_height - gap_y)
                if prefer_text and image_area_height > 0:
                    image_area_height = min(available_height * 0.25, max(60.0, available_height * 0.2))
                    text_area_height = max(0.0, available_height - image_area_height - gap_y)

                body_shape = None
                card_shape = None
                if card_layout and body_text and text_area_height > 0:
                    try:
                        card_shape = slide.Shapes.AddShape(
                            1,
                            margin_x,
                            available_top,
                            _clamp_dim(slide_width - margin_x - safe_right),
                            _clamp_dim(text_area_height, 40),
                        )
                        card_shape.Fill.Solid()
                        card_shape.Fill.ForeColor.RGB = _rgb_to_int(bg_light)
                        card_shape.Line.Visible = True
                        card_shape.Line.ForeColor.RGB = _rgb_to_int(divider)
                        try:
                            card_shape.Shadow.Visible = True
                            card_shape.Shadow.Transparency = 0.7
                        except Exception:
                            pass
                    except Exception:
                        card_shape = None
                if body_text and text_area_height > 0:
                    try:
                        body_shape = slide.Shapes.AddTextbox(
                            1,
                            margin_x,
                            available_top,
                            _clamp_dim(slide_width - margin_x - safe_right),
                            _clamp_dim(text_area_height, 40),
                        )
                        tf = body_shape.TextFrame
                        tf.TextRange.Text = body_text
                        tf.TextRange.Font.Name = body_font
                        tf.TextRange.Font.Size = body_font_size
                        tf.TextRange.Font.Color.RGB = _rgb_to_int(text_color)
                        tf.WordWrap = True
                        try:
                            tf.AutoSize = 1
                        except Exception:
                            pass
                        try:
                            tf.MarginLeft = 8
                            tf.MarginRight = 8
                            tf.MarginTop = 6
                            tf.MarginBottom = 6
                        except Exception:
                            pass
                    except Exception:
                        body_shape = None
                if body_shape is not None and body_text:
                    _style_bullet_icons(body_shape, body_text)

                if title_shape is not None:
                    try:
                        title_shape.Line.Visible = True
                        title_shape.Line.ForeColor.RGB = _rgb_to_int(divider)
                        title_shape.Fill.Visible = True
                        title_shape.Fill.Solid()
                        title_shape.Fill.ForeColor.RGB = _rgb_to_int(bg_light)
                        title_shape.TextFrame.TextRange.Font.Color.RGB = _rgb_to_int(primary)
                    except Exception:
                        pass
                if body_shape is not None:
                    try:
                        if card_shape is not None:
                            body_shape.Line.Visible = False
                            body_shape.Fill.Visible = False
                        else:
                            body_shape.Line.Visible = True
                            body_shape.Line.ForeColor.RGB = _rgb_to_int(divider)
                            body_shape.Fill.Visible = True
                            body_shape.Fill.Solid()
                            body_shape.Fill.ForeColor.RGB = _rgb_to_int(bg_light)
                        body_shape.TextFrame.TextRange.Font.Color.RGB = _rgb_to_int(text_color)
                    except Exception:
                        pass

                # Figures layout (grid or two-column)
                if (formulas or (show_images and fig_paths)) and image_area_height >= 80:
                    use_two_col = prefer_text and (formulas or fig_paths)
                    if use_two_col:
                        left = margin_x
                        col_gap = 20
                        col_w = (_clamp_dim(slide_width - margin_x - safe_right) - col_gap) / 2
                        top = available_top
                        max_h = min(available_height, slide_height - margin_bottom - top)
                        if body_shape is not None:
                            try:
                                body_shape.Left = left
                                body_shape.Top = top
                                body_shape.Width = col_w
                                body_shape.Height = max_h
                            except Exception:
                                pass
                        img_area_left = left + col_w + col_gap
                        img_area_top = top
                        img_area_w = col_w
                        img_area_h = max_h
                    else:
                        img_area_left = margin_x
                        img_area_top = available_top + (0 if prefer_image else text_area_height + (gap_y if body_text else 0))
                        img_area_w = _clamp_dim(slide_width - margin_x - safe_right)
                        img_area_h = min(image_area_height, slide_height - margin_bottom - img_area_top)
                        if prefer_image and body_shape is not None and text_area_height > 0:
                            try:
                                body_shape.Top = img_area_top + img_area_h + gap_y
                                body_shape.Height = min(text_area_height, slide_height - margin_bottom - body_shape.Top)
                            except Exception:
                                pass

                    if card_layout:
                        try:
                            img_card = slide.Shapes.AddShape(
                                1,
                                img_area_left,
                                img_area_top,
                                _clamp_dim(img_area_w),
                                _clamp_dim(img_area_h, 60),
                            )
                            img_card.Fill.Solid()
                            img_card.Fill.ForeColor.RGB = _rgb_to_int(bg_light)
                            img_card.Line.Visible = True
                            img_card.Line.ForeColor.RGB = _rgb_to_int(divider)
                            try:
                                img_card.Shadow.Visible = True
                                img_card.Shadow.Transparency = 0.75
                            except Exception:
                                pass
                        except Exception:
                            img_card = None

                    max_w = img_area_w
                    max_h = img_area_h
                    if max_h > 60 and max_w > 100:
                        all_imgs = formulas + (fig_paths if show_images else [])
                        # Swap figure path with chart render if available
                        all_imgs = [chart_map.get(str(p), p) for p in all_imgs]
                        fig_caps = {}
                        for p, cap, meta in fig_info:
                            fig_caps[str(p)] = str(cap)
                            fig_caps[str(p)] = str(cap)
                        if len(all_imgs) <= 1:
                            cols = 1
                        elif len(all_imgs) <= 4:
                            cols = 2
                        else:
                            cols = 3
                        rows = (len(all_imgs) + cols - 1) // cols
                        cell_w = max_w / cols
                        cell_h = max_h / max(1, rows)
                        pad = 6
                        try:
                            from PIL import Image  # type: ignore[import]
                        except Exception:
                            Image = None  # type: ignore[assignment]
                        for i, path in enumerate(all_imgs):
                            r = i // cols
                            c = i % cols
                            x = img_area_left + c * cell_w + pad
                            y = img_area_top + r * cell_h + pad
                            w = max(10, cell_w - pad * 2)
                            h = max(10, cell_h - pad * 2)
                            if Image is not None:
                                try:
                                    with Image.open(str(path)) as img:
                                        iw, ih = img.size
                                    fit_w, fit_h = _fit_image_box(iw, ih, w, h)
                                    x = x + (w - fit_w) / 2
                                    y = y + (h - fit_h) / 2
                                    w, h = fit_w, fit_h
                                except Exception:
                                    pass
                            try:
                                pic = slide.Shapes.AddPicture(
                                    FileName=str(path),
                                    LinkToFile=False,
                                    SaveWithDocument=True,
                                    Left=x,
                                    Top=y,
                                    Width=w,
                                    Height=h,
                                )
                                try:
                                    pic.Line.Visible = True
                                    pic.Line.Weight = 1.0
                                    pic.Line.ForeColor.RGB = _rgb_to_int(divider)
                                    pic.Shadow.Visible = True
                                    pic.Shadow.Transparency = 0.6
                                    try:
                                        pic.SoftEdge.Radius = 2
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                try:
                                    cap = fig_caps.get(str(path), "")
                                    lowered = (cap + " " + str(path)).lower()
                                    if any(key in lowered for key in ("table", "chart", "plot", "curve")):
                                        tag = slide.Shapes.AddShape(
                                            1,
                                            float(pic.Left) + 6,
                                            float(pic.Top) + 6,
                                            64,
                                            20,
                                        )
                                        tag.Fill.Solid()
                                        tag.Fill.ForeColor.RGB = _rgb_to_int(primary)
                                        tag.Line.Visible = False
                                        tr = tag.TextFrame.TextRange
                                        tr.Text = "Chart"
                                        tr.Font.Name = note_font
                                        tr.Font.Size = max(10, note_base_size - 2)
                                        tr.Font.Color.RGB = _rgb_to_int(_contrast_text_color(primary))
                                except Exception:
                                    pass
                                _apply_shape_animation(slide, pic, constants, "enter", image_anim_effect)
                            except Exception:
                                continue

                if title_shape is not None:
                    _apply_shape_animation(slide, title_shape, constants, "enter", anim_effect)
                if body_shape is not None:
                    _apply_shape_animation(slide, body_shape, constants, anim_mode, anim_effect)

                page_no += 1
                _add_header_footer(slide, section, page_no)

        print(f"[PptxComWriterTool] Saving presentation to: {output}")
        ppt.SaveAs(str(output))
        print(f"[PptxComWriterTool] Finished writing PPTX to: {output}")
        return json.dumps({"output_path": str(output)})
    except Exception as exc:  # noqa: BLE001
        print(f"[PptxComWriterTool] COM writer failed: {exc}")
        return json.dumps({"error": f"PPTX COM writer failed: {exc}"})
    finally:
        try:
            if ppt is not None:
                ppt.Close()
        except Exception:
            pass
        try:
            if app is not None:
                app.Quit()
        except Exception:
            pass
        if com_initialized and pythoncom is not None:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass


@tool(
    "PptxWriterTool",
    args_schema=PptxWriterInput,
    description=(
        "Render a .pptx PowerPoint file directly from a structured PPT plan (frames + figures). "
        "This uses the same plan_json that LatexWriterTool consumes, but writes a native PPTX file "
        "via python-pptx so that slides can be opened and edited in Microsoft PowerPoint."
    ),
)
def pptx_writer_tool(
    output_path: str,
    plan_json: str,
    aspect_ratio: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    author: Optional[str] = None,
    institute: Optional[str] = None,
    date: Optional[str] = None,
    lang: Optional[str] = None,
    pptx_template: Optional[str] = None,
    theme_palette: Optional[str] = None,
) -> str:
    if Presentation is None or Inches is None or Pt is None:
        return json.dumps(
            {
                "error": "python-pptx is not installed; please install it with 'pip install python-pptx' or via requirements.txt.",
            }
        )

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".pptx":
        output = output.with_suffix(".pptx")

    raw = (plan_json or "").strip()
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

    plan = None
    parse_error: Optional[Exception] = None
    for parser_name in ("json", "ast"):
        try:
            if parser_name == "json":
                plan = json.loads(cleaned)
            else:
                plan = ast.literal_eval(cleaned)
            break
        except Exception as exc:  # noqa: BLE001
            parse_error = exc

    if plan is None:
        raw_preview = raw[:500].replace("\n", " ")
        cleaned_preview = cleaned[:500].replace("\n", " ")
        print(f"[PptxWriterTool] Failed to parse plan_json: {parse_error}")
        print(f"[PptxWriterTool] Raw plan_json preview (first 500 chars): {raw_preview}")
        print(f"[PptxWriterTool] Cleaned plan_json preview (first 500 chars): {cleaned_preview}")
        return json.dumps({"error": f"invalid plan_json: {parse_error}"})

    frames_data = plan.get("frames") or []
    figures_meta = plan.get("figures") or plan.get("figure_index") or []
    lang_norm = str(lang or "").lower() if isinstance(lang, str) else ""

    # Resolve PPTX template usage early so that downstream styling decisions
    # (e.g., theme colors) can respect whether a user-provided template should
    # fully control the visual appearance.
    template_path_obj: Optional[Path] = None
    use_template = False
    if pptx_template:
        try:
            template_path_obj = Path(pptx_template).expanduser().resolve()
            if template_path_obj.exists():
                use_template = True
        except Exception:  # noqa: BLE001
            template_path_obj = None

    def _parse_hex_color(value: str) -> Optional[tuple[int, int, int]]:
        raw = value.strip().lstrip("#")
        if len(raw) == 3:
            raw = "".join(ch * 2 for ch in raw)
        if len(raw) != 6:
            return None
        try:
            r = int(raw[0:2], 16)
            g = int(raw[2:4], 16)
            b = int(raw[4:6], 16)
        except ValueError:
            return None
        return r, g, b

    def _blend_rgb(
        base: tuple[int, int, int], target: tuple[int, int, int], ratio: float
    ) -> tuple[int, int, int]:
        clamped = max(0.0, min(ratio, 1.0))
        return (
            int(base[0] + (target[0] - base[0]) * clamped),
            int(base[1] + (target[1] - base[1]) * clamped),
            int(base[2] + (target[2] - base[2]) * clamped),
        )

    def _parse_palette(value: Optional[str]) -> list[tuple[int, int, int]]:
        if not value or not isinstance(value, str):
            return []
        items = []
        for part in value.replace(";", ",").split(","):
            hex_val = part.strip()
            if not hex_val:
                continue
            rgb = _parse_hex_color(hex_val)
            if rgb is not None:
                items.append(rgb)
        return items

    def _ppt_theme_colors(palette: list[tuple[int, int, int]]) -> dict:
        if RGBColor is None:
            return {}
        if palette:
            primary_rgb = palette[2] if len(palette) >= 3 else palette[0]
            accent_rgb = palette[0]
            border_rgb = palette[3] if len(palette) >= 4 else _blend_rgb(primary_rgb, (0, 0, 0), 0.2)
            deep_rgb = palette[4] if len(palette) >= 5 else _blend_rgb(primary_rgb, (0, 0, 0), 0.35)
            panel_fill_rgb = _blend_rgb(primary_rgb, (255, 255, 255), 0.9)
            return {
                "primary": RGBColor(*primary_rgb),
                "accent": RGBColor(*accent_rgb),
                "deep": RGBColor(*deep_rgb),
                "panel_fill": RGBColor(*panel_fill_rgb),
                "panel_border": RGBColor(*border_rgb),
                "frame": RGBColor(*primary_rgb),
                "text": RGBColor(0, 0, 0),
            }
        primary = RGBColor(31, 78, 121)
        accent = RGBColor(0, 120, 215)
        panel_fill = RGBColor(240, 244, 252)
        panel_border = RGBColor(199, 212, 247)
        text_color = RGBColor(0, 0, 0)
        return {
            "primary": primary,
            "accent": accent,
            "panel_fill": panel_fill,
            "panel_border": panel_border,
            "frame": primary,
            "text": text_color,
        }

    def _normalize_ppt_text(value: str) -> str:
        if not value:
            return value
        s = str(value)
        if not any(ch in s for ch in ("$", "\\", "_", "^")):
            return s
        # Collapse doubled backslashes that often appear after JSON escaping
        # so that sequences like "\\leq" become "\leq".
        s = s.replace("\\\\", "\\")
        s = re.sub(r"\$\$(.+?)\$\$", r"\1", s, flags=re.DOTALL)
        s = re.sub(r"\$(.+?)\$", r"\1", s, flags=re.DOTALL)
        s = re.sub(r"\\\[(.+?)\\\]", r"\1", s, flags=re.DOTALL)
        s = re.sub(r"\\\((.+?)\\\)", r"\1", s, flags=re.DOTALL)
        s = s.replace("\\left", "").replace("\\right", "")
        # Normalize a few low-level escaped characters so that "\_" and
        # "\^" render as the intended ASCII symbols in PPT text, and
        # "\|" becomes a plain vertical bar.
        s = s.replace(r"\_", "_").replace(r"\^", "^").replace(r"\|", "|")
        replacements = {
            "\\leq": "≤",
            "\\geq": "≥",
            "\\ne": "≠",
            "\\in": "∈",
            "\\cdot": "路",
            "\\times": "脳",
            "\\pm": "卤",
            "\\sum": "Σ",
            "\\int": "∫",
            "\\sqrt": "√",
            "\\theta": "胃",
            "\\tau": "蟿",
            "\\sigma": "蟽",
            "\\xi": "尉",
            "\\mu": "渭",
            "\\lambda": "位",
            "\\gamma": "纬",
            "\\beta": "尾",
            "\\alpha": "伪",
            "\\rho": "蟻",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        s = re.sub(r"\\mathbb\{([^}]+)\}", r"\1", s)
        s = re.sub(r"\\boldsymbol\{([^}]+)\}", r"\1", s)
        s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s.strip()

    def _parse_aspect_ratio(value: Optional[str]) -> Optional[tuple[float, float]]:
        if not value or not isinstance(value, str):
            return None
        raw_value = value.strip().lower()
        raw_value = raw_value.replace("：", ":").replace("×", "x")
        if raw_value in {"auto", "default", "template"}:
            return None
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)\s*$", raw_value)
        if not match:
            return None
        try:
            width = float(match.group(1))
            height = float(match.group(2))
        except ValueError:
            return None
        if width <= 0 or height <= 0:
            return None
        return width, height

    def _force_horizontal_text(text_frame) -> None:
        try:
            body_pr = text_frame._element.bodyPr
            body_pr.set("vert", "horz")
            body_pr.set("rot", "0")
        except Exception:  # noqa: BLE001
            pass

    def _apply_text_frame_fit(text_frame, allow_auto_size: bool = True) -> None:
        try:
            text_frame.word_wrap = True
        except Exception:  # noqa: BLE001
            pass
        _force_horizontal_text(text_frame)
        if MSO_ANCHOR is not None:
            try:
                text_frame.vertical_anchor = MSO_ANCHOR.TOP
            except Exception:  # noqa: BLE001
                pass
        try:
            text_frame.margin_left = Inches(0.12)
            text_frame.margin_right = Inches(0.12)
            text_frame.margin_top = Inches(0.08)
            text_frame.margin_bottom = Inches(0.06)
        except Exception:  # noqa: BLE001
            pass
        if allow_auto_size and MSO_AUTO_SIZE is not None:
            try:
                text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
            except Exception:  # noqa: BLE001
                pass

    def _pick_line_spacing(text_len: int, bullet_count: int, para_count: int) -> tuple[float, int]:
        if text_len > 1400 or bullet_count > 9 or para_count > 6:
            return 1.0, 2
        if text_len > 900 or bullet_count > 6 or para_count > 4:
            return 1.05, 3
        return 1.12, 4

    def _emu_to_inches(value: int) -> float:
        return float(value) / 914400 if value else 0.0

    def _clamp_emu(value: float, min_in: float, max_in: float) -> int:
        if Inches is None:
            return int(value)
        return max(int(Inches(min_in)), min(int(value), int(Inches(max_in))))

    def _estimate_lines(text: str, shape_width_emu: int, font_size: int) -> int:
        if not text:
            return 0
        width_in = _emu_to_inches(shape_width_emu)
        width_pt = width_in * 72.0
        if width_pt <= 0:
            return max(1, len(text) // 20)
        char_width = font_size * (0.9 if lang_norm.startswith("zh") else 0.55)
        chars_per_line = max(int(width_pt / max(char_width, 1.0)), 8)
        return max(1, (len(text) + chars_per_line - 1) // chars_per_line)

    def _estimate_text_height(
        paragraphs: list[str],
        bullets: list[str],
        shape_width_emu: int,
        font_size: int,
        line_spacing: float,
        space_after_pt: int,
    ) -> int:
        total_lines = 0
        blocks = 0
        for para in paragraphs:
            if not para:
                continue
            total_lines += _estimate_lines(str(para), shape_width_emu, font_size)
            blocks += 1
        for bullet in bullets:
            if not bullet:
                continue
            total_lines += _estimate_lines(str(bullet), shape_width_emu, font_size)
            blocks += 1
        height_pt = total_lines * font_size * line_spacing + blocks * space_after_pt
        if height_pt <= 0:
            return 0
        if Inches is None:
            return int(height_pt / 72.0 * 914400)
        return int(Inches(height_pt / 72.0))

    def _fit_text_to_height(
        paragraphs: list[str],
        bullets: list[str],
        shape_width_emu: int,
        max_height_emu: int,
        base_font_size: int,
        text_len: int,
        para_count: int,
        bullet_count: int,
    ) -> tuple[int, float, int, int]:
        font_size = base_font_size
        line_spacing, space_after = _pick_line_spacing(text_len, bullet_count, para_count)
        est_height = _estimate_text_height(
            paragraphs,
            bullets,
            shape_width_emu,
            font_size,
            line_spacing,
            space_after,
        )
        for _ in range(6):
            if est_height <= max_height_emu or font_size <= 12:
                break
            font_size = max(12, font_size - 2)
            line_spacing, space_after = _pick_line_spacing(text_len, bullet_count, para_count)
            est_height = _estimate_text_height(
                paragraphs,
                bullets,
                shape_width_emu,
                font_size,
                line_spacing,
                space_after,
            )
        return font_size, line_spacing, space_after, est_height

    def _clamp_shape_box(shape, top_min, bottom_max) -> None:
        if shape is None:
            return
        try:
            top = int(shape.top)
            height = int(shape.height)
            if top < int(top_min):
                top = int(top_min)
            if top + height > int(bottom_max):
                height = max(int(bottom_max) - top, int(Inches(0.3)))
            shape.top = top
            if height > 0:
                shape.height = height
        except Exception:  # noqa: BLE001
            pass

    def _text_stats(paragraphs: list[str], bullets: list[str]) -> tuple[int, int, int]:
        paras = [p for p in paragraphs if p]
        bull = [b for b in bullets if b]
        text = " ".join(_normalize_ppt_text(str(item)) for item in paras + bull)
        return len(text), len(paras), len(bull)

    def _pick_body_font_size(
        text_len: int, para_count: int, bullet_count: int, has_figures: bool
    ) -> int:
        size = 20
        if has_figures:
            size -= 2
        if text_len > 800 or bullet_count > 6 or para_count > 4:
            size -= 2
        if text_len > 1200 or bullet_count > 9 or para_count > 6:
            size -= 2
        if text_len > 1700 or bullet_count > 12:
            size -= 2
        return max(size, 12)

    def _pick_title_font_size(title_text: str, base: int, min_size: int) -> int:
        length = len(title_text.strip()) if title_text else 0
        size = base
        if length > 60:
            size = base - 8
        elif length > 45:
            size = base - 6
        elif length > 32:
            size = base - 4
        return max(size, min_size)

    def _pick_toc_font_size(section_count: int) -> int:
        if section_count >= 12:
            return 16
        if section_count >= 9:
            return 18
        return 20

    def _pick_image_top(text_len: int, para_count: int, bullet_count: int):
        if text_len > 1500 or bullet_count > 10 or para_count > 6:
            return Inches(3.6)
        if text_len > 1000 or bullet_count > 7:
            return Inches(3.3)
        if text_len > 600 or bullet_count > 4:
            return Inches(3.0)
        return Inches(2.6)

    def _content_region(title_shape) -> tuple[int, int, int, int, int]:
        margin_x = _clamp_emu(slide_width * 0.07, 0.6, 1.1)
        margin_top = _clamp_emu(slide_height * 0.08, 0.6, 1.1)
        margin_bottom = _clamp_emu(slide_height * 0.08, 0.6, 1.1)
        gap_y = _clamp_emu(slide_height * 0.03, 0.15, 0.35)
        title_bottom = None
        if title_shape is not None:
            try:
                title_bottom = int(title_shape.top + title_shape.height)
            except Exception:  # noqa: BLE001
                title_bottom = None
        top = max(margin_top, int(title_bottom + gap_y) if title_bottom is not None else margin_top)
        left = margin_x
        width = int(slide_width - 2 * margin_x)
        bottom = int(slide_height - margin_bottom)
        if bottom <= top + gap_y:
            bottom = top + _clamp_emu(slide_height * 0.4, 2.0, 3.5)
        height = int(bottom - top)
        return left, top, width, height, gap_y

    def _shape_bounds(shape) -> Optional[tuple[int, int, int, int]]:
        if shape is None:
            return None
        try:
            return int(shape.left), int(shape.top), int(shape.width), int(shape.height)
        except Exception:  # noqa: BLE001
            return None

    def _rows_from_pattern(pattern: list[int], paths: list[str]) -> list[list[str]]:
        rows: list[list[str]] = []
        idx = 0
        for count in pattern:
            rows.append(paths[idx : idx + count])
            idx += count
        return rows

    def _score_layout(
        pattern: list[int],
        dims: list[tuple[Optional[int], Optional[int]]],
        max_width: int,
        max_height: int,
        gutter: int,
    ) -> tuple[float, float]:
        if not pattern:
            return -1.0, -1.0
        rows_count = len(pattern)
        height_available = max_height - gutter * (rows_count - 1)
        if height_available <= 0:
            return -1.0, -1.0
        row_height = height_available / rows_count
        idx = 0
        scales: list[float] = []
        for cols in pattern:
            width_available = max_width - gutter * (cols - 1)
            if cols <= 0 or width_available <= 0:
                return -1.0, -1.0
            col_width = width_available / cols
            for _ in range(cols):
                w, h = dims[idx] if idx < len(dims) else (None, None)
                idx += 1
                if w and h:
                    scales.append(min(col_width / w, row_height / h))
                else:
                    scales.append(min(col_width / max_width, row_height / max_height))
        if not scales:
            return -1.0, -1.0
        return min(scales), sum(scales) / len(scales)

    def _build_image_rows(
        fig_paths: list[str],
        max_width: int,
        max_height: int,
        gutter: int,
    ) -> list[list[str]]:
        if len(fig_paths) <= 1:
            return [fig_paths]
        dims: list[tuple[Optional[int], Optional[int]]] = []
        for img_path in fig_paths:
            try:
                w, h = get_image_dimensions(Path(img_path))
            except Exception:  # noqa: BLE001
                w, h = None, None
            dims.append((w, h))

        patterns: list[list[int]] = []
        count = len(fig_paths)
        if count == 2:
            patterns = [[2], [1, 1]]
        elif count == 3:
            patterns = [[3], [2, 1], [1, 2], [1, 1, 1]]
        else:
            patterns = [[4], [2, 2], [3, 1], [1, 3]]

        best_pattern = patterns[0]
        best_min, best_avg = -1.0, -1.0
        for pattern in patterns:
            min_scale, avg_scale = _score_layout(pattern, dims, max_width, max_height, gutter)
            if min_scale > best_min or (min_scale == best_min and avg_scale > best_avg):
                best_pattern = pattern
                best_min = min_scale
                best_avg = avg_scale

        return _rows_from_pattern(best_pattern, fig_paths)

    def _continuation_title(title_text: str, index: int) -> str:
        if index <= 1:
            return title_text
        suffix = "（续）" if lang_norm.startswith("zh") else " (cont.)"
        return f"{title_text}{suffix}"

    def _split_paragraph_text(text: str, max_len: int) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []
        if len(raw) <= max_len:
            return [raw]
        parts = re.split(r"(?<=[。！？!?])\s*", raw)
        cleaned = [p.strip() for p in parts if p and p.strip()]
        if not cleaned:
            cleaned = [raw]
        chunks: list[str] = []
        buf = ""
        for part in cleaned:
            if not buf:
                buf = part
                continue
            if len(buf) + len(part) + 1 <= max_len:
                buf = f"{buf} {part}"
            else:
                chunks.append(buf)
                buf = part
        if buf:
            chunks.append(buf)
        if not chunks:
            return [raw[i : i + max_len] for i in range(0, len(raw), max_len)]
        return chunks

    def _split_text_frame(
        frame: dict,
        max_chars: int,
        max_bullets: int,
        max_paras: int,
    ) -> list[dict]:
        paragraphs = frame.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        bullets = frame.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        if not paragraphs and not bullets:
            return [frame]

        items: list[tuple[str, str]] = []
        for para in paragraphs:
            if not para:
                continue
            for part in _split_paragraph_text(str(para), max_chars):
                items.append(("para", part))
        for bullet in bullets:
            if not bullet:
                continue
            items.append(("bullet", str(bullet)))

        slides: list[dict] = []
        cur_paras: list[str] = []
        cur_bullets: list[str] = []
        cur_len = 0

        def _flush() -> None:
            if not cur_paras and not cur_bullets:
                return
            new_frame = dict(frame)
            new_frame["paragraphs"] = list(cur_paras)
            new_frame["bullets"] = list(cur_bullets)
            slides.append(new_frame)

        for kind, text in items:
            text_len = len(text)
            need_flush = False
            if cur_len + text_len > max_chars and (cur_paras or cur_bullets):
                need_flush = True
            if kind == "para" and len(cur_paras) >= max_paras and (cur_paras or cur_bullets):
                need_flush = True
            if kind == "bullet" and len(cur_bullets) >= max_bullets and (cur_paras or cur_bullets):
                need_flush = True
            if need_flush:
                _flush()
                cur_paras = []
                cur_bullets = []
                cur_len = 0
            if kind == "para":
                cur_paras.append(text)
            else:
                cur_bullets.append(text)
            cur_len += text_len
        _flush()

        if len(slides) <= 1:
            return slides

        base_title = str(frame.get("frame_title") or frame.get("section") or "")
        for idx, slide in enumerate(slides, start=1):
            slide["frame_title"] = _continuation_title(base_title, idx)
        return slides

    def _expand_pptx_frames(frames: list[dict]) -> list[dict]:
        expanded: list[dict] = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            paragraphs = frame.get("paragraphs") or []
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]
            bullets = frame.get("bullets") or []
            if isinstance(bullets, str):
                bullets = [bullets]
            fig_slugs = frame.get("figure_slugs") or []
            if isinstance(fig_slugs, str):
                fig_slugs = [fig_slugs]

            text_len, para_count, bullet_count = _text_stats(paragraphs, bullets)
            has_figs = bool(fig_slugs)
            if has_figs and (text_len > 700 or bullet_count > 4 or para_count > 3):
                text_frame = dict(frame)
                text_frame["figure_slugs"] = []
                text_frames = _split_text_frame(
                    text_frame, max_chars=1300, max_bullets=8, max_paras=5
                )
                expanded.extend(text_frames)

                fig_frame = dict(frame)
                fig_frame["paragraphs"] = []
                fig_frame["bullets"] = []
                fig_frame["figure_slugs"] = fig_slugs
                base_title = str(frame.get("frame_title") or frame.get("section") or "")
                suffix = "图示" if lang_norm.startswith("zh") else "Figures"
                fig_frame["frame_title"] = f"{base_title} - {suffix}" if base_title else suffix
                expanded.append(fig_frame)
                continue

            if text_len > 1300 or bullet_count > 8 or para_count > 5:
                expanded.extend(
                    _split_text_frame(
                        frame, max_chars=1300, max_bullets=8, max_paras=5
                    )
                )
                continue

            expanded.append(frame)
        return expanded

    def _apply_shape_border(shape, line_color, line_width_pt: float = 1.0) -> None:
        if shape is None or line_color is None or Pt is None:
            return
        try:
            line = shape.line
            line.color.rgb = line_color
            line.width = Pt(line_width_pt)
        except Exception:  # noqa: BLE001
            pass

    def _add_slide_frame(slide) -> None:
        if use_template or not theme_colors or RGBColor is None or MSO_SHAPE is None:
            return
        frame_color = theme_colors.get("frame") or theme_colors.get("primary")
        if frame_color is None:
            return
        try:
            margin = int(slide_width * 0.01)
            if margin < 2:
                margin = 2
            border = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                margin,
                margin,
                slide_width - 2 * margin,
                slide_height - 2 * margin,
            )
            fill = border.fill
            fill.background()
            _apply_shape_border(border, frame_color, line_width_pt=1.2)
        except Exception:  # noqa: BLE001
            return

    def _should_move_tail_frame(frame: dict) -> bool:
        text = " ".join(
            str(frame.get(key) or "") for key in ("frame_title", "section", "subsection")
        ).strip()
        if not text:
            return False
        text_lower = text.lower()
        en_keywords = (
            "conclusion",
            "conclusions",
            "recommendation",
            "recommendations",
            "future work",
            "future research",
        )
        if any(kw in text_lower for kw in en_keywords):
            return True
        cn_keywords = (
            "\u7ed3\u8bba",
            "\u603b\u7ed3",
            "\u5efa\u8bae",
            "\u5bf9\u7b56",
            "\u653f\u7b56\u5efa\u8bae",
            "\u5c55\u671b",
            "\u672a\u6765\u5de5\u4f5c",
            "\u5efa\u8bae\u4e0e",
        )
        return any(kw in text for kw in cn_keywords)

    def _reorder_conclusion_recommendation_frames(frames: list[dict]) -> list[dict]:
        if not frames or len(frames) == 1:
            return frames
        preserved = [frames[0]]
        head: list[dict] = []
        tail: list[dict] = []
        for frame in frames[1:]:
            if _should_move_tail_frame(frame):
                tail.append(frame)
            else:
                head.append(frame)
        if not tail:
            return frames
        return preserved + head + tail

    palette = _parse_palette(theme_palette)
    theme_colors = _ppt_theme_colors(palette) if not use_template else {}

    print(
        f"[PptxWriterTool] Start writing PPTX: output={output}, frames={len(frames_data)}, "
        f"figures_meta={len(figures_meta or [])}"
    )

    # Mirror the LaTeX writer's behavior: first collapse extremely sparse
    # text-only frames and drop exact textual duplicates within the same
    # (section, subsection) so that both the Beamer PDF and the PPTX share the
    # same high-level slide structure.
    condensed_frames: List[dict] = []
    seen_text_by_section: dict[tuple[str, str | None], set[str]] = {}

    for frame in frames_data:
        if not isinstance(frame, dict):
            continue

        section = (frame.get("section") or "").strip()
        subsection_val = frame.get("subsection")
        subsection = subsection_val.strip() if isinstance(subsection_val, str) else subsection_val
        key = (section, subsection)

        paragraphs = frame.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        bullets = frame.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        figure_slugs_val = frame.get("figure_slugs") or []
        if isinstance(figure_slugs_val, str):
            figure_slugs_val = [figure_slugs_val]

        text_body = "\n".join(str(p) for p in paragraphs) + "\n" + "\n".join(str(b) for b in bullets)
        norm_text = " ".join(text_body.split()).lower()

        has_fig = bool(figure_slugs_val)
        if norm_text:
            seen_set = seen_text_by_section.setdefault(key, set())
            if norm_text in seen_set and not has_fig:
                continue
            seen_set.add(norm_text)

        text_len = len(norm_text)
        num_para = len(paragraphs)
        num_bullets = len(bullets)
        is_sparse_text_only = (
            not has_fig
            and norm_text
            and text_len < 80
            and num_para <= 1
            and num_bullets == 0
        )
        is_moderately_sparse = (
            not has_fig
            and norm_text
            and text_len < 160
            and (num_para + num_bullets) <= 3
        )

        if (is_sparse_text_only or is_moderately_sparse) and condensed_frames:
            prev = condensed_frames[-1]
            prev_section = (prev.get("section") or "").strip()
            prev_sub_val = prev.get("subsection")
            prev_sub = prev_sub_val.strip() if isinstance(prev_sub_val, str) else prev_sub_val
            prev_figs = prev.get("figure_slugs") or []
            if isinstance(prev_figs, str):
                prev_figs = [prev_figs]

            if prev_section == section and prev_sub == subsection and not prev_figs:
                prev_paras = prev.get("paragraphs") or []
                if isinstance(prev_paras, str):
                    prev_paras = [prev_paras]
                prev_bullets = prev.get("bullets") or []
                if isinstance(prev_bullets, str):
                    prev_bullets = [prev_bullets]

                prev_text_body = "\n".join(str(p) for p in prev_paras) + "\n" + "\n".join(
                    str(b) for b in prev_bullets
                )
                prev_norm_text = " ".join(prev_text_body.split()).lower()
                combined_len = len(prev_norm_text) + text_len

                if combined_len <= 320:
                    prev_paras.extend(paragraphs)
                    prev_bullets.extend(bullets)
                    prev["paragraphs"] = prev_paras
                    prev["bullets"] = prev_bullets
                    continue

        frame["paragraphs"] = paragraphs
        frame["bullets"] = bullets
        frame["figure_slugs"] = figure_slugs_val
        condensed_frames.append(frame)

    frames_data = condensed_frames
    frames_data = _expand_pptx_frames(frames_data)

    figures_by_slug: dict[str, dict] = {}
    for fig in figures_meta or []:
        if not isinstance(fig, dict):
            continue
        slug = fig.get("slug")
        path_str = fig.get("path")
        if not slug and path_str:
            slug = Path(path_str).stem
        if not slug:
            continue
        figures_by_slug[slug] = fig

    for frame in frames_data or []:
        if not isinstance(frame, dict):
            continue
        slugs_val = frame.get("figure_slugs") or []
        if isinstance(slugs_val, str):
            slugs = [slugs_val]
        else:
            slugs = list(slugs_val)
        frame["figure_slugs"] = [s for s in slugs if s]

    try:
        if use_template and template_path_obj is not None:
            prs = Presentation(str(template_path_obj))
        else:
            prs = Presentation()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"failed to initialize PowerPoint presentation: {exc}"})

    ratio = _parse_aspect_ratio(aspect_ratio)
    if ratio is not None:
        width_ratio, height_ratio = ratio
        base_height_in = 7.5
        width_in = base_height_in * (width_ratio / height_ratio)
        try:
            prs.slide_width = Inches(width_in)
            prs.slide_height = Inches(base_height_in)
        except Exception:  # noqa: BLE001
            pass

    slide_width = prs.slide_width
    slide_height = prs.slide_height

    def _add_picture_fit(slide, path: str, left: int, top: int, width: int, height: int) -> None:
        try:
            pic = slide.shapes.add_picture(path, left, top)
        except Exception as exc:  # noqa: BLE001
            print(f"[PptxWriterTool] Failed to add picture {path}: {exc}")
            return

        try:
            scale_w = width / pic.width if pic.width and pic.width > width else 1.0
            scale_h = height / pic.height if pic.height and pic.height > height else 1.0
            scale = min(scale_w, scale_h, 1.0)
        except Exception:  # noqa: BLE001
            scale = 1.0

        if scale < 1.0:
            try:
                pic.width = int(pic.width * scale)
                pic.height = int(pic.height * scale)
            except Exception:  # noqa: BLE001
                pass

        try:
            pic.left = int(left + (width - pic.width) / 2)
            pic.top = int(top + (height - pic.height) / 2)
        except Exception:  # noqa: BLE001
            pass

    def _add_decorative_chrome(slide, kind: str = "content") -> None:
        """Add a minimal built-in visual frame when no template is used.

        This draws a narrow vertical accent bar on the left and a subtle
        footer panel at the bottom so that slides look less like plain
        white text pages. When a PPTX template is provided, this helper is
        a no-op to avoid fighting with the template's own design.
        """

        if use_template or not theme_colors or RGBColor is None or MSO_SHAPE is None:
            return
        try:
            primary = theme_colors.get("primary") or RGBColor(31, 78, 121)
            panel_fill = theme_colors.get("panel_fill") or RGBColor(240, 244, 252)
            panel_border = theme_colors.get("panel_border") or RGBColor(199, 212, 247)
            accent = theme_colors.get("accent") or RGBColor(0, 120, 215)

            # Vertical accent bar on the left edge for most slide types.
            if kind in {"title", "toc", "content", "closing"}:
                bar_width = int(slide_width * 0.018)
                if bar_width > 0:
                    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, bar_width, slide_height)
                    fill = bar.fill
                    fill.solid()
                    fill.fore_color.rgb = primary
                    try:
                        bar.line.fill.background()
                    except Exception:  # noqa: BLE001
                        pass

            # Subtle footer strip along the bottom for content-like slides.
            if kind in {"toc", "content"}:
                footer_height = int(slide_height * 0.055)
                if footer_height > 0:
                    footer_top = slide_height - footer_height
                    footer = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        0,
                        footer_top,
                        slide_width,
                        footer_height,
                    )
                    fill = footer.fill
                    fill.solid()
                    fill.fore_color.rgb = panel_fill
                    line = footer.line
                    try:
                        line.color.rgb = primary
                        if Pt is not None:
                            line.width = Pt(1.5)
                    except Exception:  # noqa: BLE001
                        pass

            # Circular accent badge near the top-right corner to reinforce the
            # deep-blue theme and add an extra visual element.
            if kind in {"toc", "content", "title", "closing"}:
                diameter = int(min(slide_width, slide_height) * 0.055)
                if diameter > 0:
                    margin_x = int(slide_width * 0.035)
                    margin_y = int(slide_height * 0.035)
                    left = slide_width - diameter - margin_x
                    top = margin_y
                    badge = slide.shapes.add_shape(
                        MSO_SHAPE.OVAL,
                        left,
                        top,
                        diameter,
                        diameter,
                    )
                    fill = badge.fill
                    fill.solid()
                    fill.fore_color.rgb = accent
                    try:
                        badge.line.fill.background()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            # Decorative chrome should never break PPT generation.
            return

    title_slide_layout = prs.slide_layouts[0]
    title_slide = prs.slides.add_slide(title_slide_layout)
    title_shape = title_slide.shapes.title
    if title_shape is None and title:
        try:
            margin = Inches(0.8)
            title_shape = title_slide.shapes.add_textbox(
                int(margin),
                int(Inches(0.4)),
                int(slide_width - 2 * margin),
                int(Inches(0.9)),
            )
        except Exception:  # noqa: BLE001
            title_shape = None
    if title_shape is not None:
        title_text = str(title) if title is not None else ""
        title_shape.text = title_text
        if hasattr(title_shape, "text_frame"):
            tf = title_shape.text_frame
            try:
                title_shape.rotation = 0
            except Exception:  # noqa: BLE001
                pass
            if not use_template:
                try:
                    margin = Inches(0.8)
                    title_shape.left = int(margin)
                    title_shape.width = int(slide_width - 2 * margin)
                    title_shape.top = int(Inches(0.4))
                    title_shape.height = int(Inches(0.9))
                except Exception:  # noqa: BLE001
                    pass
            if not use_template:
                _clamp_shape_box(title_shape, Inches(0.3), Inches(1.6))
            _apply_text_frame_fit(tf, allow_auto_size=True)
            title_size = _pick_title_font_size(title_text, base=36, min_size=24)
            title_spacing, title_space_after = _pick_line_spacing(len(title_text), 0, 1)
            for p in tf.paragraphs:
                for r in p.runs:
                    f = r.font
                    if not use_template:
                        if lang_norm.startswith("zh"):
                            f.name = "Microsoft YaHei"
                        else:
                            f.name = "Calibri"
                        f.size = Pt(title_size)
                    if RGBColor is not None and theme_colors:
                        try:
                            primary = theme_colors.get("primary") or RGBColor(31, 78, 121)
                            f.color.rgb = primary
                        except Exception:
                            pass
                try:
                    p.line_spacing = title_spacing
                    if Pt is not None:
                        p.space_after = Pt(title_space_after)
                except Exception:  # noqa: BLE001
                    pass

    if not use_template and theme_colors:
        _apply_shape_border(
            title_shape,
            theme_colors.get("panel_border") or theme_colors.get("primary"),
            line_width_pt=1.2,
        )

    _add_decorative_chrome(title_slide, kind="title")
    _add_slide_frame(title_slide)
    subtitle_shape = None
    if len(title_slide.placeholders) > 1:
        try:
            subtitle_shape = title_slide.placeholders[1]
        except Exception:  # noqa: BLE001
            subtitle_shape = None
    if subtitle_shape is None and (subtitle or author or institute or date):
        try:
            margin = Inches(0.8)
            subtitle_shape = title_slide.shapes.add_textbox(
                int(margin),
                int(Inches(1.6)),
                int(slide_width - 2 * margin),
                int(Inches(1.0)),
            )
        except Exception:  # noqa: BLE001
            subtitle_shape = None
    if subtitle_shape is not None:
        lines: list[str] = []
        if subtitle:
            lines.append(str(subtitle))
        author_line_parts: list[str] = []
        if author:
            author_line_parts.append(str(author))
        if institute:
            author_line_parts.append(str(institute))
        if author_line_parts:
            lines.append(" / ".join(author_line_parts))
        if date:
            lines.append(str(date))
        subtitle_text = "\n".join(lines)
        subtitle_shape.text = subtitle_text
        if hasattr(subtitle_shape, "text_frame"):
            tf = subtitle_shape.text_frame
            try:
                subtitle_shape.rotation = 0
            except Exception:  # noqa: BLE001
                pass
            if not use_template:
                try:
                    margin = Inches(0.8)
                    subtitle_shape.left = int(margin)
                    subtitle_shape.width = int(slide_width - 2 * margin)
                    subtitle_shape.top = int(Inches(1.6))
                    subtitle_shape.height = int(Inches(1.1))
                except Exception:  # noqa: BLE001
                    pass
            if not use_template:
                _clamp_shape_box(subtitle_shape, Inches(1.4), Inches(3.0))
            _apply_text_frame_fit(tf, allow_auto_size=True)
            subtitle_size = _pick_title_font_size(subtitle_text, base=20, min_size=14)
            subtitle_spacing, subtitle_space_after = _pick_line_spacing(len(subtitle_text), 0, 1)
            for p in tf.paragraphs:
                for r in p.runs:
                    f = r.font
                    if not use_template:
                        if lang_norm.startswith("zh"):
                            f.name = "Microsoft YaHei"
                        else:
                            f.name = "Calibri"
                        f.size = Pt(subtitle_size)
                    if RGBColor is not None and theme_colors:
                        try:
                            text_col = theme_colors.get("text") or RGBColor(0, 0, 0)
                            f.color.rgb = text_col
                        except Exception:
                            pass
                try:
                    p.line_spacing = subtitle_spacing
                    if Pt is not None:
                        p.space_after = Pt(subtitle_space_after)
                except Exception:  # noqa: BLE001
                    pass

    if not use_template and theme_colors:
        _apply_shape_border(
            subtitle_shape,
            theme_colors.get("panel_border") or theme_colors.get("primary"),
            line_width_pt=1.0,
        )

    sections: list[str] = []
    for frame in frames_data:
        sec = (frame.get("section") or "").strip()
        if sec and sec not in sections:
            sections.append(sec)
    if sections:
        has_text = bool(sections)
        toc_layout = prs.slide_layouts[1]
        toc_slide = prs.slides.add_slide(toc_layout)
        toc_title_text = "鐩綍" if lang_norm.startswith("zh") else "Contents"
        toc_title = toc_slide.shapes.title
        if toc_title is None:
            try:
                margin = Inches(0.8)
                toc_title = toc_slide.shapes.add_textbox(
                    int(margin),
                    int(Inches(0.4)),
                    int(slide_width - 2 * margin),
                    int(Inches(0.7)),
                )
            except Exception:  # noqa: BLE001
                toc_title = None
        if toc_title is not None:
            toc_title.text = toc_title_text
            if hasattr(toc_title, "text_frame"):
                tf = toc_title.text_frame
                try:
                    toc_title.rotation = 0
                except Exception:  # noqa: BLE001
                    pass
                if not use_template:
                    try:
                        margin = Inches(0.8)
                        toc_title.left = int(margin)
                        toc_title.width = int(slide_width - 2 * margin)
                        toc_title.top = int(Inches(0.4))
                        toc_title.height = int(Inches(0.7))
                    except Exception:  # noqa: BLE001
                        pass
                if not use_template:
                    _clamp_shape_box(toc_title, Inches(0.3), Inches(1.4))
                _apply_text_frame_fit(tf, allow_auto_size=True)
                toc_title_spacing, toc_title_space = _pick_line_spacing(len(toc_title_text), 0, 1)
                for p in tf.paragraphs:
                    for r in p.runs:
                        f = r.font
                        if not use_template:
                            if lang_norm.startswith("zh"):
                                f.name = "Microsoft YaHei"
                            else:
                                f.name = "Calibri"
                            f.size = Pt(28)
                        if RGBColor is not None and theme_colors:
                            try:
                                primary = theme_colors.get("primary") or RGBColor(31, 78, 121)
                                f.color.rgb = primary
                            except Exception:
                                pass
                    try:
                        p.line_spacing = toc_title_spacing
                        if Pt is not None:
                            p.space_after = Pt(toc_title_space)
                    except Exception:  # noqa: BLE001
                        pass
        if not use_template and theme_colors:
            _apply_shape_border(
                toc_title,
                theme_colors.get("panel_border") or theme_colors.get("primary"),
                line_width_pt=1.0,
            )
        body_shape = None
        if len(toc_slide.placeholders) > 1:
            try:
                body_shape = toc_slide.placeholders[1]
            except Exception:  # noqa: BLE001
                body_shape = None
        if body_shape is None and sections:
            try:
                margin = Inches(0.8)
                body_shape = toc_slide.shapes.add_textbox(
                    int(margin),
                    int(Inches(1.4)),
                    int(slide_width - 2 * margin),
                    int(slide_height * 0.6),
                )
            except Exception:  # noqa: BLE001
                body_shape = None
        if body_shape is not None and not has_text:
            body_shape = None
        if body_shape is not None:
            if RGBColor is not None and theme_colors and not use_template:
                try:
                    fill = body_shape.fill
                    fill.solid()
                    panel_fill = theme_colors.get("panel_fill") or RGBColor(240, 244, 252)
                    fill.fore_color.rgb = panel_fill
                    line = body_shape.line
                    primary = theme_colors.get("primary") or RGBColor(31, 78, 121)
                    line.color.rgb = primary
                    if Pt is not None:
                        line.width = Pt(1.5)
                except Exception:
                    pass
            try:
                body_shape.rotation = 0
            except Exception:  # noqa: BLE001
                pass
            if not use_template:
                try:
                    margin = Inches(0.8)
                    body_shape.left = int(margin)
                    body_shape.width = int(slide_width - 2 * margin)
                    body_shape.top = int(Inches(1.4))
                    body_shape.height = int(slide_height * 0.6)
                except Exception:  # noqa: BLE001
                    pass
            if not use_template:
                _clamp_shape_box(body_shape, Inches(1.2), slide_height - Inches(0.9))
            if not use_template and theme_colors:
                _apply_shape_border(
                    body_shape,
                    theme_colors.get("panel_border") or theme_colors.get("primary"),
                    line_width_pt=1.0,
                )
            body = body_shape.text_frame
            body.clear()
            _apply_text_frame_fit(body, allow_auto_size=True)
            toc_font_size = _pick_toc_font_size(len(sections))
            toc_spacing, toc_space_after = _pick_line_spacing(len(sections) * 12, 0, len(sections))
            first = True
            for sec in sections:
                if first:
                    p = body.paragraphs[0]
                    first = False
                else:
                    p = body.add_paragraph()
                p.text = sec
                p.level = 0
                try:
                    p.line_spacing = toc_spacing
                    if Pt is not None:
                        p.space_after = Pt(toc_space_after)
                except Exception:  # noqa: BLE001
                    pass
                for r in p.runs:
                    f = r.font
                    if not use_template:
                        if lang_norm.startswith("zh"):
                            f.name = "Microsoft YaHei"
                        else:
                            f.name = "Calibri"
                    f.size = Pt(toc_font_size)
                    if RGBColor is not None and theme_colors:
                        try:
                            text_col = theme_colors.get("text") or RGBColor(0, 0, 0)
                            f.color.rgb = text_col
                        except Exception:
                            pass

        _add_slide_frame(toc_slide)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    for frame in frames_data:
        frame_title = frame.get("frame_title") or frame.get("section") or ""
        layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(layout)
        _add_decorative_chrome(slide, kind="content")
        _add_slide_frame(slide)
        title_shape = slide.shapes.title
        if title_shape is None and frame_title:
            try:
                margin = Inches(0.8)
                title_shape = slide.shapes.add_textbox(
                    int(margin),
                    int(Inches(0.4)),
                    int(slide_width - 2 * margin),
                    int(Inches(0.7)),
                )
            except Exception:  # noqa: BLE001
                title_shape = None
        if title_shape is not None:
            title_text = str(frame_title)
            title_shape.text = title_text
            if hasattr(title_shape, "text_frame"):
                tf = title_shape.text_frame
                _apply_text_frame_fit(tf, allow_auto_size=True)
                title_size = _pick_title_font_size(title_text, base=28, min_size=20)
                try:
                    title_shape.rotation = 0
                except Exception:  # noqa: BLE001
                    pass
                if not use_template:
                    try:
                        margin = Inches(0.8)
                        title_shape.left = int(margin)
                        title_shape.width = int(slide_width - 2 * margin)
                        title_shape.top = int(Inches(0.4))
                        title_shape.height = int(Inches(0.7))
                    except Exception:  # noqa: BLE001
                        pass
                if not use_template:
                    _clamp_shape_box(title_shape, Inches(0.3), Inches(1.3))
                title_spacing, title_space_after = _pick_line_spacing(len(title_text), 0, 1)
                for p in tf.paragraphs:
                    for r in p.runs:
                        f = r.font
                        if not use_template:
                            if lang_norm.startswith("zh"):
                                f.name = "Microsoft YaHei"
                            else:
                                f.name = "Calibri"
                        f.size = Pt(title_size)
                        if RGBColor is not None and theme_colors:
                            try:
                                primary = theme_colors.get("primary") or RGBColor(31, 78, 121)
                                f.color.rgb = primary
                            except Exception:
                                pass
                    try:
                        p.line_spacing = title_spacing
                        if Pt is not None:
                            p.space_after = Pt(title_space_after)
                    except Exception:  # noqa: BLE001
                        pass
        if not use_template and theme_colors:
            _apply_shape_border(
                title_shape,
                theme_colors.get("panel_border") or theme_colors.get("primary"),
                line_width_pt=1.0,
            )

        paragraphs = frame.get("paragraphs") or []
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        bullets = frame.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        text_len, para_count, bullet_count = _text_stats(paragraphs, bullets)

        slugs = frame.get("figure_slugs") or []
        if isinstance(slugs, str):
            slugs = [slugs]
        fig_paths: list[str] = []
        avg_img_aspect: Optional[float] = None
        for slug in slugs:
            fig = figures_by_slug.get(slug)
            if not fig:
                continue
            fig_path = fig.get("path")
            if not fig_path:
                continue
            fig_path_obj = Path(fig_path).expanduser().resolve()
            if not fig_path_obj.exists():
                continue
            ext = fig_path_obj.suffix.lower()
            if ext not in image_exts:
                continue
            fig_paths.append(str(fig_path_obj))
        if fig_paths:
            aspects: list[float] = []
            for img_path in fig_paths:
                try:
                    w, h = get_image_dimensions(Path(img_path))
                except Exception:  # noqa: BLE001
                    w, h = None, None
                if w and h:
                    aspects.append(h / w)
            if aspects:
                avg_img_aspect = sum(aspects) / len(aspects)

        has_figs = bool(fig_paths)
        has_text = bool(paragraphs or bullets)
        content_left, content_top, content_width, content_height, gap_y = _content_region(title_shape)
        content_bottom = int(content_top + content_height)

        body_shape = None
        body_shape_from_placeholder = False
        if len(slide.placeholders) > 1:
            try:
                body_shape = slide.placeholders[1]
                body_shape_from_placeholder = True
            except Exception:  # noqa: BLE001
                body_shape = None
                body_shape_from_placeholder = False
        if body_shape is None and has_text:
            try:
                body_shape = slide.shapes.add_textbox(
                    int(content_left),
                    int(content_top),
                    int(content_width),
                    int(content_height),
                )
            except Exception:  # noqa: BLE001
                body_shape = None
                body_shape_from_placeholder = False
        if body_shape is None and has_text:
            has_text = False

        body_bounds = _shape_bounds(body_shape) if (use_template and body_shape_from_placeholder) else None
        body_font_size = _pick_body_font_size(
            text_len, para_count, bullet_count, has_figures=has_figs
        )
        line_spacing, space_after = _pick_line_spacing(text_len, bullet_count, para_count)
        body_height = 0
        if has_text:
            text_width = int(body_bounds[2]) if body_bounds else int(content_width)
            min_text_height = _clamp_emu(slide_height * 0.22, 1.4, 2.6)
            if has_figs:
                image_ratio = 0.28
                if avg_img_aspect is not None:
                    if avg_img_aspect > 1.2:
                        image_ratio = 0.34
                    elif avg_img_aspect < 0.75:
                        image_ratio = 0.24
                min_image_height = _clamp_emu(slide_height * image_ratio, 1.6, 3.6)
                max_text_height = max(min_text_height, int(content_height - min_image_height - gap_y))
                max_text_height = min(int(content_height), max_text_height)
            else:
                max_text_height = int(content_height)
            if body_bounds:
                max_text_height = int(body_bounds[3]) if body_bounds[3] > 0 else max_text_height
                min_text_height = min(min_text_height, max_text_height)
            body_font_size, line_spacing, space_after, est_height = _fit_text_to_height(
                paragraphs,
                bullets,
                text_width,
                int(max_text_height),
                body_font_size,
                text_len,
                para_count,
                bullet_count,
            )
            padding = _clamp_emu(slide_height * 0.01, 0.08, 0.18)
            if body_bounds:
                body_height = int(max_text_height)
            else:
                body_height = max(min_text_height, min(max_text_height, est_height + padding))
        if body_shape is not None:
            if not use_template and RGBColor is not None and theme_colors:
                try:
                    fill = body_shape.fill
                    fill.solid()
                    panel_fill = theme_colors.get("panel_fill") or RGBColor(240, 244, 252)
                    fill.fore_color.rgb = panel_fill
                    line = body_shape.line
                    primary = theme_colors.get("primary") or RGBColor(31, 78, 121)
                    line.color.rgb = primary
                    if Pt is not None:
                        line.width = Pt(1.5)
                except Exception:  # noqa: BLE001
                    pass
            if has_figs and not has_text:
                try:
                    body_shape.fill.background()
                    body_shape.line.fill.background()
                except Exception:  # noqa: BLE001
                    pass
            try:
                body_shape.rotation = 0
            except Exception:  # noqa: BLE001
                pass
            if has_text and (not use_template or not body_shape_from_placeholder):
                try:
                    body_shape.left = int(content_left)
                    body_shape.width = int(content_width)
                    body_shape.top = int(content_top)
                    body_shape.height = int(body_height)
                except Exception:  # noqa: BLE001
                    pass
                _clamp_shape_box(body_shape, content_top, content_bottom)
                try:
                    body_height = int(body_shape.height)
                except Exception:  # noqa: BLE001
                    pass
            if not use_template and theme_colors and has_text:
                _apply_shape_border(
                    body_shape,
                    theme_colors.get("panel_border") or theme_colors.get("primary"),
                    line_width_pt=1.0,
                )
            body = body_shape.text_frame
            body.clear()
            _apply_text_frame_fit(body, allow_auto_size=True)
            first_para = True
            for para in paragraphs:
                if not para:
                    continue
                if first_para:
                    p = body.paragraphs[0]
                    first_para = False
                else:
                    p = body.add_paragraph()
                p.text = _normalize_ppt_text(str(para))
                p.level = 0
                try:
                    p.line_spacing = line_spacing
                    if Pt is not None:
                        p.space_after = Pt(space_after)
                except Exception:  # noqa: BLE001
                    pass
                for r in p.runs:
                    f = r.font
                    if not use_template:
                        if lang_norm.startswith("zh"):
                            f.name = "Microsoft YaHei"
                        else:
                            f.name = "Calibri"
                    f.size = Pt(body_font_size)
                    if RGBColor is not None and theme_colors:
                        try:
                            text_col = theme_colors.get("text") or RGBColor(0, 0, 0)
                            f.color.rgb = text_col
                        except Exception:
                            pass
            for bullet in bullets:
                if not bullet:
                    continue
                if first_para:
                    p = body.paragraphs[0]
                    first_para = False
                else:
                    p = body.add_paragraph()
                p.text = _normalize_ppt_text(str(bullet))
                p.level = 1
                try:
                    p.line_spacing = line_spacing
                    if Pt is not None:
                        p.space_after = Pt(space_after)
                except Exception:  # noqa: BLE001
                    pass
                for r in p.runs:
                    f = r.font
                    if not use_template:
                        if lang_norm.startswith("zh"):
                            f.name = "Microsoft YaHei"
                        else:
                            f.name = "Calibri"
                    f.size = Pt(body_font_size)
                    if RGBColor is not None and theme_colors:
                        try:
                            text_col = theme_colors.get("text") or RGBColor(0, 0, 0)
                            primary = theme_colors.get("primary") or text_col
                            f.color.rgb = primary
                        except Exception:
                            pass

        if not fig_paths:
            continue
        if len(fig_paths) > 4:
            print(
                f"[PptxWriterTool] Too many figures for one slide ({len(fig_paths)}); "
                "showing the first 4."
            )
            fig_paths = fig_paths[:4]
        # Place figures in the remaining content region under the text box.
        gutter = _clamp_emu(slide_height * 0.02, 0.15, 0.3)
        image_left = int(content_left)
        max_width = int(content_width)
        image_top_base = content_top
        if has_text and body_shape is not None:
            bounds = _shape_bounds(body_shape)
            if bounds:
                image_top_base = bounds[1]
                if use_template and body_shape_from_placeholder:
                    image_left = bounds[0]
                    max_width = bounds[2]
        top = int(image_top_base + body_height + gap_y) if has_text else int(content_top)
        if top < content_top:
            top = int(content_top)
        max_height = int(content_bottom - top)
        if max_height <= 0:
            continue

        if len(fig_paths) == 1:
            _add_picture_fit(
                slide,
                fig_paths[0],
                int(image_left),
                int(top),
                int(max_width),
                int(max_height),
            )
            continue

        rows = _build_image_rows(fig_paths, int(max_width), int(max_height), int(gutter))

        row_count = len(rows)
        row_height = (max_height - gutter * (row_count - 1)) / row_count if row_count else max_height

        for row_idx, row in enumerate(rows):
            col_count = len(row)
            if col_count == 0:
                continue
            col_width = (max_width - gutter * (col_count - 1)) / col_count
            row_top = top + row_idx * (row_height + gutter)
            row_total_width = col_width * col_count + gutter * (col_count - 1)
            row_left = image_left + (max_width - row_total_width) / 2
            for col_idx, img_path in enumerate(row):
                left = row_left + col_idx * (col_width + gutter)
                _add_picture_fit(
                    slide,
                    img_path,
                    int(left),
                    int(row_top),
                    int(col_width),
                    int(row_height),
                )

    # Append a closing "thank you" slide so that the PPTX mirrors the
    # Beamer deck's final slide.
    closing_title = "谢谢观看！" if lang_norm.startswith("zh") else "Thank you!"
    closing_body = "谢谢观看！" if lang_norm.startswith("zh") else "Thank you!"
    try:
        closing_layout = prs.slide_layouts[1]
    except Exception:  # noqa: BLE001
        closing_layout = prs.slide_layouts[0]
    closing_slide = prs.slides.add_slide(closing_layout)
    _add_decorative_chrome(closing_slide, kind="closing")
    _add_slide_frame(closing_slide)
    closing_title_shape = closing_slide.shapes.title
    if closing_title_shape is None:
        try:
            margin = Inches(0.8)
            closing_title_shape = closing_slide.shapes.add_textbox(
                int(margin),
                int(Inches(0.4)),
                int(slide_width - 2 * margin),
                int(Inches(0.7)),
            )
        except Exception:  # noqa: BLE001
            closing_title_shape = None
    if closing_title_shape is not None:
        closing_title_shape.text = closing_title
        try:
            closing_title_shape.rotation = 0
        except Exception:  # noqa: BLE001
            pass
        if hasattr(closing_title_shape, "text_frame"):
            tf = closing_title_shape.text_frame
            if not use_template:
                try:
                    margin = Inches(0.8)
                    closing_title_shape.left = int(margin)
                    closing_title_shape.width = int(slide_width - 2 * margin)
                    closing_title_shape.top = int(Inches(0.4))
                    closing_title_shape.height = int(Inches(0.7))
                except Exception:  # noqa: BLE001
                    pass
            if not use_template:
                _clamp_shape_box(closing_title_shape, Inches(0.3), Inches(1.4))
            _apply_text_frame_fit(tf, allow_auto_size=True)
            closing_title_spacing, closing_title_space = _pick_line_spacing(len(closing_title), 0, 1)
            for p in tf.paragraphs:
                for r in p.runs:
                    f = r.font
                    if not use_template:
                        if lang_norm.startswith("zh"):
                            f.name = "Microsoft YaHei"
                        else:
                            f.name = "Calibri"
                        f.size = Pt(28)
                        if RGBColor is not None and theme_colors:
                            try:
                                primary = theme_colors.get("primary") or RGBColor(31, 78, 121)
                                f.color.rgb = primary
                            except Exception:
                                pass
                try:
                    p.line_spacing = closing_title_spacing
                    if Pt is not None:
                        p.space_after = Pt(closing_title_space)
                except Exception:  # noqa: BLE001
                    pass
    if not use_template and theme_colors:
        _apply_shape_border(
            closing_title_shape,
            theme_colors.get("panel_border") or theme_colors.get("primary"),
            line_width_pt=1.0,
        )
    closing_body_shape = None
    if len(closing_slide.placeholders) > 1:
        try:
            closing_body_shape = closing_slide.placeholders[1]
        except Exception:  # noqa: BLE001
            closing_body_shape = None
    if closing_body_shape is None:
        try:
            margin = Inches(0.8)
            closing_body_shape = closing_slide.shapes.add_textbox(
                int(margin),
                int(Inches(1.5)),
                int(slide_width - 2 * margin),
                int(Inches(1.0)),
            )
        except Exception:  # noqa: BLE001
            closing_body_shape = None
    if closing_body_shape is not None:
        try:
            closing_body_shape.rotation = 0
        except Exception:  # noqa: BLE001
            pass
        if not use_template:
            try:
                margin = Inches(0.8)
                closing_body_shape.left = int(margin)
                closing_body_shape.width = int(slide_width - 2 * margin)
                closing_body_shape.top = int(Inches(1.5))
                closing_body_shape.height = int(Inches(1.0))
            except Exception:  # noqa: BLE001
                pass
        if not use_template:
            _clamp_shape_box(closing_body_shape, Inches(1.2), Inches(3.0))
        if not use_template and theme_colors:
            _apply_shape_border(
                closing_body_shape,
                theme_colors.get("panel_border") or theme_colors.get("primary"),
                line_width_pt=1.0,
            )
        body_tf = closing_body_shape.text_frame
        body_tf.clear()
        _apply_text_frame_fit(body_tf, allow_auto_size=True)
        closing_body_spacing, closing_body_space = _pick_line_spacing(len(closing_body), 0, 1)
        if body_tf.paragraphs:
            body_tf.paragraphs[0].text = closing_body
        else:
            p = body_tf.add_paragraph()
            p.text = closing_body
        for p in body_tf.paragraphs:
            for r in p.runs:
                f = r.font
                if not use_template:
                    if lang_norm.startswith("zh"):
                        f.name = "Microsoft YaHei"
                    else:
                        f.name = "Calibri"
                    f.size = Pt(20)
                if RGBColor is not None and theme_colors:
                    try:
                        text_col = theme_colors.get("text") or RGBColor(0, 0, 0)
                        f.color.rgb = text_col
                    except Exception:
                        pass
            try:
                p.line_spacing = closing_body_spacing
                if Pt is not None:
                    p.space_after = Pt(closing_body_space)
            except Exception:  # noqa: BLE001
                pass

    try:
        prs.save(str(output))
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"failed to save PPTX file: {exc}"})

    print(f"[PptxWriterTool] Finished writing PPTX to: {output}")
    return json.dumps({"output_path": str(output)})


class CompileCheckInput(BaseModel):
    tex_path: str
    workdir: Optional[str] = None
    log_dir: Optional[str] = None


@tool(
    "CompileCheckTool",
    args_schema=CompileCheckInput,
    description="Run pdflatex/latexmk to compile a LaTeX file and return success flag and log snippet.",
)
def compile_check_tool(tex_path: str, workdir: Optional[str] = None, log_dir: Optional[str] = None) -> str:
    tex_file = Path(tex_path).expanduser().resolve()
    if not tex_file.exists():
        print(f"[CompileCheckTool] LaTeX file not found: {tex_file}")
        return json.dumps({"success": False, "error": f"tex file not found: {tex_file}"})
    cwd = Path(workdir).expanduser().resolve() if workdir else tex_file.parent
    # Use xelatex for better Unicode and Chinese support in Beamer documents.
    if shutil.which("xelatex") is None:
        return json.dumps(
            {"success": False, "error": "xelatex not found in PATH; please install TeX Live or MiKTeX."}
        )
    cmd = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name]
    print(f"[CompileCheckTool] Start compiling LaTeX with xelatex: {tex_file} (workdir: {cwd})")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as exc:
        print(f"[CompileCheckTool] Exception while running pdflatex: {exc}")
        return json.dumps({"success": False, "error": str(exc), "returncode": -1, "pdf_generated": False})

    full_log = proc.stdout or ""
    log = full_log[-6000:] if full_log else ""
    success = proc.returncode == 0
    pdf_path = tex_file.with_suffix(".pdf")
    pdf_generated = pdf_path.exists()
    log_root = Path(log_dir).expanduser().resolve() if log_dir else cwd
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{tex_file.stem}_compile.log"
    try:
        log_path.write_text(full_log, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    error_msg = ""
    if not success:
        if "!" in log:
            # Keep the last meaningful LaTeX error line when available.
            for line in reversed(log.splitlines()):
                if line.strip().startswith("!"):
                    error_msg = line.strip()
                    break
        if not error_msg:
            error_msg = f"xelatex failed with return code {proc.returncode}"

    print(f"[CompileCheckTool] Compile {'succeeded' if success else 'failed'}: output PDF = {pdf_path}")
    return json.dumps(
        {
            "success": success,
            "log": log,
            "pdf_path": str(pdf_path),
            "returncode": proc.returncode,
            "pdf_generated": pdf_generated,
            "log_path": str(log_path),
            "error": error_msg,
        }
    )


class PptSlideExportInput(BaseModel):
    pdf_path: str
    output_dir: str = "export/slides"
    format: str = "png"
    dpi: int = 200


@tool(
    "PptSlideExportTool",
    args_schema=PptSlideExportInput,
    description="Export each page of a compiled Beamer PDF into slide images (PNG/JPEG).",
)
def ppt_slide_export_tool(pdf_path: str, output_dir: str = "export/slides", format: str = "png", dpi: int = 200) -> str:
    pdf_file = Path(pdf_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pdf_file.exists():
        print(f"[PptSlideExportTool] PDF not found for slide export: {pdf_file}")
        return json.dumps({"images": [], "error": f"pdf not found: {pdf_file}"})
    try:
        print(
            f"[PptSlideExportTool] Start exporting slide images: PDF={pdf_file}, output_dir={out_dir}, "
            f"format={format}, dpi={dpi}"
        )
        images = convert_from_path(str(pdf_file), dpi=dpi)
    except Exception as exc:
        print(f"[PptSlideExportTool] Exception while exporting slides: {exc}")
        return json.dumps({"images": [], "error": str(exc)})
    saved = []
    ext = format.lower()
    for index, image in enumerate(images, start=1):
        filename = f"slide_{index:03d}.{ext}"
        out_path = out_dir / filename
        image.save(out_path, format=ext.upper())
        saved.append(str(out_path))
    print(f"[PptSlideExportTool] Finished exporting {len(saved)} slide images to {out_dir}")
    return json.dumps({"images": saved})
