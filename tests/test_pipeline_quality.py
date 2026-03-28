from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_stubs() -> None:
    if "langchain.tools" not in sys.modules:
        langchain_mod = types.ModuleType("langchain")
        langchain_tools = types.ModuleType("langchain.tools")
        langchain_messages = types.ModuleType("langchain.messages")

        def tool(*_args, **_kwargs):  # noqa: ANN001, ANN002
            def _decorator(fn):
                fn.invoke = lambda payload=None, _fn=fn: _fn(**(payload or {}))
                return fn

            return _decorator

        class _Msg:
            def __init__(self, content: str = "") -> None:
                self.content = content

        langchain_tools.tool = tool
        langchain_messages.HumanMessage = _Msg
        langchain_messages.SystemMessage = _Msg
        langchain_mod.tools = langchain_tools
        langchain_mod.messages = langchain_messages

        sys.modules["langchain"] = langchain_mod
        sys.modules["langchain.tools"] = langchain_tools
        sys.modules["langchain.messages"] = langchain_messages

    if "langchain_text_splitters" not in sys.modules:
        splitters_mod = types.ModuleType("langchain_text_splitters")

        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=2000, chunk_overlap=200):  # noqa: D401
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_documents(self, docs):
                return docs

        splitters_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        sys.modules["langchain_text_splitters"] = splitters_mod

    if "langchain_core.documents" not in sys.modules:
        core_docs_mod = types.ModuleType("langchain_core.documents")

        class Document:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        core_docs_mod.Document = Document
        sys.modules["langchain_core.documents"] = core_docs_mod

    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv_stub

    if "pdf2image" not in sys.modules:
        pdf2image_stub = types.ModuleType("pdf2image")
        pdf2image_stub.convert_from_path = lambda *_args, **_kwargs: []
        sys.modules["pdf2image"] = pdf2image_stub

    if "ppt_agent.agent" not in sys.modules:
        agent_stub = types.ModuleType("ppt_agent.agent")
        agent_stub.build_model = lambda: None
        agent_stub.build_agent = lambda *_args, **_kwargs: None
        agent_stub.build_pipeline_agents = lambda *_args, **_kwargs: {}
        sys.modules["ppt_agent.agent"] = agent_stub

    openai_mod = sys.modules.get("openai")
    if openai_mod is None:
        openai_mod = types.ModuleType("openai")
        sys.modules["openai"] = openai_mod

    if not hasattr(openai_mod, "OpenAI"):
        class _DummyOpenAI:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        openai_mod.OpenAI = _DummyOpenAI


_install_stubs()

from ppt_agent.cli import (  # noqa: E402
    REPORT_TYPE_ZHENGDA,
    _apply_zhengda_template_to_section_blocks,
    _align_frame_titles_to_outline,
    _build_outline_from_section_blocks,
    _ensure_conclusion_frames,
    _ensure_conclusion_section_block,
    _ensure_outline_coverage_frames,
    _map_frames_to_outline_sections,
    _expand_frames_to_target_min,
    _refine_section_blocks_for_outline,
)
from ppt_agent.tools import evaluate_text_quality, load_text_chunks_with_fallback  # noqa: E402


class TextQualityTests(unittest.TestCase):
    def test_quality_scoring_variants(self):
        clean_zh = "这是一段正常中文文本，用于测试质量评分逻辑。"
        mojibake = "nnnnnnnn IIIIIIII nnnnnnnn IIIIIIII"
        mojibake_gbk_utf8 = "鐩綍 鍥? 涓€銆? 锛堜竴锛?"
        mojibake_weird_zh = "缁撴灉涓庡垎鏋? 鐮旂┒鏂规硶涓庢ā鍨? 闂"
        mixed = "研究显示 accuracy=0.91，但存在少量乱码占位符 nnnn。"
        english = "This is an English abstract with clean readable text."

        q_clean = evaluate_text_quality(clean_zh, lang="zh")
        q_bad = evaluate_text_quality(mojibake, lang="zh")
        q_bad_gbk = evaluate_text_quality(mojibake_gbk_utf8, lang="zh")
        q_bad_weird = evaluate_text_quality(mojibake_weird_zh, lang="zh")
        q_mixed = evaluate_text_quality(mixed, lang="zh")
        q_en = evaluate_text_quality(english, lang="en")

        self.assertFalse(q_clean["is_mojibake"])
        self.assertTrue(q_bad["is_mojibake"])
        self.assertTrue(q_bad_gbk["is_mojibake"])
        self.assertTrue(q_bad_weird["is_mojibake"])
        self.assertGreater(q_mixed["score"], q_bad["score"])
        self.assertFalse(q_en["is_mojibake"])


class FallbackDecisionTests(unittest.TestCase):
    def test_pdf_bad_docx_good(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "report.pdf"
            docx = root / "report.docx"
            pdf.write_bytes(b"%PDF")
            docx.write_bytes(b"PK")

            def fake_load(path: Path, chunk_size=2000, chunk_overlap=200):  # noqa: ARG001
                if path.suffix.lower() == ".pdf":
                    return [{"page_content": "nnnnnnnn IIIIIIII nnnnnnnn IIIIIIII", "metadata": {}}]
                return [{"page_content": "这是可读的中文正文内容。", "metadata": {}}]

            with patch("ppt_agent.tools._load_text_chunks_from_path", side_effect=fake_load):
                result = load_text_chunks_with_fallback(
                    pdf_path=str(pdf),
                    lang="zh",
                    mojibake_policy="auto_fallback",
                    enable_ocr_fallback=True,
                )

            self.assertEqual(result["source"], "docx")
            self.assertFalse(result["quality"]["is_mojibake"])
            self.assertTrue(result["chunks"])
            self.assertGreaterEqual(len(result.get("fallback_chain") or []), 2)

    def test_pdf_bad_ocr_good(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF")

            def fake_load(path: Path, chunk_size=2000, chunk_overlap=200):  # noqa: ARG001
                return [{"page_content": "nnnnnnnn IIIIIIII nnnnnnnn IIIIIIII", "metadata": {}}]

            def fake_ocr(path: Path, chunk_size=2000, chunk_overlap=200):  # noqa: ARG001
                return [{"page_content": "这是 OCR 提取的可读中文内容。", "metadata": {"page": 1}}]

            with patch("ppt_agent.tools._load_text_chunks_from_path", side_effect=fake_load):
                with patch("ppt_agent.tools._ocr_pdf_to_chunks", side_effect=fake_ocr):
                    result = load_text_chunks_with_fallback(
                        pdf_path=str(pdf),
                        lang="zh",
                        mojibake_policy="auto_fallback",
                        enable_ocr_fallback=True,
                    )

            self.assertEqual(result["source"], "ocr")
            self.assertFalse(result["quality"]["is_mojibake"])
            self.assertTrue(result["chunks"])

    def test_all_paths_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF")

            with patch(
                "ppt_agent.tools._load_text_chunks_from_path",
                side_effect=RuntimeError("loader failure"),
            ):
                with patch(
                    "ppt_agent.tools._ocr_pdf_to_chunks",
                    side_effect=RuntimeError("ocr failure"),
                ):
                    result = load_text_chunks_with_fallback(
                        pdf_path=str(pdf),
                        lang="zh",
                        mojibake_policy="auto_fallback",
                        enable_ocr_fallback=True,
                    )

            self.assertEqual(result["source"], "none")
            self.assertIn("error", result)


class DetailedPageTargetTests(unittest.TestCase):
    def test_expand_to_detailed_target_range(self):
        frames = []
        for i in range(8):
            frames.append(
                {
                    "section": f"章节{i % 4 + 1}",
                    "subsection": None,
                    "frame_title": f"分析页 {i + 1}",
                    "bullets": ["要点A", "要点B", "要点C", "要点D", "要点E", "要点F"],
                    "paragraphs": ["这是用于扩展页数机制测试的段落说明。"],
                    "figure_slugs": [],
                    "source_pages": [i + 1],
                    "source_chunk_indices": [i],
                }
            )
        figures = []
        for i in range(20):
            figures.append(
                {
                    "slug": f"fig_{i}",
                    "path": f"/tmp/fig_{i}.png",
                    "caption": f"图表 {i}",
                    "context": "图表用于补充分析。",
                    "page": i + 1,
                }
            )
        expanded = _expand_frames_to_target_min(
            frames=frames,
            aligned_figures=figures,
            target_min_pages=25,
            target_max_pages=35,
            lang="zh",
        )
        self.assertGreaterEqual(len(expanded), 25)
        self.assertLessEqual(len(expanded), 35)


class OutlineQualityTests(unittest.TestCase):
    def test_zhengda_template_outline_order(self):
        blocks = [
            {
                "section": "绪论",
                "subsection": None,
                "text": "研究背景说明",
                "pages": [1],
                "chunk_indices": [0],
            },
            {
                "section": "研究设计",
                "subsection": None,
                "text": "调查框架与样本设计",
                "pages": [5],
                "chunk_indices": [2],
            },
            {
                "section": "实证分析",
                "subsection": "结构方程模型",
                "text": "采用 SEM 模型进行路径检验",
                "pages": [16],
                "chunk_indices": [8],
            },
            {
                "section": "实证分析",
                "subsection": "随机森林",
                "text": "采用随机森林模型进行变量重要性评估",
                "pages": [19],
                "chunk_indices": [10],
            },
        ]
        normalized = _apply_zhengda_template_to_section_blocks(blocks, lang="zh")
        outline = _build_outline_from_section_blocks(
            normalized,
            lang="zh",
            report_type=REPORT_TYPE_ZHENGDA,
        )
        self.assertEqual(outline[0], "背景分析")
        self.assertEqual(outline[1], "文献综述")
        self.assertEqual(outline[2], "调查框架设计")
        self.assertIn("问卷分析：结构方程模型", outline)
        self.assertIn("问卷分析：随机森林模型", outline)

    def test_outline_preserves_readable_pdf_heading(self):
        blocks = [
            {
                "section": "调研过程质量控制",
                "subsection": None,
                "text": "本节介绍调查执行过程中的质量控制与抽样复核。",
                "pages": [7],
                "chunk_indices": [3],
            },
            {
                "section": "问卷预测试与信效度分析",
                "subsection": None,
                "text": "包含 Cronbach α 与 KMO/Bartlett 检验。",
                "pages": [10],
                "chunk_indices": [5],
            },
        ]
        outline = _build_outline_from_section_blocks(blocks, lang="zh")
        self.assertIn("调研过程质量控制", outline)
        self.assertIn("问卷预测试与信效度分析", outline)

    def test_refine_section_blocks_drops_toc_noise(self):
        blocks = [
            {
                "section": "目录",
                "subsection": None,
                "text": "1 绪论...................3\n2 方法...................8\n3 结果...................15",
                "pages": [1],
                "chunk_indices": [0],
            },
            {
                "section": "Paper Content",
                "subsection": None,
                "text": "本节介绍 K-modes 聚类方法、变量筛选和模型评估流程。",
                "pages": [8],
                "chunk_indices": [5],
            },
        ]
        refined = _refine_section_blocks_for_outline(blocks, lang="zh")
        self.assertEqual(len(refined), 1)
        self.assertNotIn("目录", refined[0]["section"])

    def test_ensure_conclusion_section_block(self):
        blocks = [
            {
                "section": "研究背景与问题",
                "subsection": None,
                "text": "背景说明",
                "pages": [1],
                "chunk_indices": [0],
            },
            {
                "section": "结果与分析",
                "subsection": None,
                "text": "结果说明",
                "pages": [10],
                "chunk_indices": [8],
            },
        ]
        out, appended = _ensure_conclusion_section_block(blocks, pdf_chunks=[], lang="zh")
        self.assertTrue(appended)
        self.assertEqual(out[-1]["section"], "结论与建议")

    def test_ensure_conclusion_frames(self):
        frames = [
            {
                "section": "结果与分析",
                "subsection": None,
                "frame_title": "核心结果",
                "bullets": ["结果 A", "结果 B"],
                "paragraphs": [],
                "figure_slugs": [],
                "source_pages": [10],
                "source_chunk_indices": [8],
            }
        ]
        out, injected = _ensure_conclusion_frames(frames, lang="zh")
        self.assertTrue(injected)
        self.assertEqual(out[-1]["section"], "结论与建议")

    def test_outline_coverage_and_title_alignment(self):
        blocks = [
            {
                "section": "研究背景与问题",
                "subsection": None,
                "text": "背景内容",
                "pages": [1],
                "chunk_indices": [0],
            },
            {
                "section": "结果与分析",
                "subsection": None,
                "text": "结果内容",
                "pages": [10],
                "chunk_indices": [8],
            },
        ]
        outline = _build_outline_from_section_blocks(blocks, lang="zh")
        frames = [
            {
                "section": "研究背景与问题",
                "subsection": None,
                "frame_title": "图25 聚类评估指标",
                "bullets": ["说明"],
                "paragraphs": [],
                "figure_slugs": [],
                "source_pages": [1],
                "source_chunk_indices": [0],
            }
        ]
        covered, injected = _ensure_outline_coverage_frames(
            frames,
            blocks,
            outline,
            lang="zh",
        )
        self.assertGreaterEqual(injected, 1)
        aligned, rewritten = _align_frame_titles_to_outline(covered, outline, lang="zh")
        self.assertGreaterEqual(rewritten, 1)
        self.assertTrue(any("结果与分析" == f.get("section") for f in aligned))

    def test_map_frames_to_outline_sections_deduplicates(self):
        blocks = [
            {
                "section": "研究背景与问题",
                "subsection": None,
                "text": "背景",
                "pages": [1, 2],
                "chunk_indices": [0, 1],
            },
            {
                "section": "研究方法与模型",
                "subsection": None,
                "text": "方法",
                "pages": [3, 4],
                "chunk_indices": [2, 3],
            },
        ]
        outline = _build_outline_from_section_blocks(blocks, lang="zh")
        frames = [
            {
                "section": "图25 聚类评估指标",
                "subsection": None,
                "frame_title": "图25 聚类评估指标",
                "bullets": ["说明"],
                "paragraphs": [],
                "figure_slugs": [],
                "source_pages": [3],
                "source_chunk_indices": [2],
            },
            {
                "section": "研究方法与模型",
                "subsection": None,
                "frame_title": "方法页",
                "bullets": ["说明"],
                "paragraphs": [],
                "figure_slugs": [],
                "source_pages": [4],
                "source_chunk_indices": [3],
            },
        ]
        mapped, rewrites = _map_frames_to_outline_sections(frames, blocks, outline, lang="zh")
        self.assertGreaterEqual(rewrites, 1)
        unique_sections = {f.get("section") for f in mapped}
        self.assertEqual(unique_sections, {"研究方法与模型"})


if __name__ == "__main__":
    unittest.main()
