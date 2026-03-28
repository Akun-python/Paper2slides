from __future__ import annotations

import os
from typing import List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelCallLimitMiddleware, ToolRetryMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from ppt_agent.logging_utils import get_logger, make_log_print
from ppt_agent.tools import (
    pdf_loader_tool,
    figure_indexer_tool,
    pdf_figure_extract_tool,
    figure_caption_align_tool,
    section_planner_tool,
    latex_writer_tool,
    compile_check_tool,
    ppt_slide_export_tool,
    keyword_search_tool,
    image_understanding_tool,
    pptx_com_writer_tool,
)

logger = get_logger("paper2slides.agent")
print = make_log_print(logger)


class LoggingMiddleware(AgentMiddleware):
    """Lightweight logging middleware for all agents in this project.

    It prints a short message before and after each model call so that the
    end-to-end CLI pipeline remains observable without altering agent logic.
    """

    def before_model(self, state: AgentState, runtime) -> dict | None:  # type: ignore[override]
        try:
            messages = state["messages"]
        except Exception:  # noqa: BLE001
            messages = []
        num_messages = len(messages)
        print(f"[Agent][Middleware] Calling model with {num_messages} messages")
        return None

    def after_model(self, state: AgentState, runtime) -> dict | None:  # type: ignore[override]
        try:
            messages = state["messages"]
        except Exception:  # noqa: BLE001
            messages = []
        if not messages:
            return None
        last = messages[-1]
        content = (getattr(last, "content", None) or "").strip()
        preview = content[:200].replace("\n", " ")
        print(f"[Agent][Middleware] Model output preview: {preview}")
        return None


class NormalizedSection(BaseModel):
    original_section: str
    normalized_section: str


class SectionTitleResponse(BaseModel):
    normalized_sections: List[NormalizedSection]


class SummaryResponse(BaseModel):
    title: str
    bullets: List[str]
    paragraphs: List[str]


class FormulaItem(BaseModel):
    id: Optional[str] = None
    raw_text: str
    latex: Optional[str] = None
    description: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None


class FormulaResponse(BaseModel):
    formulas: List[FormulaItem]


class FigureItem(BaseModel):
    id: Optional[str] = None
    kind: Optional[str] = None
    caption: str
    context: Optional[str] = None
    number: Optional[int] = None
    page: Optional[int] = None


class FigureMetadataResponse(BaseModel):
    figures: List[FigureItem]


class PlannerFrameItem(BaseModel):
    section: Optional[str] = None
    subsection: Optional[str] = None
    frame_title: str
    bullets: List[str] = []
    paragraphs: List[str] = []
    figure_slugs: List[str] = []


class PlannerResponse(BaseModel):
    frames: List[PlannerFrameItem]


def build_model():
    """Initialize chat model.

    Prefer DeepSeek API configuration (APIYI_BASE_URL, APIYI_API_KEY, LLM_MODEL) if provided.
    If these environment variables are not set, fall back to init_chat_model's default behavior.
    """

    base_url = os.environ.get("APIYI_BASE_URL")
    api_key = os.environ.get("APIYI_API_KEY")
    model_name = os.environ.get("LLM_MODEL", "deepseek-chat")

    if base_url and api_key:
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        # Use the OpenAI-compatible HTTP API to talk to DeepSeek
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
        )

    # Fallback: if no DeepSeek-related env vars are configured, use init_chat_model
    return init_chat_model(model_name)


def build_agent(model=None):
    if model is None:
        model = build_model()
    tools = [
        pdf_loader_tool,
        figure_indexer_tool,
        pdf_figure_extract_tool,
        figure_caption_align_tool,
        section_planner_tool,
        latex_writer_tool,
        compile_check_tool,
        ppt_slide_export_tool,
        pptx_com_writer_tool,
        keyword_search_tool,
        image_understanding_tool,
    ]
    system_prompt = (
        "You are a controller agent that converts a research-paper PDF and a figures directory into a Beamer PPT workflow. "
        "The user provides a JSON instructions object that may include pdf_path, figures_dir, template_path, output_path, "
        "export_images, slides_dir, title, subtitle, author, institute, date, lang, max_frames, max_frames_per_section, "
        "appendix_images, and pptx_aspect_ratio. Follow these fields strictly. "
        "Recommended workflow: "
        "1) Use FigureIndexerTool to inspect figures_dir; if images are missing or too few, call PdfFigureExtractTool. "
        "2) Rebuild figure index and optionally use FigureCaptionAlignTool for caption alignment. "
        "3) Use PDFLoaderTool to read source text; optionally use KeywordSearchTool for finer section-level grounding. "
        "4) Plan frames by section with adaptive detail: usually 2-8 frames per section depending on complexity, "
        "while respecting max_frames_per_section (hard cap 10). "
        "5) Build full plan_json (frames + figures), call LatexWriterTool, then CompileCheckTool. "
        "6) If export_images=true and compilation succeeds, call PptSlideExportTool. "
        "Always prioritize factual grounding in source text and existing captions; avoid hallucinated figure captions. "
        "Use ReAct internally when needed, but do not expose Thought/Action/Observation in the final answer."
    )
    checkpointer = InMemorySaver()
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=[
            LoggingMiddleware(),
            # Protect against unbounded model/tool loops in interactive ReAct
            # sessions when this top-level agent is used directly.
            ModelCallLimitMiddleware(thread_limit=16, run_limit=8, exit_behavior="end"),
            # For tool-heavy ReAct flows, automatically retry transient tool
            # failures (e.g., I/O hiccups) a few times with backoff.
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=30.0,
                jitter=True,
            ),
        ],
    )


def build_pipeline_agents(model=None):
    if model is None:
        model = build_model()

    # Reuse a shared, lightweight middleware stack for all specialized
    # pipeline agents so that their behavior stays consistent and observable.
    shared_middlewares = [
        LoggingMiddleware(),
        ModelCallLimitMiddleware(thread_limit=16, run_limit=16, exit_behavior="end"),
    ]

    section_title_system_prompt = (
        "You are an assistant that cleans and normalizes noisy section titles "
        "for a research paper. You receive a list of candidate section titles "
        "extracted heuristically from the PDF together with short text samples "
        "and page ranges. Your job is to map each original title to a concise, "
        "human-readable section title that matches the real presentation-oriented "
        "structure of the paper. Respond ONLY with JSON.\n\n"
        "Output JSON schema:\n"
        '{"normalized_sections": [{"original_section": string, "normalized_section": string }, ... ] }\n'
        "Global rules:\n"
        "- Preserve the coarse order implied by the input; do not invent extra sections or reorder them aggressively.\n"
        "- If an original title is already a clean PDF heading/TOC heading, keep it unchanged; only rewrite noisy or placeholder titles.\n"
        "- If instructions['lang'] is 'zh', write normalized_section in Chinese; otherwise use English.\n"
        "- normalized_section must be a short, presentation-ready heading (ideally <= 15 Chinese chars or <= 60 English chars).\n"
        "- Never return raw data rows, table IDs, parameter codes, or sensor IDs as section titles. When you see noisy titles such as 'kN Bearing2_2 161 2 h 41 outer race', map them to a higher-level conceptual section instead.\n\n"
        "Preferred conceptual buckets (when appropriate):\n"
        "- Background / Introduction (研究背景 / 引言)\n"
        "- Literature Review / Related Work (文献综述 / 相关工作)\n"
        "- Methods / Model (研究方法 / 模型构建)\n"
        "- Data and Sampling (数据来源 / 抽样设计 / 预处理)\n"
        "- Experiments / Results / Evaluation (实验设置 / 结果分析 / 评价指标)\n"
        "- Discussion / Case Analysis (讨论 / 案例分析)\n"
        "- Conclusion / Recommendations / Future Work (结论与建议 / 未来工作)\n"
        "- Other necessary sections such as Abstract, Problem Formulation, References, Acknowledgments.\n"
        "Guidelines:\n"
        "- For numeric-only titles like '3' or '4.2', infer a descriptive title from sample_text.\n"
        "- For author names, affiliations, or grant IDs, do NOT keep them as section titles; map them to the nearest real section inferred from sample_text.\n"
        "- For parameter-table-like titles, dataset IDs, or machine IDs, map to higher-level conceptual headings such as '数据与样本', '实验设计', or '结果分析'.\n"
        "- Do NOT map normal content sections to '目录/Contents' unless the sample_text is clearly a table-of-contents page.\n"
        "- Prefer merging highly similar noisy titles into the SAME normalized_section to keep a compact outline.\n"
        "- Do not force every section into the standard buckets if the sample_text clearly serves a different role (e.g., 'Abstract', 'Problem Formulation', 'References'); in that case, choose an appropriate descriptive title instead.\n\n"
        "Example (zh): map '1 引言' -> '研究背景', map '4.2' -> '实验结果分析', map noisy table id -> '数据与样本'.\n"
        "Example (en): map '3' -> 'Methods', map noisy equipment id -> 'Experimental Setup'.\n"
        "When answering for the real input, follow the exact JSON schema and do not output any prose outside the JSON object.\n"
    )

    section_title_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=section_title_system_prompt,
        middleware=shared_middlewares,
        response_format=ToolStrategy(SectionTitleResponse),
    )

    summary_system_prompt = (
        "You are an assistant that writes a high-quality overview slide for a research paper. "
        "Your ONLY output must be a single JSON object with keys: title, bullets, paragraphs.\n\n"
        "Output requirements:\n"
        "1. Respond with JSON only, no markdown, no code fences, no comments.\n"
        "2. JSON schema: {\"title\": string, \"bullets\": [string], \"paragraphs\": [string]}.\n"
        "3. Use instructions['lang']: if 'zh', write in Chinese; otherwise use English.\n"
        "4. Use ONLY the provided source_text and section_titles; do NOT invent facts or numbers.\n"
        "5. bullets: 3-5 concise points covering motivation, method, data/setting, key findings, and conclusion.\n"
        "6. paragraphs: 1 short paragraph (1-2 sentences) that ties the narrative together; can be empty if bullets suffice.\n"
        "7. title: a short, presentation-ready heading (<= 12 Chinese chars or <= 60 English chars), "
        "   e.g., '研究概述', '核心贡献', 'Overview', or 'Key Contributions'.\n"
    )

    summary_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=summary_system_prompt,
        middleware=shared_middlewares,
        response_format=ToolStrategy(SummaryResponse),
    )

    planner_system_prompt = (
        "You are an assistant that plans Beamer PPT frames for a single logical section of a research paper. "
        "Your ONLY output must be a single JSON object with a key 'frames'.\n\n"
        "Output requirements:\n"
        "1. Respond with JSON only, no markdown, no code fences, no comments.\n"
        "2. The JSON structure must be: {\"frames\": [...]} .\n"
        "   - frames: each item has {section, subsection, frame_title, bullets, paragraphs, figure_slugs}.\n"
        "   - bullets: list of strings; they may contain LaTeX math and should NOT be escaped.\n"
        "   - paragraphs: list of short paragraph strings (also raw LaTeX, not escaped) that express the main narrative content of the slide.\n"
        "   - figure_slugs: list of strings referencing items from section_figures (if any).\n"
        "3. Use instructions['lang']: if 'zh', write bullets and paragraphs in Chinese; if 'en', write them in English.\n"
        "4. Use ONLY the content from source_text, retrieved_context, global_planning_context, and section_formulas in the user input for this section. "
        "   You may translate, compress, and reorganize the wording, but do NOT invent facts or numeric values that are not supported by these texts.\n"
        "4.1 global_planning_context provides full-document storyline and section transitions. "
        "Use it to keep narrative continuity and avoid isolated or contradictory slides.\n"
        "4.2 Respect section_order_index/section_order_total/previous_section/next_section when present, so each section's frames connect naturally to adjacent sections.\n"
        "5. For this section, generate an adaptive number of frames (typically 2-10). "
        "Prefer instructions['target_frames'] when provided; otherwise follow instructions['max_frames_per_section']. "
        "Do not exceed instructions['max_frames_per_section'] and never exceed 10.\n"
        "6. If detail_profile is 'detailed', expand this section with a richer template when evidence exists: "
        "(a) background/problem, (b) method details, (c) data/sampling, (d) core results, "
        "(e) comparative analysis, (f) interpretability/explanatory analysis, (g) conclusion/recommendations. "
        "Use only stages supported by the provided text; do not invent missing evidence.\n"
        "7. Keep each frame compact: 1-2 short paragraphs and/or up to 5 bullets. Avoid long lists.\n"
        "8. frame_title must be clean and presentation-ready. Never copy raw table-of-contents lines, "
        "dot leaders, page-number strings, or long catalog text into frame titles or bullets.\n"
        "9. Every frame's bullets/paragraphs must match its section theme and the global storyline. Do not mix unrelated topics across sections.\n"
        "8. If section_figures is non-empty, include them by referencing figure_slugs. "
        "   You MAY place multiple figure_slugs on the same frame (aim for 1-3 total per frame) and briefly describe them in text. "
        "   Do not invent meanings for figures.\n"
        "10. You MUST NOT write any LaTeX figure environments or \\includegraphics commands directly in bullets or paragraphs. "
        "   When you want to show a figure on a slide, only add its slug (from section_figures[*].slug) to figure_slugs and describe it in text.\n"
        "11. section_formulas contains formula candidates for this section. Include the most central 0-2 formulas. "
        "    If you include a formula, write a short explanation (1-2 sentences) grounded in the provided text.\n"
        "12. All formulas you output MUST use standard ASCII-only LaTeX syntax (e.g., W_{ko}, \\gamma_{kl}, \\mathbf{p}_k). You MUST NEVER emit Unicode "
        "    mathematical symbols copied directly from PDF glyph artifacts. Always rewrite them using plain ASCII "
        "    letters, subscripts (^, _) and LaTeX commands (e.g., \\mathbf{p}, \\dot{p}, \\tau, \\sigma, \\xi, \\top), wrapped in $...$ or \\[...\\].\n"
        "13. For short definition-style statements (e.g., '邻域集合定义为：...', 'The neighborhood is defined as: ...'), strongly prefer the pattern "
        "    'natural-language description: $LaTeX formula$'. The part after the colon must be a compact ASCII LaTeX expression only, without Unicode math.\n"
        "14. Use only the provided source_text, retrieved_context, section_figures, and section_formulas. "
        "    Do NOT call any external tools in this planner stage.\n"
        "15. Keep the output deterministic and compact: one JSON object only, no extra prose.\n"
    )

    planner_tools_enabled = os.getenv("PIPELINE_ENABLE_PLANNER_TOOLS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    planner_tools = [keyword_search_tool, figure_indexer_tool, image_understanding_tool] if planner_tools_enabled else []
    if planner_tools_enabled:
        print("[Pipeline] Planner tools are enabled (PIPELINE_ENABLE_PLANNER_TOOLS=1).")
    else:
        print("[Pipeline] Planner tools are disabled for stability; using provided retrieved_context only.")

    planner_agent = create_agent(
        model=model,
        tools=planner_tools,
        system_prompt=planner_system_prompt,
        middleware=shared_middlewares,
        response_format=ToolStrategy(PlannerResponse),
    )

    latex_fix_system_prompt = (
        "You are an expert LaTeX engineer. "
        "Given a Beamer .tex document and the xelatex compile log, fix the LaTeX source so that it compiles successfully. "
        "You MUST preserve all existing\\includegraphics commands and their arguments exactly as they appear in the original file; "
        "do not change image paths, filenames, or remove any figure environments. "
        "Return ONLY the full corrected LaTeX source code, without any explanations, comments, markdown, or code fences."
    )

    latex_fix_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=latex_fix_system_prompt,
        middleware=shared_middlewares,
    )

    formula_system_prompt = (
        "You are an assistant that extracts mathematical formulas from short blocks of scientific text. "
        "Your ONLY output must be a single JSON object with a key 'formulas'.\n\n"
        "JSON schema:\n"
        '{"formulas": [ {"id": string (optional), "raw_text": string, "latex": string (optional), '
        '"description": string (optional), "page": int (optional), "chunk_index": int (optional) }, ... ] }\n'
        "- raw_text: the original formula text as it appears in the input text.\n"
        "- latex: a cleaned-up LaTeX math version of the same formula (inline or display), if you can infer it. When you write latex, you MUST use "
        "  standard ASCII-only LaTeX syntax (e.g., W_{ko}, \\gamma_{kl}, \\mathbf{p}_k) and you MUST NOT copy Unicode mathematical characters or "
        "  malformed PDF glyphs directly; always rewrite them into plain ASCII LaTeX commands and subscripts/superscripts.\n"
        "If there are no formulas in the input, return {\"formulas\": []}.\n"
    )

    formula_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=formula_system_prompt,
        middleware=shared_middlewares,
        response_format=ToolStrategy(FormulaResponse),
    )

    figure_metadata_system_prompt = (
        "You are an assistant that extracts figure and table captions from short blocks of scientific text. "
        "Your ONLY output must be a single JSON object with a key 'figures'.\n\n"
        "JSON schema:\n"
        '{"figures": [ {"id": string (optional), "kind": "figure" or "table" (optional), '
        '"caption": string, "context": string (optional), "number": int (optional), "page": int (optional) }, ... ] }\n'
        "- caption: the full caption line, usually containing words like 'Figure 1.', 'Fig. 2', 'Table 3', etc.\n"
        "- context: a short nearby text snippet that helps explain the figure or table.\n"
        "If there are no figures or tables in the input, return {\"figures\": []}.\n"
    )

    figure_metadata_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=figure_metadata_system_prompt,
        middleware=shared_middlewares,
        response_format=ToolStrategy(FigureMetadataResponse),
    )

    return {
        "model": model,
        "section_titles": section_title_agent,
        "summary": summary_agent,
        "planner": planner_agent,
        "latex_fix": latex_fix_agent,
        "formula_extractor": formula_agent,
        "figure_metadata": figure_metadata_agent,
    }

