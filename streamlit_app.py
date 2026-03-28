from __future__ import annotations

import argparse
import time
import contextlib
import io
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
import textwrap
import zipfile

import streamlit as st
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from ppt_agent import cli as ppt_cli


def _suppress_websocket_closed_errors() -> None:
    try:
        import asyncio
        from tornado.iostream import StreamClosedError
        from tornado.websocket import WebSocketClosedError
    except Exception:
        return

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return

    if getattr(loop, "_streamlit_ws_filter_installed", False):
        return

    default_handler = loop.get_exception_handler()

    def _handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, (WebSocketClosedError, StreamClosedError)):
            return
        if default_handler is not None:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    setattr(loop, "_streamlit_ws_filter_installed", True)


_suppress_websocket_closed_errors()


def _suppress_missing_script_run_context_warnings() -> None:
    import logging

    class _MissingScriptRunContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "missing ScriptRunContext" in msg:
                return False
            return True

    logger = logging.getLogger("streamlit")
    logger.addFilter(_MissingScriptRunContextFilter())


_suppress_missing_script_run_context_warnings()


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _run_cli_with_config(config: dict, on_log=None) -> dict:
    run_id = config.get("run_id")
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        config["run_id"] = run_id

    env_overrides = config.get("env") or {}
    for key, value in env_overrides.items():
        if value is None:
            continue
        os.environ[str(key)] = str(value)

    def _parse_args_patch() -> argparse.Namespace:
        ns = argparse.Namespace()
        ns.pdf = config["pdf_path"]
        ns.figures = config.get("figures")
        ns.template = config.get("template") or "ppt_template.tex"
        ns.output = config.get("output") or "ppt论文汇报_agent.tex"
        ns.output_dir = config.get("output_dir") or "output"
        ns.run_id = config.get("run_id")
        ns.export_images = bool(config.get("export_images"))
        ns.slides_dir = config.get("slides_dir") or "export/slides"
        ns.title = config.get("title")
        ns.subtitle = config.get("subtitle")
        ns.author = config.get("author")
        ns.institute = config.get("institute")
        ns.date = config.get("date")
        ns.lang = (config.get("lang") or "zh")
        ns.mode = config.get("mode") or "pipeline"
        ns.pptx_template = config.get("pptx_template")
        ns.pptx_aspect_ratio = config.get("pptx_aspect_ratio")
        ns.max_frames = config.get("max_frames") or 58
        ns.report_type = config.get("report_type")
        ns.detail_profile = config.get("detail_profile")
        ns.target_content_pages_min = config.get("target_content_pages_min")
        ns.target_content_pages_max = config.get("target_content_pages_max")
        ns.mojibake_policy = config.get("mojibake_policy")
        ns.preferred_text_source = config.get("preferred_text_source")
        ns.enable_ocr_fallback = bool(config.get("enable_ocr_fallback", True))
        ns.disable_ocr_fallback = not bool(config.get("enable_ocr_fallback", True))
        ns.text_quality_debug = bool((config.get("env") or {}).get("PIPELINE_TEXT_QUALITY_DEBUG"))
        ns.auto_fix_latex = bool(config.get("auto_fix_latex", True))
        ns.max_fix_attempts = config.get("max_fix_attempts") or 2
        ns.pptx_theme_palette = config.get("pptx_theme_palette")
        return ns

    def _interactive_fill_args_patch(args: argparse.Namespace) -> argparse.Namespace:
        return args

    orig_parse_args = ppt_cli.parse_args
    orig_fill_args = ppt_cli._interactive_fill_args

    buf = io.StringIO()
    logs_path = None

    pdf_path_obj = Path(config["pdf_path"]).expanduser().resolve()
    project_name = pdf_path_obj.stem
    project_dir_name = ppt_cli._slugify_project_name(project_name)
    output_root = Path(config.get("output_dir") or "output").expanduser().resolve()
    run_output_dir = output_root / project_dir_name / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_logs_path = run_output_dir / "logs_live.txt"
    try:
        candidate_logs_path.write_text("", encoding="utf-8")
        logs_path = candidate_logs_path
    except Exception:
        logs_path = None

    def _session_is_active() -> bool:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
        except Exception:
            return False
        try:
            ctx = get_script_run_ctx()
        except Exception:
            return False
        if ctx is None:
            return False
        session_id = getattr(ctx, "session_id", None)
        if not session_id:
            return True
        try:
            from streamlit.runtime import get_instance as get_runtime_instance
        except Exception:
            return True
        try:
            runtime = get_runtime_instance()
        except Exception:
            return True
        session_mgr = getattr(runtime, "session_manager", None)
        if session_mgr is None:
            return True
        if hasattr(session_mgr, "get_active_session_info"):
            try:
                return session_mgr.get_active_session_info(session_id) is not None
            except Exception:
                return True
        if hasattr(session_mgr, "is_active_session"):
            try:
                return bool(session_mgr.is_active_session(session_id))
            except Exception:
                return True
        return True

    class _Stream(io.TextIOBase):
        def __init__(self) -> None:
            super().__init__()
            self._last_emit = 0.0
            self._min_interval = 0.2
        @staticmethod
        def _has_streamlit_context() -> bool:
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
            except Exception:
                return False
            return get_script_run_ctx() is not None

        def write(self, s):
            buf.write(s)
            if logs_path:
                try:
                    with open(logs_path, "a", encoding="utf-8") as f:
                        f.write(s)
                except Exception:
                    pass
            if on_log is not None:
                now = datetime.now().timestamp()
                if now - self._last_emit >= self._min_interval:
                    self._last_emit = now
                    try:
                        on_log(buf.getvalue())
                    except Exception:
                        pass
            return len(s)

        def flush(self) -> None:
            pass

    stream = _Stream()

    try:
        ppt_cli.parse_args = _parse_args_patch
        ppt_cli._interactive_fill_args = _interactive_fill_args_patch
        with contextlib.redirect_stdout(stream):
            ppt_cli.main()
    finally:
        ppt_cli.parse_args = orig_parse_args
        ppt_cli._interactive_fill_args = orig_fill_args

    logs = buf.getvalue()

    project_figures_dir = run_output_dir / "figures"
    slides_dir_path = run_output_dir / "slides"

    template_path = str(Path(config.get("template") or "ppt_template.tex").expanduser().resolve())
    output_name = Path(config.get("output") or "ppt论文汇报_agent.tex").name
    output_path = str((run_output_dir / output_name).resolve())

    latex_path = None
    pdf_path = None
    pptx_path = None

    for line in logs.splitlines():
        if "LaTeX file generated at:" in line:
            latex_path = line.split("LaTeX file generated at:", 1)[1].strip()
        if "PDF compiled at:" in line:
            pdf_path = line.split("PDF compiled at:", 1)[1].strip()
        if "PPTX file generated at:" in line:
            pptx_path = line.split("PPTX file generated at:", 1)[1].strip()

    if not latex_path:
        latex_path = output_path
    if not pdf_path:
        pdf_path = str(Path(output_path).with_suffix(".pdf"))
    if not pptx_path:
        pptx_path = str(Path(output_path).with_suffix(".pptx"))

    result = {
        "project_name": project_name,
        "project_dir_name": project_dir_name,
        "output_dir": str(run_output_dir),
        "figures_dir": str(project_figures_dir),
        "slides_dir": str(slides_dir_path),
        "template_path": template_path,
        "output_tex_path": latex_path,
        "output_pdf_path": pdf_path,
        "output_pptx_path": pptx_path,
        "logs": logs,
    }
    return result


def main() -> None:
    if not _running_in_streamlit():
        print("请使用 `streamlit run streamlit_app.py` 启动应用。")
        return

    st.set_page_config(page_title="Paper2slides", page_icon="📊", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        :root {
            --bg: #fff5e1;
            --bg-2: #ffe7c2;
            --card: #ffffff;
            --ink: #0f172a;
            --muted: #475569;
            --accent: #f97316;
            --accent-2: #0ea5a4;
            --stroke: rgba(15, 23, 42, 0.12);
            --shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
        }
        html, body, [class*="css"]  {
            font-family: "Space Grotesk", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 0%, rgba(249, 115, 22, 0.22), transparent 45%),
                radial-gradient(circle at 90% 10%, rgba(14, 165, 164, 0.18), transparent 50%),
                linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
        }
        .block-container {
            padding-top: 3.5rem;
            padding-bottom: 2.5rem;
            max-width: 1150px;
        }
        .hero {
            padding: 1.2rem 1.6rem;
            background: var(--card);
            border-radius: 1.2rem;
            border: 1px solid var(--stroke);
            box-shadow: var(--shadow);
            animation: riseIn 0.6s ease-out;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--ink);
            letter-spacing: 0.02em;
            margin-bottom: 0.3rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: var(--muted);
            margin-bottom: 0.8rem;
        }
        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .hero-tag {
            padding: 0.28rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.15);
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--ink);
            background: linear-gradient(135deg, #fff7ed, #fed7aa);
        }
        .status-card {
            padding: 1.1rem 1.2rem;
            border-radius: 1rem;
            border: 1px solid var(--stroke);
            background: rgba(255, 255, 255, 0.72);
            box-shadow: var(--shadow);
            animation: riseIn 0.8s ease-out;
        }
        .status-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--ink);
            margin-bottom: 0.5rem;
        }
        .status-list {
            margin: 0;
            padding-left: 1.2rem;
            color: var(--muted);
            font-size: 0.85rem;
        }
        .stForm {
            background-color: var(--card);
            padding: 1.8rem 2rem;
            border-radius: 1.1rem;
            border: 1px solid var(--stroke);
            box-shadow: var(--shadow);
            animation: riseIn 0.9s ease-out;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--ink);
            margin-top: 1.4rem;
            margin-bottom: 0.7rem;
        }
        .section-subtitle {
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: rgba(71, 85, 105, 0.75);
            margin-top: 1.2rem;
        }
        .stButton>button {
            border-radius: 999px;
            padding: 0.7rem 1.8rem;
            font-weight: 600;
            border: none;
            background: linear-gradient(135deg, var(--accent), #fb923c);
            color: #ffffff;
            box-shadow: 0 12px 30px rgba(249, 115, 22, 0.3);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #ea580c, #fb923c);
        }
        .stDownloadButton>button {
            border-radius: 999px;
            padding: 0.55rem 1.3rem;
            font-weight: 500;
            border: 1px solid rgba(148, 163, 184, 0.6);
            background: #ffffff;
        }
        div[data-testid="stFileUploader"] {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.16), rgba(14, 165, 164, 0.12));
            border-radius: 1rem;
            padding: 0.7rem 0.8rem 1.1rem 0.8rem;
            border: 1px dashed rgba(249, 115, 22, 0.45);
        }
        div[data-testid="stFileUploader"] section {
            background-color: rgba(255, 255, 255, 0.96);
            border-radius: 0.85rem;
        }
        .stTextInput>div>div>input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox>div>div {
            background-color: #fff7ed;
            border-radius: 0.7rem;
        }
        .stCheckbox>label {
            background-color: rgba(254, 215, 170, 0.6);
            padding: 0.18rem 0.7rem;
            border-radius: 999px;
        }
        div[data-testid="stExpander"] {
            border-radius: 1rem;
            border: 1px solid var(--stroke);
            background: #ffffff;
            box-shadow: var(--shadow);
        }
        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 1rem;
            border: 1px solid rgba(14, 165, 164, 0.35);
            background: rgba(255, 255, 255, 0.9);
            margin-top: 1rem;
        }
        .result-path {
            font-family: "JetBrains Mono", ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.78rem;
            color: #0f766e;
            word-break: break-all;
        }
        @keyframes riseIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 900px) {
            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 2rem;
            }
            .hero-title {
                font-size: 1.7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-title">Paper2slides</div>
                <div class="hero-subtitle">面向论文汇报与项目答辩的 PDF 到幻灯片生成工作台</div>
                <div class="hero-tags">
                    <span class="hero-tag">文档解析</span>
                    <span class="hero-tag">图文对齐</span>
                    <span class="hero-tag">LaTeX / PPTX 输出</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown(
            """
            <div class="status-card">
                <div class="status-title">运行提示</div>
                <ul class="status-list">
                    <li>支持 PDF / DOCX / TXT</li>
                    <li>多文件自动合并 PDF</li>
                    <li>结果落盘到 output 子目录</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_logs" not in st.session_state:
        st.session_state["last_logs"] = ""
    if "last_zip_path" not in st.session_state:
        st.session_state["last_zip_path"] = None

    last_result = st.session_state.get("last_result")
    last_logs = st.session_state.get("last_logs") or ""
    last_zip_path = st.session_state.get("last_zip_path")

    if last_result:
        st.markdown("<div class='section-title'>上一次运行结果</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-card">
                <div class="section-title">项目会话目录</div>
                <div class="result-path">{last_result.get('output_dir')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        last_tex = Path(last_result.get("output_tex_path", "")).expanduser().resolve()
        last_pdf = Path(last_result.get("output_pdf_path", "")).expanduser().resolve()
        last_pptx = Path(last_result.get("output_pptx_path", "")).expanduser().resolve()
        last_slides = Path(last_result.get("slides_dir", "")).expanduser().resolve()
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        if last_pptx.is_file():
            with open(last_pptx, "rb") as f:
                dl_col1.download_button("下载上次 PPTX", f, file_name=last_pptx.name)
        if last_pdf.is_file():
            with open(last_pdf, "rb") as f:
                dl_col2.download_button("下载上次 PDF", f, file_name=last_pdf.name)
        if last_tex.is_file():
            with open(last_tex, "rb") as f:
                dl_col3.download_button("下载上次 LaTeX", f, file_name=last_tex.name)
        if last_zip_path:
            zip_path_obj = Path(last_zip_path).expanduser().resolve()
            if zip_path_obj.is_file():
                with open(zip_path_obj, "rb") as f:
                    st.download_button(
                        "下载上次全部结果 ZIP",
                        data=f,
                        file_name=zip_path_obj.name,
                        mime="application/zip",
                    )
        if last_slides.is_dir():
            images = sorted(last_slides.glob("*.png"))
            if images:
                st.subheader("上次幻灯片预览（最多显示前 5 张）")
                for img in images[:5]:
                    st.image(str(img), caption=img.name)

        log_expander = st.expander("上一次运行日志", expanded=True)
        log_expander.code(last_logs or "", language="bash")

    with st.form("config_form"):
        tabs = st.tabs(["基础配置", "高级与资源"])
        with tabs[0]:
            st.markdown("<div class='section-title'>1. 上传与任务配置</div>", unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "上传文档（支持 PDF / Word / TXT，可多选）",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="input_files",
            )
            st.caption("可选择合并多文档为一个 PDF；默认使用主文档名称作为输出目录名。")
            merge_files = False
            primary_file_name = None
            if uploaded_files:
                file_names = [f.name for f in uploaded_files]
                if len(uploaded_files) > 1:
                    merge_files = st.checkbox("合并多文档为一个 PDF", value=True)
                primary_file_name = st.selectbox(
                    "主文档（用于命名输出目录）",
                    options=file_names,
                    index=len(file_names) - 1,
                )
            top_col1, top_col2 = st.columns(2)
            with top_col1:
                lang = st.selectbox("PPT 语言", ["zh", "en"], index=0)
            with top_col2:
                mode = st.selectbox("执行模式", ["pipeline", "controller"], index=0)

            st.markdown("<div class='section-title'>2. PPT 元信息</div>", unsafe_allow_html=True)
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                title = st.text_input("标题")
                author = st.text_input("作者")
                date = st.text_input("日期")
            with meta_col2:
                subtitle = st.text_input("副标题")
                institute = st.text_input("单位")

            st.markdown("<div class='section-title'>3. 生成参数</div>", unsafe_allow_html=True)
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                max_frames = st.number_input("最大内容页数", min_value=10, max_value=200, value=120, step=1)
                export_images = st.checkbox("编译后导出每页 PPT 图片", value=False)
                auto_fix_latex = st.checkbox("自动修复 LaTeX 编译错误", value=True)
                max_fix_attempts = st.number_input("自动修复最大次数", min_value=0, max_value=5, value=2, step=1)
                detail_profile = st.selectbox(
                    "详尽度档位",
                    ["detailed", "standard", "compact"],
                    index=0,
                    help="默认 detailed：目标内容页 25-35。",
                )
                report_template_label = st.selectbox(
                    "报告模板",
                    ["通用报告", "正大杯报告"],
                    index=0,
                    help="正大杯报告会固定章节为：背景分析、文献综述、调查框架设计、访谈分析、网络舆情分析、问卷分析（多模型按模型拆分章节）。",
                )
                report_type = "zhengda_cup" if report_template_label == "正大杯报告" else "general"
                mojibake_policy = st.selectbox(
                    "乱码策略",
                    ["auto_fallback", "strict_fail", "warn_only"],
                    index=0,
                    help="auto_fallback：自动重抽并降级（推荐）。",
                )
                enable_ocr_fallback = st.checkbox("启用 OCR 文本兜底", value=True)
            with param_col2:
                template_path = st.text_input("LaTeX 模板路径", value="ppt_template.tex")
                pptx_template = st.text_input("PPTX 模板路径（可选）", value="")
                pptx_aspect_ratio = st.selectbox(
                    "PPTX 页面比例",
                    ["16:9", "16:10", "16:8", "4:3", "1:1", "auto"],
                    index=0,
                    help="auto 表示使用模板或默认比例。",
                )
                slides_root = st.text_input("幻灯片图片导出目录", value="export/slides")
                max_frames_per_section = st.number_input(
                    "每章节页数上限（2-10）",
                    min_value=2,
                    max_value=10,
                    value=8,
                    step=1,
                )
                default_min = 25 if detail_profile == "detailed" else (18 if detail_profile == "standard" else 12)
                default_max = 35 if detail_profile == "detailed" else (26 if detail_profile == "standard" else 18)
                target_content_pages_min = st.number_input(
                    "目标内容页下限",
                    min_value=6,
                    max_value=200,
                    value=default_min,
                    step=1,
                )
                target_content_pages_max = st.number_input(
                    "目标内容页上限",
                    min_value=int(target_content_pages_min),
                    max_value=240,
                    value=max(default_max, int(target_content_pages_min)),
                    step=1,
                )
                enable_appendix_images = st.checkbox("生成附录图页", value=False)

        with tabs[1]:
            st.markdown("<div class='section-subtitle'>模板与资源</div>", unsafe_allow_html=True)
            pptx_engine = st.selectbox(
                "PPTX 引擎",
                ["PowerPoint COM（高级）", "python-pptx（基础）"],
                index=0,
                help="COM 仅限 Windows 且需安装 PowerPoint。",
            )
            pptx_template_file = st.file_uploader(
                "上传 PPTX 模板文件（可选）",
                type=["pptx"],
                key="pptx_template_file",
            )
            pptx_theme_palette = st.text_input(
                "PPTX 主题色（逗号分隔）",
                value="#34dbcb, #34c2db, #3498db, #346edb, #3445db",
            )
            style_preset = st.selectbox(
                "PPTX style preset",
                ["auto", "tech", "academic", "business", "blackgold"],
                index=0,
                help="Preset style overrides palette/fonts when custom palette is empty.",
            )
            header_footer = st.checkbox("Header/footer (page number + section)", value=True)
            header_logo = st.text_input("Header logo path", value="")
            header_author = st.text_input("Header author", value=author or "")
            header_date = st.text_input("Header date", value=date or "")
            header_right = st.text_input("Header right text", value="")
            image_anim_effect = st.selectbox(
                "Image animation effect",
                ["fly_from_right", "fly_from_left", "fly_from_bottom", "fade", "appear"],
                index=0,
            )
            use_external_figures = st.checkbox("使用已有 figures 目录", value=False)
            figures_root_text = ""
            if use_external_figures:
                figures_root_text = st.text_input("figures 目录（可选）", value="./figures")
            figures_files = st.file_uploader(
                "上传 figures 资源（图片或 ZIP，可选）",
                type=["png", "jpg", "jpeg", "pdf", "svg", "zip"],
                accept_multiple_files=True,
                key="figures_files",
            )
            st.caption("若未上传 figures 且未启用外部目录，将仅使用本次上传文档解析得到的图像。")

            st.markdown("<div class='section-subtitle'>PPT 背景设置</div>", unsafe_allow_html=True)
            bg_mode = st.selectbox(
                "背景模式",
                ["继承模板", "上传背景图", "AI 生成背景", "无背景"],
                index=0,
            )
            bg_image_file = None
            bg_prompt = ""
            bg_model = ""
            bg_size = ""
            bg_quality = ""
            bg_format = ""
            bg_base_url = ""
            bg_api_key = ""
            if bg_mode == "上传背景图":
                bg_image_file = st.file_uploader(
                    "上传背景图片（建议 16:9）",
                    type=["png", "jpg", "jpeg"],
                    key="pptx_bg_image",
                )
            elif bg_mode == "AI 生成背景":
                bg_prompt = st.text_area(
                    "背景生成提示词",
                    value="Abstract tech gradient background, clean and minimal, blue and cyan palette",
                )
                bg_model = st.text_input("AI 背景模型", value="gpt-image-1")
                bg_size = st.selectbox(
                    "AI 背景尺寸",
                    ["1536x1024", "1024x1024", "1024x1536", "auto"],
                    index=0,
                )
                bg_quality = st.selectbox("AI 背景质量", ["low", "medium", "high", "auto"], index=0)
                bg_format = st.selectbox("AI 背景格式", ["png", "jpeg"], index=0)
                bg_base_url = st.text_input("AI 背景 API Base URL", value="https://api.apiyi.com/v1")
                bg_api_key = st.text_input("AI 背景 API Key（可选）", value="", type="password")

            st.markdown("<div class='section-subtitle'>布局与渲染</div>", unsafe_allow_html=True)
            layout_from_template = st.checkbox(
                "根据模板背景自动避让布局",
                value=True,
                help="启用后分析模板背景，尽量避开 logo/底部装饰区域。",
            )
            render_latex = st.checkbox(
                "将数学公式渲染为图片",
                value=True,
                help="识别 $...$ 或 \\( ... \\) 公式并渲染为图片插入。",
            )
            fill_image_on_empty = st.checkbox(
                "连续2页无文字时自动生成配图",
                value=True,
                help="基于当前图片风格生成插图，避免空白页。",
            )
            template_replace_only = st.checkbox(
                "模板保真填充（仅替换文本/图片）",
                value=True,
                help="启用后使用 Win32 COM 保留模板边框、配色、装饰，只替换模板文本框和图片框内容。",
            )
            rep_col1, rep_col2 = st.columns(2)
            with rep_col1:
                template_start_slide = st.number_input(
                    "模板起始页",
                    min_value=1,
                    max_value=50,
                    value=1,
                    step=1,
                    help="从第几页模板开始写入内容。",
                )
            with rep_col2:
                template_base_slide = st.number_input(
                    "模板复用页",
                    min_value=1,
                    max_value=50,
                    value=1,
                    step=1,
                    help="当页数不够时，复制这页模板继续填充。",
                )
            template_trim_unused = st.checkbox(
                "删除未使用模板尾页",
                value=True,
                help="启用后会删除写入范围之外的模板尾页。",
            )

            st.markdown("<div class='section-subtitle'>输出与缓存</div>", unsafe_allow_html=True)
            output_dir = st.text_input("输出目录", value="output")
            run_id_input = st.text_input("运行编号（可选，留空自动生成）", value="")
            use_cache = st.checkbox("使用缓存（降低重复请求）", value=True)
            parallel_extract = st.checkbox("并行解析图表", value=True)
            conc_col1, conc_col2 = st.columns(2)
            with conc_col1:
                max_concurrency = st.number_input("Pipeline 并发度", min_value=1, max_value=16, value=4, step=1)
            with conc_col2:
                llm_concurrency = st.number_input("LLM 并发度", min_value=1, max_value=16, value=4, step=1)

            with st.expander("高级稳定性设置（防卡死）", expanded=False):
                stab_col1, stab_col2 = st.columns(2)
                with stab_col1:
                    section_norm_batch = st.number_input("标题规范化批大小", min_value=1, max_value=50, value=20, step=1)
                    section_title_timeout = st.number_input("标题规范化超时(秒)", min_value=0, max_value=600, value=180, step=10)
                    summary_timeout = st.number_input("摘要生成超时(秒)", min_value=0, max_value=600, value=180, step=10)
                    planner_timeout = st.number_input("单节规划超时(秒)", min_value=0, max_value=600, value=180, step=10)
                with stab_col2:
                    planner_batch_timeout = st.number_input("批量规划超时(秒)", min_value=0, max_value=900, value=300, step=10)
                    figure_batch_timeout = st.number_input("图表抽取批量超时(秒)", min_value=0, max_value=900, value=300, step=10)
                    formula_batch_timeout = st.number_input("公式抽取批量超时(秒)", min_value=0, max_value=900, value=300, step=10)
                    disable_vector_rag = st.checkbox("禁用向量检索（仅用关键词）", value=False)

                drop_back_matter = st.checkbox(
                    "过滤附录/参考文献/致谢等非正文",
                    value=True,
                    help="启用后在分块和规划阶段过滤非正文内容。",
                )
                text_quality_debug = st.checkbox("输出文本质量调试日志", value=True)

                emb_col1, emb_col2 = st.columns(2)
                with emb_col1:
                    embedding_timeout = st.number_input("Embedding 超时(秒, 0=默认)", min_value=0, max_value=300, value=0, step=5)
                with emb_col2:
                    embedding_retries = st.number_input("Embedding 重试次数", min_value=1, max_value=5, value=1, step=1)
                    embedding_backoff = st.number_input("Embedding 重试间隔(秒)", min_value=0, max_value=10, value=1, step=1)

        submitted = st.form_submit_button("开始生成", use_container_width=True)

    if not submitted:
        return
    if not uploaded_files:
        st.error("请先上传至少一个文档（PDF / Word / TXT）。")
        return

    run_id_value = run_id_input.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_root = Path("web_uploads").expanduser().resolve()
    upload_dir = upload_root / run_id_value
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_paths: list[Path] = []
    selected_files = list(uploaded_files)
    primary_file = None
    if uploaded_files:
        if primary_file_name:
            for uf in uploaded_files:
                if uf.name == primary_file_name:
                    primary_file = uf
                    break
        if primary_file is None:
            primary_file = uploaded_files[-1]
        if not merge_files:
            selected_files = [primary_file]
    for uf in selected_files:
        raw_path = upload_dir / uf.name
        with open(raw_path, "wb") as f:
            f.write(uf.read())
        doc_paths.append(raw_path)

    preferred_text_source = None
    if primary_file is not None:
        candidate = upload_dir / primary_file.name
        if candidate.exists():
            preferred_text_source = str(candidate)

    pdf_paths: list[Path] = []
    for path in doc_paths:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pdf_paths.append(path)
        elif suffix == ".docx":
            out_path = path.with_suffix(".pdf")
            try:
                doc = Document(str(path))
                text = "\n".join(p.text for p in doc.paragraphs)
                c = canvas.Canvas(str(out_path), pagesize=letter)
                width, height = letter
                margin = 72
                y = height - margin
                c.setFont("Helvetica", 10)
                for para in text.split("\n\n"):
                    for line in textwrap.wrap(para, width=90):
                        if y < margin:
                            c.showPage()
                            c.setFont("Helvetica", 10)
                            y = height - margin
                        c.drawString(margin, y, line)
                        y -= 14
                    y -= 10
                c.save()
                pdf_paths.append(out_path)
            except Exception:
                continue
        elif suffix == ".txt":
            out_path = path.with_suffix(".pdf")
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    content = path.read_text(encoding="gbk", errors="ignore")
                except Exception:
                    content = ""
            if not content.strip():
                continue
            c = canvas.Canvas(str(out_path), pagesize=letter)
            width, height = letter
            margin = 72
            y = height - margin
            c.setFont("Helvetica", 10)
            for para in content.split("\n\n"):
                for line in textwrap.wrap(para, width=90):
                    if y < margin:
                        c.showPage()
                        c.setFont("Helvetica", 10)
                        y = height - margin
                    c.drawString(margin, y, line)
                    y -= 14
                y -= 10
            c.save()
            pdf_paths.append(out_path)

    if not pdf_paths:
        st.error("上传文件无法转换为 PDF，请确认文件类型是否为 PDF / DOCX / TXT。")
        return

    from pypdf import PdfReader, PdfWriter

    if len(pdf_paths) == 1:
        final_pdf_path = pdf_paths[0]
    else:
        primary_stem = (
            Path(primary_file.name).stem if primary_file is not None else pdf_paths[0].stem
        )
        writer = PdfWriter()
        for p in pdf_paths:
            try:
                reader = PdfReader(str(p))
            except Exception:
                continue
            for page in reader.pages:
                writer.add_page(page)
        combined_path = upload_dir / f"{primary_stem}_combined.pdf"
        with open(combined_path, "wb") as f:
            writer.write(f)
        final_pdf_path = combined_path

    pdf_temp_path = final_pdf_path

    figures_arg = None
    if figures_files:
        fig_root = upload_dir / "figures"
        fig_root.mkdir(parents=True, exist_ok=True)
        figures_upload_dir = fig_root / pdf_temp_path.stem
        figures_upload_dir.mkdir(parents=True, exist_ok=True)
        for uf in figures_files:
            data = uf.read()
            name = uf.name
            suffix = Path(name).suffix.lower()
            if suffix == ".zip":
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        zf.extractall(figures_upload_dir)
                except Exception:
                    continue
            else:
                target = figures_upload_dir / name
                try:
                    with open(target, "wb") as out_f:
                        out_f.write(data)
                except Exception:
                    continue
        figures_arg = str(figures_upload_dir)
    elif use_external_figures and figures_root_text:
        figures_arg = figures_root_text

    pptx_template_arg = None
    if pptx_template_file is not None:
        tpl_root = upload_dir / "pptx_templates"
        tpl_root.mkdir(parents=True, exist_ok=True)
        tpl_path = tpl_root / pptx_template_file.name
        try:
            with open(tpl_path, "wb") as f:
                f.write(pptx_template_file.read())
            pptx_template_arg = str(tpl_path)
        except Exception:
            pptx_template_arg = None
    elif pptx_template:
        pptx_template_arg = pptx_template

    bg_image_path = None
    if bg_image_file is not None:
        bg_root = upload_dir / "backgrounds"
        bg_root.mkdir(parents=True, exist_ok=True)
        bg_path = bg_root / bg_image_file.name
        try:
            with open(bg_path, "wb") as f:
                f.write(bg_image_file.read())
            bg_image_path = str(bg_path)
        except Exception:
            bg_image_path = None

    config = {
        "pdf_path": str(pdf_temp_path),
        "figures": figures_arg,
        "template": template_path or "ppt_template.tex",
        "output": "ppt论文汇报_agent.tex",
        "output_dir": output_dir or "output",
        "run_id": run_id_value,
        "export_images": export_images,
        "slides_dir": slides_root or "export/slides",
        "title": title or None,
        "subtitle": subtitle or None,
        "author": author or None,
        "institute": institute or None,
        "date": date or None,
        "lang": lang,
        "mode": mode,
        "report_type": report_type,
        "pptx_template": pptx_template_arg,
        "pptx_aspect_ratio": pptx_aspect_ratio,
        "pptx_theme_palette": pptx_theme_palette or None,
        "max_frames": int(max_frames),
        "auto_fix_latex": auto_fix_latex,
        "max_fix_attempts": int(max_fix_attempts),
        "detail_profile": detail_profile,
        "target_content_pages_min": int(target_content_pages_min),
        "target_content_pages_max": int(target_content_pages_max),
        "mojibake_policy": mojibake_policy,
        "enable_ocr_fallback": bool(enable_ocr_fallback),
        "preferred_text_source": preferred_text_source,
        "env": {
            "PIPELINE_USE_CACHE": "1" if use_cache else "0",
            "PIPELINE_PARALLEL_EXTRACT": "1" if parallel_extract else "0",
            "PIPELINE_MAX_CONCURRENCY": str(int(max_concurrency)),
            "LLM_MAX_CONCURRENCY": str(int(llm_concurrency)),
            "PIPELINE_MAX_FRAMES_PER_SECTION": str(int(max_frames_per_section)),
            "PIPELINE_REPORT_TYPE": report_type,
            "PIPELINE_DETAIL_PROFILE": detail_profile,
            "PIPELINE_TARGET_CONTENT_PAGES_MIN": str(int(target_content_pages_min)),
            "PIPELINE_TARGET_CONTENT_PAGES_MAX": str(int(target_content_pages_max)),
            "PIPELINE_MOJIBAKE_POLICY": mojibake_policy,
            "PIPELINE_ENABLE_OCR_FALLBACK": "1" if enable_ocr_fallback else "0",
            "PIPELINE_TEXT_QUALITY_DEBUG": "1" if text_quality_debug else "0",
            "PIPELINE_PREFERRED_TEXT_SOURCE": preferred_text_source,
            "PIPELINE_ENABLE_APPENDIX_IMAGES": "1" if enable_appendix_images else "0",
            "PIPELINE_SECTION_NORMALIZATION_BATCH_SIZE": str(int(section_norm_batch)),
            "PIPELINE_SECTION_TITLE_TIMEOUT": str(int(section_title_timeout)),
            "PIPELINE_SUMMARY_TIMEOUT": str(int(summary_timeout)),
            "PIPELINE_PLANNER_TIMEOUT": str(int(planner_timeout)),
            "PIPELINE_PLANNER_BATCH_TIMEOUT": str(int(planner_batch_timeout)),
            "PIPELINE_FIGURE_BATCH_TIMEOUT": str(int(figure_batch_timeout)),
            "PIPELINE_FORMULA_BATCH_TIMEOUT": str(int(formula_batch_timeout)),
            "PIPELINE_DISABLE_VECTOR_RAG": "1" if disable_vector_rag else "0",
            "PIPELINE_DROP_BACK_MATTER": "1" if drop_back_matter else "0",
            "EMBEDDING_TIMEOUT": (str(int(embedding_timeout)) if embedding_timeout else None),
            "EMBEDDING_RETRIES": str(int(embedding_retries)),
            "EMBEDDING_RETRY_BACKOFF": str(int(embedding_backoff)),
            "PPTX_ENGINE": "com" if "COM" in pptx_engine else "python",
            "PPTX_BACKGROUND_MODE": (
                "inherit"
                if bg_mode == "继承模板"
                else "upload"
                if bg_mode == "上传背景图"
                else "ai"
                if bg_mode == "AI 生成背景"
                else "none"
            ),
            "PPTX_BACKGROUND_IMAGE": bg_image_path,
            "PPTX_BACKGROUND_PROMPT": bg_prompt or None,
            "IMAGE_MODEL": bg_model or None,
            "IMAGE_SIZE": bg_size or None,
            "IMAGE_QUALITY": bg_quality or None,
            "IMAGE_OUTPUT_FORMAT": bg_format or None,
            "IMAGE_BASE_URL": bg_base_url or None,
            "IMAGE_API_KEY": bg_api_key or None,
            "PPTX_LAYOUT_FROM_TEMPLATE": "1" if layout_from_template else "0",
            "PPTX_RENDER_LATEX": "1" if render_latex else "0",
            "PPTX_FILL_IMAGE_ON_EMPTY": "1" if fill_image_on_empty else "0",
            "PPTX_TEMPLATE_REPLACE_ONLY": "1" if template_replace_only else "0",
            "PPTX_TEMPLATE_START_SLIDE": str(int(template_start_slide)),
            "PPTX_TEMPLATE_CONTENT_BASE_SLIDE": str(int(template_base_slide)),
            "PPTX_TEMPLATE_TRIM_UNUSED": "1" if template_trim_unused else "0",
            "PPTX_TEMPLATE_BODY_SLOTS": "3",
            "PPTX_TEMPLATE_BODY_MAX_CHARS": "640",
            "PPTX_TEMPLATE_CLEAR_UNFILLED": "0",
            "PPTX_STYLE_PRESET": ("" if style_preset == "auto" else style_preset),
            "PPTX_HEADER_FOOTER": "1" if header_footer else "0",
            "PPTX_IMAGE_ANIMATION_EFFECT": image_anim_effect,
            "PPTX_HEADER_LOGO": header_logo or None,
            "PPTX_HEADER_AUTHOR": header_author or None,
            "PPTX_HEADER_DATE": header_date or None,
            "PPTX_HEADER_RIGHT": header_right or None,
        },
    }

    status_placeholder = st.empty()
    log_expander = st.expander("实时运行日志", expanded=True)
    log_placeholder = log_expander.empty()

    result_holder = {"result": None}
    latest_log = {"text": ""}
    logs_live_path = None

    # Precompute logs_live_path to improve real-time log visibility
    try:
        run_id = config.get("run_id")
        pdf_path_obj = Path(config["pdf_path"]).expanduser().resolve()
        project_name = pdf_path_obj.stem
        project_dir_name = ppt_cli._slugify_project_name(project_name)
        output_root = Path(config.get("output_dir") or "output").expanduser().resolve()
        run_output_dir = output_root / project_dir_name / run_id
        candidate = run_output_dir / "logs_live.txt"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        if not candidate.exists():
            try:
                candidate.write_text("", encoding="utf-8")
            except Exception:
                pass
        logs_live_path = candidate
    except Exception:
        logs_live_path = None

    def _on_log(text: str) -> None:
        latest_log["text"] = text or ""

    def _run_job() -> None:
        try:
            result_holder["result"] = _run_cli_with_config(config, on_log=_on_log)
        except Exception as exc:
            result_holder["result"] = {"error": str(exc), "logs": latest_log.get("text", "")}

    with st.spinner("正在执行 PDF→PPT 管道，请稍候..."):
        worker = threading.Thread(target=_run_job, daemon=True)
        worker.start()
        start_ts = datetime.now().timestamp()
        while worker.is_alive():
            elapsed = int(datetime.now().timestamp() - start_ts)
            status_placeholder.info(f"运行中… {elapsed}s（保持连接中）")
            # Refresh logs from live file if available
            if logs_live_path is None:
                try:
                    run_id = config.get("run_id")
                    pdf_path_obj = Path(config["pdf_path"]).expanduser().resolve()
                    project_name = pdf_path_obj.stem
                    project_dir_name = ppt_cli._slugify_project_name(project_name)
                    output_root = Path(config.get("output_dir") or "output").expanduser().resolve()
                    run_output_dir = output_root / project_dir_name / run_id
                    candidate = run_output_dir / "logs_live.txt"
                    if candidate.exists():
                        logs_live_path = candidate
                except Exception:
                    logs_live_path = None
            if logs_live_path and logs_live_path.exists():
                try:
                    content = logs_live_path.read_text(encoding="utf-8", errors="ignore")
                    if content:
                        latest_log["text"] = content
                except Exception:
                    pass
            log_placeholder.code(latest_log["text"] or "等待日志输出...", language="bash")
            time.sleep(0.5)
        worker.join()

        result = result_holder["result"]
        if result is None:
            status_placeholder.error("Run failed: no result. Check logs and retry.")
            return
        if isinstance(result, dict) and result.get("error"):
            status_placeholder.error("Run failed: " + str(result.get("error")))
            logs = result.get("logs") or ""
            if logs:
                log_placeholder.code(logs, language="bash")
            return

        status_placeholder.success("Generation complete")

    logs = result["logs"]
    log_placeholder.code(logs or "", language="bash")

    tex_path = Path(result["output_tex_path"]).expanduser().resolve()
    pdf_path = Path(result["output_pdf_path"]).expanduser().resolve()
    pptx_path = Path(result["output_pptx_path"]).expanduser().resolve()

    run_output_dir = Path(result["output_dir"]).expanduser().resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = run_output_dir / "logs.txt"
    try:
        logs_path.write_text(logs or "", encoding="utf-8")
    except Exception:
        pass

    st.markdown(
        f"""
        <div class="result-card">
            <div class="section-title">输出目录</div>
            <div class="result-path">{result['output_dir']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    if pptx_path.is_file():
        with open(pptx_path, "rb") as f:
            col_a.download_button("下载 PPTX", f, file_name=pptx_path.name)
    if pdf_path.is_file():
        with open(pdf_path, "rb") as f:
            col_b.download_button("下载 PDF", f, file_name=pdf_path.name)
    if tex_path.is_file():
        with open(tex_path, "rb") as f:
            col_c.download_button("下载 LaTeX 源文件", f, file_name=tex_path.name)

    slides_dir = Path(result["slides_dir"]).expanduser().resolve()
    figures_dir = run_output_dir / "figures"
    figures_selected_dir = run_output_dir / "figures_selected"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if pptx_path.is_file():
            zf.write(pptx_path, arcname=pptx_path.name)
        if pdf_path.is_file():
            zf.write(pdf_path, arcname=pdf_path.name)
        if tex_path.is_file():
            zf.write(tex_path, arcname=tex_path.name)
        if slides_dir.is_dir():
            for img in sorted(slides_dir.glob("*.png")):
                zf.write(img, arcname=f"slides/{img.name}")
        if figures_dir.is_dir():
            for img in sorted(figures_dir.rglob("*")):
                if img.is_file():
                    rel = img.relative_to(figures_dir)
                    zf.write(img, arcname=f"figures/{rel.as_posix()}")
        if figures_selected_dir.is_dir():
            for img in sorted(figures_selected_dir.rglob("*")):
                if img.is_file():
                    rel = img.relative_to(figures_selected_dir)
                    zf.write(img, arcname=f"figures_selected/{rel.as_posix()}")
        if logs:
            zf.writestr("logs.txt", logs)
    zip_buffer.seek(0)

    zip_output_path = run_output_dir / f"{result['project_name']}_outputs.zip"
    try:
        zip_output_path.write_bytes(zip_buffer.getvalue())
    except Exception:
        zip_output_path = None

    st.session_state["last_result"] = result
    st.session_state["last_logs"] = logs or ""
    st.session_state["last_zip_path"] = str(zip_output_path) if zip_output_path else None

    st.download_button(
        "下载全部结果 ZIP",
        data=zip_buffer,
        file_name=f"{result['project_name']}_outputs.zip",
        mime="application/zip",
    )

    if slides_dir.is_dir():
        images = sorted(slides_dir.glob("*.png"))
        if images:
            st.subheader("幻灯片预览（最多显示前 5 张）")
            for img in images[:5]:
                st.image(str(img), caption=img.name)


if __name__ == "__main__":
    main()


