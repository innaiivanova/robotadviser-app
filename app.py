# app.py — Robot Adviser (BART + optional soft prompt) with Concise & Controlled Style Guide
import os
import re
import json
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Try PEFT (for soft prompts / adapters)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# ===== Config =====
BASE_MODEL_ID = "facebook/bart-large-cnn"      # summarizer backbone
ADAPTER_ZIP   = "softprompt_adapter.zip"       # optional; place next to app.py
SHOW_PRODUCT_IMAGE = False                     # set True if you have reliable image URLs
MAX_TARGET = 180                               # rich paragraph while staying <200 words

st.set_page_config(page_title="Robot Adviser", page_icon="🤖", layout="wide")

# ===== Style Guide (Concise & Controlled – kept for reference) =====
STYLE_GUIDE = (
  "Write ONE coherent paragraph (150–180 words) for shoppers. "
  "Start with the exact category name followed by a colon. "
  "Cover in order: (1) the top three products and how they differ, "
  "(2) the most common complaints across the category, "
  "(3) the single worst product and why to avoid it. "
  "Use complete sentences, neutral tone, and varied transitions (e.g., ‘However’, ‘In contrast’, ‘Overall’). "
  "Do NOT use bullets, lists, or headings. Do NOT invent facts or mention other categories. "
  "If a detail is missing in the Article, omit it—don’t guess. "
  "Avoid repeated phrases and marketing superlatives."
)

# ===== Utilities =====
def show_image_safe(src, caption=None, width=None):
    try:
        st.image(src, caption=caption, width=width)
    except Exception:
        try:
            st.image(src, width=width)
        except Exception:
            pass

@st.cache_data(show_spinner=False)
def load_df_from_path_or_upload(path_text, upload):
    if upload is not None:
        name = upload.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(upload), upload.name
        if name.endswith(".zip"):
            with zipfile.ZipFile(upload) as z:
                csvs = [f for f in z.namelist() if f.lower().endswith(".csv")]
                if not csvs:
                    st.error("No CSV found inside the uploaded ZIP.")
                    return None, None
                target = csvs[0]
                with z.open(target) as f:
                    return pd.read_csv(f), target
        st.error("Please upload a .csv or .zip file.")
        return None, None

    if path_text and Path(path_text).exists():
        if path_text.lower().endswith(".csv"):
            return pd.read_csv(path_text), path_text
        if path_text.lower().endswith(".zip"):
            with zipfile.ZipFile(path_text) as z:
                csvs = [f for f in z.namelist() if f.lower().endswith(".csv")]
                if not csvs:
                    st.error("No CSV found inside the ZIP path.")
                    return None, None
                target = csvs[0]
                with z.open(target) as f:
                    return pd.read_csv(f), f"{path_text}::{target}"
    return None, None

def ensure_display_name(df: pd.DataFrame) -> pd.DataFrame:
    if "display_name" not in df.columns:
        def _disp(r):
            for c in ["name", "reviews.title", "asins", "brand", "categories"]:
                if c in r and pd.notna(r[c]) and str(r[c]).strip():
                    return str(r[c]).strip()
            return "Unknown"
        df = df.copy()
        df["display_name"] = df.apply(_disp, axis=1)
    return df

CATEGORY_KEYWORDS = {
    "Tablets & E-Readers (General/Adult)": ["tablet", "kindle", "ereader", "e-reader", "fire", "hd", "display"],
    "Fire Tablets (Alexa / Special Offers)": ["fire", "tablet", "hd", "alexa", "special", "offers"],
    "Kids Tablets & Protective Cases": ["kids", "kid", "edition", "case", "proof"],
    "Smart Speakers & Portable Audio": ["echo", "speaker", "bluetooth", "smart", "show", "tap"],
    "Batteries & Household Power (AmazonBasics)": ["battery", "batteries", "aa", "aaa", "alkaline", "rechargeable", "nimh", "lithium"],
}

def plausible_for_category(name: str, category_label: str) -> bool:
    if not isinstance(name, str):
        return False
    keys = CATEGORY_KEYWORDS.get(category_label, [])
    if not keys:
        return True
    s = name.lower()
    return any(k in s for k in keys)

def detect_image_col(df):
    for c in [
        "imageURLs","imageURL","image_url","ImageURL","imUrl",
        "main_image_url","large_image","medium_image","small_image",
        "image","img","picture","picture_url"
    ]:
        if c in df.columns:
            return c
    return None

def detect_product_url_col(df):
    for c in ["product_url","detail_page_url","url","productLink","link","productURL","product_page_url"]:
        if c in df.columns:
            return c
    return None

URL_RE = re.compile(r"https?://[^\s,'\"\]]+")
def first_valid_url(x: str | None) -> str | None:
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    if s.startswith("//"):
        return "https:" + s
    if s.startswith(("http://","https://")):
        return s
    # Try JSON-like contents
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    if item.startswith("//"): return "https:" + item
                    if item.startswith(("http://","https://")): return item
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str):
                    if v.startswith("//"): return "https:" + v
                    if v.startswith(("http://","https://")): return v
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            if item.startswith("//"): return "https:" + item
                            if item.startswith(("http://","https://")): return item
    except Exception:
        pass
    m = URL_RE.search(s)
    return m.group(0) if m else None

# --- Adapter ZIP handling (search adapter_config.json anywhere) ---
def unzip_to_tmp(zip_path: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="adapter_")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmpdir)
    return tmpdir

def find_adapter_root(root_dir: str) -> str | None:
    for current_root, _, files in os.walk(root_dir):
        has_cfg = "adapter_config.json" in files
        has_model = any(
            f.startswith("adapter_model") and (f.endswith(".bin") or f.endswith(".safetensors"))
            for f in files
        )
        if has_cfg and has_model:
            return current_root
    return None

# ===== Facts block builder =====
def build_facts_block(df_cat: pd.DataFrame, category_label: str, top_k=3, worst_k=1, min_reviews=3):
    if "reviews.rating" not in df_cat.columns:
        return "No rating data.", [], None

    df_cat = df_cat.copy()
    df_cat["reviews.rating"] = pd.to_numeric(df_cat["reviews.rating"], errors="coerce")
    df_cat = df_cat[df_cat["display_name"].apply(lambda x: plausible_for_category(x, category_label))]
    if df_cat.empty:
        return "No plausible products for this category.", [], None

    image_col = detect_image_col(df_cat)
    agg_map = {
        "product_name": ("display_name", "first"),
        "n_reviews": ("reviews.rating", "count"),
        "avg_rating": ("reviews.rating", "mean"),
    }
    if image_col:
        agg_map["image_url"] = (image_col, "first")

    grp = (df_cat.groupby("asins", dropna=False).agg(**agg_map).reset_index())
    grp = grp[grp["n_reviews"] >= min_reviews].copy()
    if grp.empty:
        return "Not enough reviews to summarize.", [], None

    top = grp.sort_values(["avg_rating","n_reviews"], ascending=[False, False]).head(top_k)
    worst = grp.sort_values(["avg_rating","n_reviews"], ascending=[True, False]).head(worst_k)

    lines = ["Top products:"]
    for i, r in enumerate(top.itertuples(index=False), 1):
        lines.append(f"- {i}. {r.product_name} (ASIN {r.asins}): avg_rating={r.avg_rating:.2f}, n_reviews={int(r.n_reviews)}")
    if not worst.empty:
        w = worst.iloc[0]
        lines.append("Worst product:")
        lines.append(f"- {w.product_name} (ASIN {w.asins}): avg_rating={w.avg_rating:.2f}, n_reviews={int(w.n_reviews)}")

    return "\n".join(lines), top.to_dict("records"), (None if worst.empty else worst.to_dict("records")[0])

def choose_category_column(df):
    n_labels = df["meta_category_label"].nunique(dropna=True) if "meta_category_label" in df.columns else 0
    n_ids    = df["meta_category"].nunique(dropna=True) if "meta_category" in df.columns else 0
    if n_labels <= 1 and n_ids > 1 and "meta_category" in df.columns:
        mapping = {
            0: "Smart Speakers & Portable Audio",
            1: "Tablets & E-Readers (General/Adult)",
            2: "Fire Tablets (Alexa / Special Offers)",
            3: "Kids Tablets & Protective Cases",
            4: "Batteries & Household Power (AmazonBasics)",
        }
        df = df.copy()
        df["meta_category"] = pd.to_numeric(df["meta_category"], errors="coerce").astype("Int64")
        df["meta_category_label"] = df["meta_category"].map(mapping)
        return df, "meta_category_label"
    if n_labels > 1:
        return df, "meta_category_label"
    if n_ids > 1:
        return df, "meta_category"
    return df, None

# ===== Build a fluent seed (no instructions) =====
def _fmt_num(n):
    try:
        return f"{int(n):,}"
    except Exception:
        try:
            return f"{float(n):.2f}"
        except Exception:
            return str(n)

def build_article_seed(category_label: str, top_list: list[dict], worst_item: dict | None) -> str:
    """Create clean sentences from the top products + worst item; BART will polish."""
    lines = []
    lines.append(f"{category_label}: here is how the highest-rated options stack up based on customer reviews.")

    if top_list:
        top_sorted = sorted(
            top_list,
            key=lambda r: (float(r.get('avg_rating', 0)), int(r.get('n_reviews', 0))),
            reverse=True
        )
        names = []
        for r in top_sorted[:3]:
            name = str(r.get("product_name", "Unknown")).strip()
            asin = str(r.get("asins", "")).strip()
            rating = f"{float(r.get('avg_rating', 0)):.2f}"
            nrev = _fmt_num(r.get("n_reviews", 0))
            names.append(f"{name} (ASIN {asin}, {rating}★ over {nrev} reviews)" if asin else f"{name} ({rating}★ over {nrev} reviews)")

        if len(names) == 1:
            lines.append(f"The current front-runner is {names[0]}.")
        elif len(names) == 2:
            lines.append(f"Top picks are {names[0]} and {names[1]}.")
        else:
            lines.append(f"The top three are {names[0]}, {names[1]}, and {names[2]}.")

        lines.append("Rankings weigh both average rating and review volume to reduce noise from small samples.")

    if worst_item:
        w_name = str(worst_item.get("product_name", "the lowest-rated item")).strip()
        w_asin = str(worst_item.get("asins", "")).strip()
        w_rating = f"{float(worst_item.get('avg_rating', 0)):.2f}"
        w_nrev = _fmt_num(worst_item.get("n_reviews", 0))
        lines.append(
            f"Avoid {w_name} (ASIN {w_asin}); it trails the category with a {w_rating}★ average across {w_nrev} reviews."
            if w_asin else
            f"Avoid {w_name}; it trails the category with a {w_rating}★ average across {w_nrev} reviews."
        )

    lines.append("Overall, choose models that pair strong ratings with substantial review counts for dependable performance.")
    seed = " ".join(s.rstrip().rstrip(",") for s in lines if s and s.strip())
    return seed

# Anti-bleed guard: detect batteries language in non-battery categories
def suspicious_for_category(text: str, category_label: str) -> bool:
    t = (text or "").lower()
    if "batter" in category_label.lower():
        return False
    red_flags = ["battery", "batteries", "amazonbasics", " aa ", " aaa ", "alkaline", "nimh"]
    return any(flag in t for flag in red_flags)

# ===== Sidebar =====
st.sidebar.header("📥 Load Data & Model")
data_path_text = st.sidebar.text_input("Path to CSV/ZIP (or leave default)", value="/mnt/data/clustered_with_urls.csv")
data_upload    = st.sidebar.file_uploader("…or upload CSV/ZIP", type=["csv","zip"])
robot_img      = st.sidebar.file_uploader("Upload Robot Adviser Image (optional)", type=["png","jpg","jpeg"])

st.sidebar.divider()
st.sidebar.subheader("🧠 Summarizer")
use_model = st.sidebar.toggle("Enable summarizer", value=True)
st.sidebar.caption(f"Base model: `{BASE_MODEL_ID}`")

adapter_exists = Path(ADAPTER_ZIP).exists()
if adapter_exists:
    st.sidebar.success(f"Found {ADAPTER_ZIP} — will auto-load soft prompt.")
else:
    st.sidebar.warning("No softprompt_adapter.zip found. Using base model only.")

show_product_panel = st.sidebar.toggle("Show product panel", value=True)
show_diagnostics   = st.sidebar.toggle("Show diagnostics", value=False)

# ===== Load summarizer =====
summarizer = None
if use_model:
    try:
        _ = AutoConfig.from_pretrained(BASE_MODEL_ID, local_files_only=False)
        tok  = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)

        if adapter_exists and PEFT_AVAILABLE:
            extracted = unzip_to_tmp(ADAPTER_ZIP)
            adapter_dir = find_adapter_root(extracted)
            if not adapter_dir:
                files = []
                for r, _, fs in os.walk(extracted):
                    for f in fs:
                        files.append(os.path.relpath(os.path.join(r, f), extracted))
                raise RuntimeError(
                    "Could not locate a PEFT adapter inside the ZIP. "
                    "Expected adapter_config.json + adapter_model.*\n"
                    + "\n".join(files[:200])
                )
            model = PeftModel.from_pretrained(base, adapter_dir)
            st.sidebar.success("Soft prompt adapter loaded ✅")
        else:
            if adapter_exists and not PEFT_AVAILABLE:
                st.sidebar.error("peft not installed; using base model.")
            model = base

        summarizer = pipeline("summarization", model=model, tokenizer=tok, device_map=None)
        _ = summarizer("One-line test.", max_length=24, min_length=8, do_sample=False, num_beams=2)
        st.sidebar.success("Summarizer ready ✅")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load/generate: {e}")
        st.stop()
else:
    st.sidebar.info("Summarizer disabled.")

# ===== Header =====
header_left, header_right = st.columns([1, 6], vertical_alignment="center")
with header_left:
    if robot_img is not None:
        show_image_safe(robot_img, width=140)
    elif Path("agent.jpg").exists():
        show_image_safe("agent.jpg", width=140)
    else:
        show_image_safe("https://via.placeholder.com/140?text=🤖", width=140)

with header_right:
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; justify-content:center;">
          <h1 style="margin:0; font-size:46px;">Robot Adviser</h1>
          <div style="color:#6b7280; font-size:16px; margin-top:6px;">
            Pick a category and sentiment to see a representative product and a concise summary.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== Load data =====
df, source_name = load_df_from_path_or_upload(data_path_text, data_upload)
if df is None:
    st.info("Provide a valid path or upload a CSV/ZIP in the sidebar to begin.")
    st.stop()

df = ensure_display_name(df)
df, cat_col = choose_category_column(df)
if not cat_col:
    st.error("No usable category column found (meta_category_label/meta_category).")
    st.stop()
sent_col = "sentiment_class" if "sentiment_class" in df.columns else None

# ===== Controls =====
left, right = st.columns([2, 1], vertical_alignment="center")
with left:
    cats = df[cat_col].astype(str).str.strip()
    counts = cats.value_counts()
    options = [f"{k} ({counts[k]} items)" for k in counts.index]
    map_to_val = {f"{k} ({counts[k]} items)": k for k in counts.index}
    sel_cat_disp = st.selectbox("Category", options=options, index=0 if options else None)
    sel_cat = map_to_val[sel_cat_disp]
with right:
    sel_sent = st.selectbox("Sentiment", options=["All", "Negative", "Neutral", "Positive"], index=0)

df_view = df[df[cat_col].astype(str).str.strip() == str(sel_cat).strip()].copy()
if sel_sent != "All" and sent_col and sent_col in df_view.columns:
    df_view = df_view[df_view[sent_col] == sel_sent]

st.caption(f"Matches: **{len(df_view):,}** rows from **{source_name or 'uploaded file'}**")
if df_view.empty:
    st.warning("No items match the current selection.")
    st.stop()

# ===== Summary generation (seed → summarize; no instructions fed to BART) =====
def first_valid_product_url(series: pd.Series) -> str | None:
    s = series.dropna().astype(str).str.strip()
    s = s[s.str.startswith(("http://","https://"))]
    return s.iloc[0] if not s.empty else None

def generate_summary_bart(category_label, top_list, worst_item):
    """
    Build a fluent seed from structured facts, summarize with BART,
    ensure category prefix, and guard against cross-category bleed.
    """
    seed = build_article_seed(category_label, top_list, worst_item)

    # 1st pass
    try:
        out = summarizer(
            seed,
            max_length=MAX_TARGET,
            min_length=max(110, int(MAX_TARGET * 0.65)),
            do_sample=False,
            num_beams=8,
            no_repeat_ngram_size=5,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True,
        )[0]["summary_text"].strip()
    except Exception as e:
        st.warning(f"Summarization error: {e}")
        out = seed

    # Prefix & tidy
    if not out.lower().startswith(category_label.lower()):
        out = f"{category_label}: {out}"
    out = re.sub(r"\s+", " ", out).strip()
    if not out.endswith("."):
        out += "."

    # Guard against bleed; if bad, fall back to seed (already fluent)
    if suspicious_for_category(out, category_label):
        safe = seed
        if not safe.lower().startswith(category_label.lower()):
            safe = f"{category_label}: {safe}"
        return safe if safe.endswith(".") else safe + "."

    return out

facts_text, top_list, worst_item = build_facts_block(df_view, str(sel_cat), top_k=3, worst_k=1, min_reviews=3)
image_col       = detect_image_col(df_view)
product_url_col = detect_product_url_col(df_view)

rep_img_url, rep_name, rep_url = None, "Unknown product", None
top_asin = None
if "asins" in df_view.columns and len(df_view):
    tmp = df_view.copy()
    tmp["reviews.rating"] = pd.to_numeric(tmp["reviews.rating"], errors="coerce")
    grp = tmp.groupby("asins")["reviews.rating"].agg(["count","mean"]).sort_values(["mean","count"], ascending=[False, False])
    if len(grp):
        top_asin = grp.index[0]

if top_asin is not None and image_col:
    cand = df_view[df_view["asins"] == top_asin]
    if not cand.empty:
        val = first_valid_url(str(cand[image_col].dropna().astype(str).iloc[0]))
        if val:
            rep_img_url = val
            rep_name = cand.get("display_name", pd.Series(["Unknown"])).iloc[0]
            if product_url_col and product_url_col in cand.columns:
                rep_url = first_valid_product_url(cand[product_url_col])

if not rep_img_url and image_col:
    s = df_view[image_col].dropna().astype(str).map(first_valid_url).dropna()
    if not s.empty:
        rep_img_url = s.iloc[0]
        rep_name = df_view.loc[s.index[0], "display_name"] if "display_name" in df_view.columns else rep_name
        if product_url_col and product_url_col in df_view.columns:
            rep_url = first_valid_product_url(df_view[product_url_col])

# ===== Layout =====
col_summary, col_product = st.columns([2.2, 1], vertical_alignment="top") if show_product_panel else (st.container(), None)

with col_summary:
    st.subheader("Summary")
    if use_model and summarizer is not None:
        st.write(generate_summary_bart(str(sel_cat), top_list, worst_item))
    else:
        st.info("Summarizer disabled — showing the facts:")
        st.code(f"Category: {sel_cat}\n{facts_text}")

if show_product_panel and col_product is not None:
    with col_product:
        st.subheader("Product")
        if SHOW_PRODUCT_IMAGE:
            if rep_img_url:
                show_image_safe(rep_img_url, caption=rep_name, width=280)
            else:
                show_image_safe("https://via.placeholder.com/280?text=Product", caption=rep_name, width=280)
                st.caption(f"No valid image URL found. Checked column: `{image_col or '—'}`.")
        if product_url_col and product_url_col in df_view.columns:
            v = first_valid_product_url(df_view[product_url_col])
            if v:
                try:
                    st.link_button("Open product page ↗", v)
                except Exception:
                    st.markdown(f"[Open product page ↗]({v})")

# ===== Diagnostics =====
if show_diagnostics:
    with st.expander("Diagnostics", expanded=False):
        st.write("**Facts block used for summarization:**")
        st.code(f"Category: {sel_cat}\n{facts_text}", language="text")

        st.divider()
        st.write("**Sample rows (filtered view):**")
        cols = ["display_name","asins","reviews.rating","reviews.title","reviews.text"]
        puc = detect_product_url_col(df_view)
        if puc: cols.append(puc)
        imgc = detect_image_col(df_view)
        if imgc: cols.append(imgc)
        cols = [c for c in cols if c in df_view.columns]
        st.dataframe(df_view[cols].head(12), use_container_width=True)

        st.divider()
        st.write("**Category overview**")
        st.write("Detected image column:", f"`{imgc or '—'}`")
        if "meta_category" in df.columns:
            st.dataframe(
                df.groupby("meta_category").size().rename("count").reset_index().sort_values("meta_category"),
                use_container_width=True
            )
        if "meta_category_label" in df.columns:
            st.dataframe(
                df.groupby("meta_category_label").size().rename("count").reset_index().sort_values("count", ascending=False),
                use_container_width=True
            )

# ===== Run hint (for reference):
# pip install streamlit pandas numpy transformers peft accelerate sentencepiece torch
# streamlit run app.py
