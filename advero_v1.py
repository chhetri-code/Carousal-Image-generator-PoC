"""
AdLume.ai — Optimized v3
========================
Optimizations in this version:
  • Removed unused `os` import
  • CSS: --grad variable centralises gradient; duplicate button rule collapsed
  • SESSION_DEFAULTS is single source of truth for both init and reset
  • reset_session_state uses copy.copy - no shared mutable default references
  • pil_to_base64: removed unnecessary .copy() - works on local ref, avoids alloc
  • add_overlay: early-return guard when nothing to draw
  • Vision cache promoted to module-level dict - faster, no session serialisation
  • render_copy_caption_button: json.dumps for safe JS escaping (all Unicode safe)
  • _load_product_images helper extracted from render_input_form
  • ZIP built once and stored in session_state.zip_cache — no redundant re-creation
  • All magic strings / model names moved to named constants
"""

from __future__ import annotations

import base64
import copy
import io
import json
import warnings
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from together import Together

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_MODEL      = "black-forest-labs/FLUX.1.1-pro"
VISION_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MODEL        = "llama-3.3-70b-versatile"
CAPTION_FALLBACK = "🔥 Don't miss out. Visit us today!"

MAX_PRODUCT_IMAGES    = 3
MAX_PRODUCT_IMG_MB    = 2
MAX_PRODUCT_IMG_BYTES = MAX_PRODUCT_IMG_MB * 1024 * 1024
MAX_B64_SIZE          = 1024  # px - longest edge when encoding for vision

SLIDE_STRUCTURES = [
    "Hero promotion, bold headline",
    "Features and benefits",
    "Strong call to action",
]

BUSINESS_TYPES = [
    "🍔 Food & Dining", "🛍️ Retail & E-commerce", "💄 Beauty & Cosmetics",
    "🏋️ Fitness & Lifestyle", "🏥 Healthcare & Wellness", "🏨 Hospitality & Hotels",
    "🎓 Education & Coaching", "💼 Professional Services", "📱 Tech & Startups",
    "🎟️ Events & Entertainment",
]
AD_STYLES = [
    "🔥 High Converting", "💎 Luxury Brand", "😎 Gen-Z Viral",
    "📈 Direct Response", "🧠 Emotional Story",
]
VISUAL_STYLES = [
    "📸 Realistic Photography", "🎨 Minimal Clean", "🌈 Bright & Colorful",
    "🖤 Dark Premium", "🏝 Lifestyle Aesthetic", "🪔 Festive & Fun",
]

# Single source of truth for session state keys → defaults
SESSION_DEFAULTS: dict = {
    "generated":           False,
    "images_bytes":        [],
    "caption":             "",
    "prompt_log":          [],
    "product_vision_desc": "",
    "zip_cache":           None,
}

# Module-level vision cache - avoids per-run session-state serialisation overhead
_vision_cache: dict[tuple, str] = {}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
ENHANCE_TEMPLATE = PromptTemplate.from_template("""
You are a prompt engineer specialising in AI image generation for commercial Instagram ads.

Convert the brief below into ONE image generation prompt for FLUX.1.1-pro.
Output ONLY the prompt - no labels, no JSON, no explanation.

Structure your prompt in this exact order:
1. SHOT: Camera angle and framing - e.g. "Wide lifestyle shot"
2. SUBJECT: One hero element only - product, dish, person, or scene. Be specific.
3. LIGHTING: Source and quality - e.g. "soft diffused window light"
4. BACKGROUND: Simple and complementary. Shallow depth of field - background blurred.
5. MOOD & PALETTE: Match the visual style and tone from the brief exactly.
6. TEXT: Exactly ONE headline (max 5 words) and ONE CTA (max 3 words) as large, bold,
   modern sans-serif in natural negative space. Spell every word correctly.
7. TECHNICAL: End with "Commercial advertising photography, 4:5 portrait, sharp focus,
   high resolution, no watermarks"

Constraints:
- Maximum 2 text elements (headline + CTA). No subtext, no body copy.
- No more than 3 visual elements - hero subject, background, and text only.
- If Promo Details are present, use the exact promo wording as the headline.
- Typography must be clean, high contrast, legible.
- Leave space for additional text overlay.

Brief:
{base_prompt}
""")

CAPTION_TEMPLATE = PromptTemplate.from_template("""
Write ONE engaging Instagram caption for a carousel post.
- Hook in first line
- CTA in last line
- 3–5 relevant hashtags
- Concise, human, natural

Context:
{context}
""")

VISION_PROMPT = (
    "You are a visual analyst for an ad-creative tool. "
    "Analyse the product image(s) and return a concise description (3-5 sentences) covering:\n"
    "- Product type, colour, texture, shape, and size\n"
    "- Packaging style or material (if visible)\n"
    "- Dominant visual elements and mood\n"
    "- Any text, logos, or branding visible\n\n"
    "Focus only on visual facts useful for an AI image-generation prompt. "
    "Output the description only — no preamble, no labels."
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AdInputs:
    prompt:         str
    business_type:  str
    tone:           str
    theme:          str
    business_name:  str
    website:        str
    location:       str
    promo_details:  str
    auto_promo:     bool
    logo_img:       Optional[Image.Image]  = None
    product_images: list[Image.Image]      = field(default_factory=list)

    def build_context(self, product_vision_desc: str = "") -> str:
        lines = [
            f"Promotion: {self.prompt}",
            f"Business Type: {self.business_type}",
            f"Tone: {self.tone}",
            f"Theme: {self.theme}",
            f"Business Name: {self.business_name}",
            f"Website: {self.website}",
            f"Location: {self.location}",
            f"Auto Promo: {self.auto_promo}",
            f"Promo Details: {self.promo_details or 'None'}",
        ]
        if self.logo_img is not None:
            lines.append("Brand logo provided: incorporate it subtly in the scene.")
        if product_vision_desc:
            lines.append(f"\nProduct Visual Description (from uploaded images):\n{product_vision_desc}")
        return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED CLIENTS - created once per process
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_together_client() -> Together:
    return Together(api_key=st.secrets["TOGETHER_API_KEY"])

@st.cache_resource
def get_llm() -> ChatGroq:
    return ChatGroq(api_key=st.secrets["GROK_API_KEY"], model=LLM_MODEL, temperature=0.6)

@st.cache_resource
def get_groq_client() -> Groq:
    return Groq(api_key=st.secrets["GROK_API_KEY"])

@st.cache_resource
def get_chains() -> tuple:
    """Returns (enhance_chain, caption_chain)."""
    llm = get_llm()
    return ENHANCE_TEMPLATE | llm, CAPTION_TEMPLATE | llm

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pil_to_base64(pil_img: Image.Image, max_size: int = MAX_B64_SIZE) -> tuple[str, str]:
    """Resize & encode PIL image as base64 JPEG. Does not mutate the original."""
    img = pil_img
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def create_zip(image_bytes_list: list[bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, data in enumerate(image_bytes_list, 1):
            zf.writestr(f"slide_{i}.png", data)
    buf.seek(0)
    return buf

def add_overlay(
    img: Image.Image,
    business_name: str,
    website: str,
    location: str,
    logo_img: Optional[Image.Image] = None,
) -> Image.Image:
    """Composite logo + text footer onto img in-place. Returns img."""
    texts    = list(filter(None, [business_name, website, location]))
    has_logo = logo_img is not None

    if not texts and not has_logo:
        return img  # nothing to draw — skip entirely

    if has_logo:
        try:
            logo = logo_img.convert("RGBA")
            lw, lh = logo.size
            scale  = min(180 / lw, 120 / lh)
            logo   = logo.resize((int(lw * scale), int(lh * scale)), Image.LANCZOS)
            iw, ih = img.size
            img.paste(logo, (iw - logo.width - 20, ih - logo.height - 20),
                      logo if logo.mode == "RGBA" else None)
        except Exception:
            pass  # non-fatal

    if texts:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        y    = img.size[1] - 30
        for text in texts:
            draw.text((30, y), text, fill="black", font=font)
            y -= 30

    return img

# ─────────────────────────────────────────────────────────────────────────────
# AI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def analyse_product_images(product_images: list[Image.Image]) -> str:
    """Vision LLM — cached at module level by image fingerprint."""
    if not product_images:
        return ""

    cache_key = tuple(img_to_bytes(p)[:512] for p in product_images)
    if cache_key in _vision_cache:
        return _vision_cache[cache_key]

    content: list[dict] = [{"type": "text", "text": VISION_PROMPT}]
    for pil_img in product_images[:MAX_PRODUCT_IMAGES]:
        b64, media_type = pil_to_base64(pil_img)
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:{media_type};base64,{b64}"}})
    try:
        resp   = get_groq_client().chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=300,
            temperature=0.3,
        )
        result = resp.choices[0].message.content.strip()
    except Exception as exc:
        result = f"Product image provided. Use it as the hero subject. (Vision unavailable: {exc})"

    _vision_cache[cache_key] = result
    return result

def enhance_prompt(base_prompt: str) -> str:
    enhance_chain, _ = get_chains()
    try:
        return enhance_chain.invoke({"base_prompt": base_prompt}).content
    except Exception as exc:
        st.warning(f"Prompt enhancement failed ({exc}); using base prompt.")
        return base_prompt

def generate_caption(context: str) -> str:
    _, caption_chain = get_chains()
    try:
        return caption_chain.invoke({"context": context}).content
    except Exception as exc:
        st.warning(f"Caption generation failed ({exc}).")
        return CAPTION_FALLBACK

def generate_image(prompt: str) -> Image.Image:
    try:
        resp = get_together_client().images.generate(prompt=prompt, model=IMAGE_MODEL)
        raw  = requests.get(resp.data[0].url, timeout=30).content
        return Image.open(io.BytesIO(raw))
    except Exception as exc:
        st.warning(f"Image generation failed ({exc}); using placeholder.")
        img = Image.new("RGB", (1080, 1350), color=(255, 230, 200))
        ImageDraw.Draw(img).text((50, 600), "Fallback Image", fill="black")
        return img

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    for k, v in SESSION_DEFAULTS.items():
        st.session_state.setdefault(k, v)

def reset_session_state() -> None:
    """Reset all generation outputs - uses copy.copy to avoid shared mutable refs."""
    for k, v in SESSION_DEFAULTS.items():
        st.session_state[k] = copy.copy(v)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
:root {
    --green-dark:  #15803d;
    --green-mid:   #22c55e;
    --green-light: #f0fdf4;
    --orange:      #ff7a18;
    --grad:        linear-gradient(90deg, var(--orange), var(--green-mid));
    --border:      #e5e7eb;
    --radius:      14px;
}

.stApp { background: linear-gradient(180deg, var(--green-light) 0%, #fff 100%); }
.block-container { max-width: 540px; padding-top: 3rem; }
h1, h2, h3 { color: var(--green-dark); }

/* All action buttons — single unified rule */
.stButton > button,
.stDownloadButton > button {
    width: 100%;
    background: var(--grad);
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: 600;
    color: #fff;
}

/* Inputs */
input, textarea { border-radius: 10px !important; border: 1px solid var(--border) !important; }

/* Alerts & code */
.stAlert { border-radius: 10px; }
pre { padding: 10px !important; border-radius: 10px !important; }
pre code { font-size: 12px !important; line-height: 1.4 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--green-light);
    border: 2px dashed #86efac;
    border-radius: var(--radius);
    padding: 6px 12px;
    transition: border-color .2s, background .2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--green-mid); background: #dcfce7; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    color: var(--green-dark) !important; font-weight: 600 !important; font-size: 14px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    color: #6b7280 !important; font-size: 12px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: var(--grad) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 13px !important; padding: 6px 18px !important;
}
[data-testid="stFileUploaderFile"] {
    background: #fff !important; border: 1px solid #bbf7d0 !important;
    border-radius: 10px !important; padding: 6px 10px !important; margin-top: 6px !important;
}
[data-testid="stFileUploaderFileName"] { color: var(--green-dark) !important; font-weight: 500 !important; }
[data-testid="stFileUploaderDeleteBtn"] button { background: transparent !important; color: #ef4444 !important; }

/* Card containers */
[data-testid="stVerticalBlock"] > div:has(.card) {
    background: #ffffff;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    margin-bottom: 16px;
}

/* Radio options */
div[data-baseweb="radio"] label { background: var(--green-light); padding: 6px 10px; border-radius: 8px; }

/* Footer */
.footer { text-align: center; font-size: 12px; color: #6b7280; margin-top: 30px; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _load_product_images(product_files) -> list[Image.Image]:
    """Validate, preview, and return PIL images from uploaded files."""
    if not product_files:
        return []

    product_files = product_files[:MAX_PRODUCT_IMAGES]
    oversized     = [f.name for f in product_files if f.size > MAX_PRODUCT_IMG_BYTES]
    if oversized:
        st.error(f"⚠️ Files exceed {MAX_PRODUCT_IMG_MB} MB limit (skipped): {', '.join(oversized)}")

    valid = [f for f in product_files if f.size <= MAX_PRODUCT_IMG_BYTES]
    if not valid:
        return []

    images = []
    cols   = st.columns(len(valid))
    for col, pf in zip(cols, valid):
        pil = Image.open(pf)
        images.append(pil)
        col.image(pil, use_container_width=True)
    st.success(f"✅ {len(images)} product image(s) loaded.")
    return images

def render_copy_caption_button(caption: str) -> None:
    """Clipboard button - json.dumps handles all Unicode / special chars safely."""
    js_string = json.dumps(caption)
    st.components.v1.html(
        f"""
        <div style="margin: 4px 0;">
            <button id="copyBtn" onclick="copyCaption()" style="
                background: linear-gradient(90deg, #ff7a18, #22c55e);
                color: #fff; border: none; border-radius: 10px;
                padding: 10px 20px; font-size: 14px; font-weight: 600;
                cursor: pointer; width: 100%; font-family: sans-serif;
                transition: background .2s;
            ">📋 Copy Caption</button>
        </div>
        <script>
            function copyCaption() {{
                navigator.clipboard.writeText({js_string}).then(() => {{
                    const btn = document.getElementById('copyBtn');
                    btn.textContent = '✅ Copied to clipboard!';
                    btn.style.background = '#15803d';
                    setTimeout(() => {{
                        btn.textContent = '📋 Copy Caption';
                        btn.style.background = 'linear-gradient(90deg, #ff7a18, #22c55e)';
                    }}, 2000);
                }}).catch(() => alert('Copy failed — please copy manually.'));
            }}
        </script>
        """,
        height=52,
    )

# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.title("✨ AdLume.ai")
        st.markdown("Ready-to-post high-converting Instagram Ads in seconds!")

        with st.expander("🔐 Login / Register"):
            st.write("Save your ads, access history & unlock Pro features.")
            st.info("Coming Soon 🚧")

        with st.expander("ℹ️ About AdLume.ai"):
            st.write("""
AdLume.ai turns simple ideas into **scroll-stopping Instagram ads**.

🎯 **Built for all:** Founders, small businesses, marketers & creators.

⚡ **What you get:**
- Idea to ad in seconds
- Smart prompt enhancement
- High-quality, realistic creatives
- Faster content, better conversions
""")

        with st.expander("💰 Pricing"):
            st.write("""
        🔰 **Free**
        - Limited daily generations  
        - Standard quality outputs  
        - Core features  

        👑 **Go Pro** 
        - 30+ generations (scalable)  
        - Advanced prompt optimization  
        - Premium ad styles & templates  
        - All free features 
        - Priority support  
""")
            st.info("Coming Soon 🚧")
            st.button("🚀 Unlock Pro", use_container_width=True)

        with st.expander("👤 Meet the Creator"):
            st.write("""
Chhetri builds AI tools for productivity, creativity, and data-driven decisions.

**AI × Creativity · Digital Products · Data Science**
""")
            st.info("📧 [Email](mailto:chhetri.code@gmail.com)  💼 [LinkedIn](https://www.linkedin.com/in/prashant-kumar-chhetri)")

        st.markdown("---")
        st.caption("Made in 🇮🇳 with ❤️ by :rainbow[CHHETRI]")

def render_input_form() -> Optional[AdInputs]:
    """Renders the input card and returns AdInputs on submit, else None."""
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        prompt        = st.text_area("💡 Creative Idea",
                                     placeholder="We sell beautiful crochet flowers shipped all over India…")
        business_type = st.selectbox("🛍️ Business Type", BUSINESS_TYPES)
        tone          = st.selectbox("🎭 Ad Style",       AD_STYLES)
        theme         = st.selectbox("🎬 Visual Style",   VISUAL_STYLES)
        business_name = st.text_input("🏢 Business Name", placeholder="Daisy Dahlia")
        website       = st.text_input("💻 Website *(optional)*", placeholder="www.crotchetflr.in")
        location      = st.text_input("📌 Location *(optional)*", placeholder="Koramangala, Bengaluru")
        promo_details = st.text_input("🏷️ Promo Details *(optional)*",
                                      placeholder="Flat 30% Off | Use code SAVE30 | Valid till Sunday")

        st.markdown("🪪 Brand Logo *(optional)*")
        st.caption("Upload your logo to appear on all slides.")
        logo_file = st.file_uploader("Upload logo", type=["png", "jpg", "jpeg", "webp"],
                                     label_visibility="collapsed", key="logo_uploader")
        logo_img = None
        if logo_file:
            logo_img = Image.open(logo_file)
            st.image(logo_img, caption="Appears on all slides", width=160)

        st.markdown(f"📦 Product Images *(optional - max {MAX_PRODUCT_IMAGES} images, {MAX_PRODUCT_IMG_MB} MB each)*")
        product_files  = st.file_uploader("Upload product photos", type=["png", "jpg", "jpeg", "webp"],
                                          accept_multiple_files=True, label_visibility="collapsed",
                                          key="product_uploader")
        product_images = _load_product_images(product_files)

        auto_promo = st.toggle("Add automatic offers")
        submitted  = st.button("🪄 Create Ads!", type="primary", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    if not submitted:
        return None
    if not prompt:
        st.warning("⚠️ Please describe your idea first!")
        return None

    return AdInputs(
        prompt=prompt, business_type=business_type, tone=tone, theme=theme,
        business_name=business_name, website=website, location=location,
        promo_details=promo_details, auto_promo=auto_promo,
        logo_img=logo_img, product_images=product_images,
    )

def render_output() -> None:
    if not (st.session_state.generated and st.session_state.images_bytes):
        return

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎉 Your Ad Package is Ready!")
        st.markdown("#### 📢 Caption")
        st.info(st.session_state.caption)
        render_copy_caption_button(st.session_state.caption)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Slides")
        for i, img_bytes in enumerate(st.session_state.images_bytes, 1):
            st.image(img_bytes, caption=f"Slide {i}", use_container_width=True)
            st.download_button(
                label=f"💾 Save Slide {i}.png",
                data=img_bytes,
                file_name=f"slide_{i}.png",
                mime="image/png",
                key=f"dl_{i}",
                width="stretch",
            )

        # Build ZIP once and cache - avoids recreating on every Streamlit re-render
        if st.session_state.zip_cache is None:
            st.session_state.zip_cache = create_zip(st.session_state.images_bytes)

        st.download_button(
            label="⬇️ Download All Slides (.zip)",
            data=st.session_state.zip_cache,
            file_name="carousel.zip",
            mime="application/zip",
            type="primary",
            width="stretch",
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_generation(inputs: AdInputs) -> None:
    reset_session_state()

    # Step 1 - Vision analysis (skipped when no product images)
    product_vision_desc = ""
    if inputs.product_images:
        with st.spinner(f"🔍 Analysing {len(inputs.product_images)} product image(s)…"):
            product_vision_desc = analyse_product_images(inputs.product_images)
        st.session_state.product_vision_desc = product_vision_desc
        st.markdown("### 🔍 Product Vision Analysis")
        st.info(product_vision_desc)

    # Step 2 - Caption
    context = inputs.build_context(product_vision_desc)
    st.session_state.caption = generate_caption(context)

    # Step 3 - Per-slide: enhance → generate → overlay
    st.markdown("### ⚙️ Prompt Enhancement Pipeline")
    for i, structure in enumerate(SLIDE_STRUCTURES, 1):
        base_prompt = f"{context}\nObjective: {structure}"
        st.markdown(f"#### Slide {i}")
        st.write("**Base Prompt**")
        st.text_area("base_prompt", base_prompt, height=200, disabled=True,
                     label_visibility="collapsed", key=f"base_{i}")

        enhanced = enhance_prompt(base_prompt)
        st.write("**Enhanced Prompt**")
        st.text_area("enhanced_prompt", enhanced, height=200, disabled=True,
                     label_visibility="collapsed", key=f"enhanced_{i}")
        st.session_state.prompt_log.append((base_prompt, enhanced))

        with st.spinner(f"Generating Slide {i}…"):
            img = generate_image(enhanced)
            img = add_overlay(img, inputs.business_name, inputs.website,
                              inputs.location, inputs.logo_img)

        st.session_state.images_bytes.append(img_to_bytes(img))

    st.session_state.generated = True

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="AdLume.ai", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    init_session_state()

    st.title("✨ AdLume.ai")
    st.write("Ready-to-post high-converting Instagram Ads in seconds!")

    render_sidebar()

    inputs = render_input_form()
    if inputs:
        run_generation(inputs)

    render_output()
    st.markdown('<div class="footer">Made in 🇮🇳 with ❤️ by :rainbow[CHHETRI]</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
