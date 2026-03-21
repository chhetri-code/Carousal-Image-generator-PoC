import streamlit as st
import os
import io
import zipfile
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
import json

import warnings

warnings.filterwarnings('ignore')

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from together import Together
from groq import Groq





# ---------------- CONFIG ----------------
st.set_page_config(page_title="AdLume.ai", layout="centered")

# ---------------- CSS ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
}

.block-container {
    max-width: 520px;
    padding-top: 3rem;
}

/* Card container styling */
[data-testid="stVerticalBlock"] > div:has(.card) {
    background: #ffffff;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    margin-bottom: 16px;
}

/* Titles */
h1, h2, h3 {
    color: #15803d;
}

.section-title {
    font-weight: 600;
    margin-bottom: 10px;
    font-size:16px;
}

/* Reduce font size of code blocks */
pre code {
    font-size: 12px !important;
    line-height: 1.4 !important;
}

pre {
    padding: 10px !important;
    border-radius: 10px !important;
}

/* Primary button (Generate) */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ff7a18, #22c55e);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 12px;
    border: none;
}


/* Download buttons */
.stDownloadButton > button {
    background: #22c55e;
    color: white;
    border-radius: 10px;
}

/* Radio + Select highlight */
div[data-baseweb="radio"] label {
    background: #f0fdf4;
    padding: 6px 10px;
    border-radius: 8px;
}

/* Inputs */
input, textarea {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
}

/* Caption box */
.stAlert {
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    color: #6b7280;
    margin-top: 30px;
}

.stCardContainer {
    background: #ffffff;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    margin-bottom: 16px;
}
.card-container {
    background: #ffffff;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    margin-bottom: 16px;
}

/* ── File Uploader: drop zone ── */
[data-testid="stFileUploader"] {
    background: #f0fdf4;
    border: 2px dashed #86efac;
    border-radius: 14px;
    padding: 6px 12px;
    transition: border-color 0.2s ease, background 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #22c55e;
    background: #dcfce7;
}

/* Drop zone inner section */
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 12px 8px !important;
}

/* "Drag and drop" label */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    color: #15803d !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* "Limit 2MB" / file type hint */
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    color: #6b7280 !important;
    font-size: 12px !important;
}

/* Browse-files button inside uploader */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, #ff7a18, #22c55e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 6px 18px !important;
    transition: opacity 0.2s ease;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    opacity: 0.88 !important;
}

/* Uploaded file pill */
[data-testid="stFileUploaderFile"] {
    background: #ffffff !important;
    border: 1px solid #bbf7d0 !important;
    border-radius: 10px !important;
    padding: 6px 10px !important;
    margin-top: 6px !important;
}

/* File name text */
[data-testid="stFileUploaderFileName"] {
    color: #15803d !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

/* File size text */
[data-testid="stFileUploaderFileSize"] {
    color: #6b7280 !important;
    font-size: 12px !important;
}

/* Delete (x) button on uploaded file */
[data-testid="stFileUploaderDeleteBtn"] button {
    background: transparent !important;
    border: none !important;
    color: #ef4444 !important;
    font-size: 14px !important;
    padding: 0 4px !important;
}

[data-testid="stFileUploaderDeleteBtn"] button:hover {
    color: #b91c1c !important;
}

/* Upload progress bar */
[data-testid="stFileUploaderProgressBar"] > div {
    background: linear-gradient(90deg, #ff7a18, #22c55e) !important;
    border-radius: 99px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("✨ AdLume.ai")
st.write("Ready-to-post high-converting Instagram Ads in seconds!")

# ---------------- PROMPTS ----------------
enhance_template = PromptTemplate.from_template("""
You are a prompt engineer specialising in AI image generation for commercial Instagram ads.

Convert the brief below into ONE image generation prompt for FLUX.1.1-pro. Utilize colors and themes present in the brief.
Output ONLY the prompt — no labels, no JSON, no explanation.

Structure your prompt in this exact order:

1. SHOT: Experiment with Camera angle and framing like a ad director - Examples: "Wide lifestyle shot", etc. 
2. SUBJECT: One hero element only — the product, dish, person, or scene. Be specific: colour, texture, size.
3. LIGHTING: Source and quality — e.g. "soft diffused window light", "warm golden-hour rim light", "sharp studio softbox from left"
4. BACKGROUND: Simple and complementary. Shallow depth of field — background blurred, subject sharp.
5. MOOD & PALETTE: Match the visual style and tone from the brief exactly.
6. TEXT: Exactly ONE headline (max 5 words) and ONE CTA (max 3 words) rendered as large, bold, modern sans-serif typography placed in natural negative space. Spell every word correctly. No other text.
7. TECHNICAL: End with "Commercial advertising photography, 4:5 portrait, sharp focus, high resolution, no watermarks"

Constraints:
- Maximum 2 text elements total (headline + CTA). No subtext, no body copy, no decorative labels.
- No more than 3 visual elements in the frame — hero subject, background, and text only.
- If Promo Details are in the brief, include the exact promo wording as the headline text.
- Typography must look professionally designed — clean, high contrast, legible.
- Make design conversion oriented.
- Leave space for additional text that can be overlaid by user.

Base Input:
{base_prompt}
""")

caption_template = PromptTemplate.from_template("""
Write ONE engaging Instagram caption for a carousel post.

- Hook in first line
- CTA in last line
- Include 3 to 5 relevant hashtags
- Keep it concise, human sounding and natural

Context:
{context}
""")

# ---------------- IMAGE CLIENT ----------------
client = Together(api_key=st.secrets["TOGETHER_API_KEY"])

# ---------------- LLM ----------------
llm = ChatGroq(
    api_key=st.secrets["GROK_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0.6
)

# Vision-capable Groq client for product image analysis
groq_client = Groq(api_key=st.secrets["GROK_API_KEY"])

enhance_chain = enhance_template | llm
caption_chain = caption_template | llm


# ---------------- HELPERS ----------------

def pil_image_to_base64(pil_img: Image.Image, max_size: int = 1024) -> tuple[str, str]:
    """
    Resize PIL image to fit within max_size and return (base64_string, media_type).
    Uses JPEG for smaller payloads.
    """
    img = pil_img.copy()

    # Resize so longest side <= max_size
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Convert RGBA → RGB for JPEG compatibility
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"


def analyse_product_images(product_images: list[Image.Image]) -> str:
    """
    Send product images to the LLM vision model and return a detailed
    description that will enrich the prompt enhancement step.
    """
    if not product_images:
        return ""

    content = []

    # Instruction text first
    content.append({
        "type": "text",
        "text": (
            "You are a visual analyst for an ad-creative tool. "
            "Analyse the product image(s) below and return a concise description "
            "(3-5 sentences) covering:\n"
            "- Product type, colour, texture, shape, and size\n"
            "- Packaging style or material (if visible)\n"
            "- Dominant visual elements and mood\n"
            "- Any text, logos, or branding visible on the product\n\n"
            "Focus only on visual facts useful for an AI image-generation prompt. "
            "Do NOT include any preamble or labels — output the description only."
        )
    })

    # Attach each image (max 3 to stay within token limits)
    for pil_img in product_images[:3]:
        b64, media_type = pil_image_to_base64(pil_img)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{b64}"
            }
        })

    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # vision-capable model on Groq
            messages=[{"role": "user", "content": content}],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Graceful fallback: return a generic hint so generation still works
        return f"Product image provided. Use it as the hero subject. (Vision analysis unavailable: {e})"


def enhance_prompt(base_prompt):
    try:
        return enhance_chain.invoke({"base_prompt": base_prompt}).content
    except:
        return base_prompt


def generate_caption(context):
    try:
        return caption_chain.invoke({"context": context}).content
    except:
        return "🔥 Don't miss out. Visit us today!"


def generate_image(prompt):
    try:
        response = client.images.generate(
            prompt=prompt,
            model="black-forest-labs/FLUX.1.1-pro"
        )
        img_url = response.data[0].url
        return Image.open(io.BytesIO(requests.get(img_url).content))
    except:
        img = Image.new("RGB", (1080, 1350), color=(255, 230, 200))
        draw = ImageDraw.Draw(img)
        draw.text((50, 600), "Fallback Image", fill="black")
        return img


def add_overlay(img, business_name, website, location, logo_img=None):
    """Add text overlay and optionally paste a logo onto the image."""
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # --- Paste logo in bottom-right corner if provided ---
    if logo_img is not None:
        try:
            logo = logo_img.copy().convert("RGBA")

            # Resize logo: max 180px wide, maintain aspect ratio
            max_logo_w = 180
            logo_w, logo_h = logo.size
            scale = min(max_logo_w / logo_w, 120 / logo_h)
            new_w = int(logo_w * scale)
            new_h = int(logo_h * scale)
            logo = logo.resize((new_w, new_h), Image.LANCZOS)

            # Position: bottom-right with 20px padding
            img_w, img_h = img.size
            paste_x = img_w - new_w - 20
            paste_y = img_h - new_h - 20

            # Use alpha channel as mask if available
            if logo.mode == "RGBA":
                img.paste(logo, (paste_x, paste_y), logo)
            else:
                img.paste(logo, (paste_x, paste_y))
        except Exception as e:
            pass  # If logo pasting fails, continue without it

    # --- Text overlay ---
    y = img.size[1] - 30  # Start near bottom
    if business_name:
        draw.text((30, y), business_name, fill="black", font=font)
        y -= 30
    if website:
        draw.text((30, y), website, fill="black", font=font)
        y -= 30
    if location:
        draw.text((30, y), location, fill="black", font=font)

    return img


def img_to_bytes(img):
    """Convert PIL image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_zip(image_bytes_list):
    """Create a zip from a list of raw PNG byte strings."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i, img_bytes in enumerate(image_bytes_list, 1):
            zf.writestr(f"slide_{i}.png", img_bytes)
    buffer.seek(0)
    return buffer


# ---------------- SESSION STATE INIT ----------------
if "generated" not in st.session_state:
    st.session_state.generated = False
if "images_bytes" not in st.session_state:
    st.session_state.images_bytes = []
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "prompt_log" not in st.session_state:
    st.session_state.prompt_log = []
if "product_vision_desc" not in st.session_state:
    st.session_state.product_vision_desc = ""

# ---------------- INPUT CARD ----------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    prompt = st.text_area(
        "💡Creative Idea",
        placeholder="We sell beautiful flowers that never leave you! Cute & beautiful crotchet flowers shipped all over India. "
    )

    business_type = st.selectbox(
        "🛍️Business Type",
        [
            "🍔 Food & Dining",
            "🛍️ Retail & E-commerce",
            "💄 Beauty & Cosmetics",
            "🏋️ Fitness & Lifestyle",
            "🏥 Healthcare & Wellness",
            "🏨 Hospitality & Hotels",
            "🎓 Education & Coaching",
            "💼 Professional Services",
            "📱 Tech & Startups",
            "🎟️ Events & Entertainment"
        ]
    )

    tone = st.selectbox(
        "🎭Ad Style",
        [
            "🔥 High Converting",
            "💎 Luxury Brand",
            "😎 Gen-Z Viral",
            "📈 Direct Response",
            "🧠 Emotional Story"
        ]
    )

    theme = st.selectbox(
        "🎬Visual Style",
        [
            "📸 Realistic Photography",
            "🎨 Minimal Clean",
            "🌈 Bright & Colorful",
            "🖤 Dark Premium",
            "🏝 Lifestyle Aesthetic",
            "🪔 Festive & Fun"
        ]
    )

    business_name = st.text_input("🏢Business Name", placeholder="Daisy Dahlia")
    website = st.text_input("💻Website *(Optional)*", placeholder="www.crotchetflr.in")
    location = st.text_input("📌Location *(Optional)*", placeholder="Koramangala, Bengaluru")

    # ---- Promo Details ----
    promo_details = st.text_input(
        "🏷️ Promo Details *(optional)*",
        placeholder="e.g. Flat 30% Off | Use code SAVE30 | Valid till Sunday"
    )

    # ---- Logo Upload ----
    st.markdown("🪪Brand Logo *(optional)*")
    st.caption(f"Upload Logo")
    logo_file = st.file_uploader(
        "Upload your logo to appear on all slides",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="logo_uploader"
    )
    logo_img = None
    if logo_file is not None:
        logo_img = Image.open(logo_file)
        st.image(logo_img, caption="Appears on all slides", width=160)

    # ---- NEW: Product Images Upload ----
    MAX_PRODUCT_IMG_MB = 2
    MAX_PRODUCT_IMG_BYTES = MAX_PRODUCT_IMG_MB * 1024 * 1024

    st.markdown("📦 Product Images *(optional)*")
    st.caption(f"Upload 1–3 product photos (max {MAX_PRODUCT_IMG_MB} MB each). The AI will analyse them to generate more accurate, on-brand ads.")
    product_files = st.file_uploader(
        "Upload up to 3 product photos",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="product_uploader"
    )

    product_images: list[Image.Image] = []
    if product_files:
        # Cap at 3 images
        product_files = product_files[:3]
        oversized = [pf.name for pf in product_files if pf.size > MAX_PRODUCT_IMG_BYTES]
        if oversized:
            st.error(f"⚠️ These files exceed the {MAX_PRODUCT_IMG_MB} MB limit and were skipped: {', '.join(oversized)}")
        valid_files = [pf for pf in product_files if pf.size <= MAX_PRODUCT_IMG_BYTES]
        if valid_files:
            cols = st.columns(len(valid_files))
            for col, pf in zip(cols, valid_files):
                pil = Image.open(pf)
                product_images.append(pil)
                col.image(pil, use_container_width=True)
            st.success(f"✅ {len(product_images)} product image(s) loaded: AI will analyse before generating ads.")

    promo = st.toggle("Add automatic offers")
    generate = st.button("🪄 Create Ads!", type="primary", width="stretch")

    st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("✨ AdLume.ai")

    st.markdown("Ready-to-post high-converting Instagram Ads in seconds!")

    with st.expander("🔐 Login/Register"):
        st.write("Save your ads, access history & unlock Pro features.")
        st.info("Coming Soon 🚧")

    # ABOUT APP
    with st.expander("ℹ️ AdLume.ai"):
        st.write("""
        AdLume.ai helps you turn simple ideas into ready to post **scroll-stopping Instagram ads**.

        🎯 **Built for:**
        - Founders & startups  
        - Small businesses  
        - Marketers & agencies  
        - Creators   
        """)
        st.info("""
        ⚡ **What you get:**
        - Idea to ad in seconds  
        - Smart prompt enhancement (no effort needed)  
        - High-quality, realistic ad creatives  
        - Faster content, better conversions """)

    # PRICING
    with st.expander("💰 Pricing"):
        st.write("""
        Built for creators who want to win with **SPEED + RELIABILITY**.

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
        st.info("""Coming Soon 🚧
        """)

        st.button("🚀 Unlock Pro", use_container_width=True)

    # ABOUT CHHETRI
    with st.expander("👤 About Me"):
        st.write("""
        Chhetri builds simple yet effective AI tools for **productivity, creativity, and data-driven decision making**.

        **📌Focus:**
        - AI × creativity  
        - Data science & analytics  
        - Digital products  
        - Real-world utility 
        """)
        st.info(
            """
            📫 **Contact**

            📧 [Email](mailto:chhetri.code@gmail.com)  
            💼 [LinkedIn](https://www.linkedin.com/in/prashant-kumar-chhetri)
            """)
    st.markdown("---")
    st.caption("Made in 🇮🇳 with ❤️ by Chhetri")

# ---------------- GENERATION ----------------
if generate:
    if not prompt:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.warning("⚠️ Describe your Idea!")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Reset session state for a fresh generation
        st.session_state.generated = False
        st.session_state.images_bytes = []
        st.session_state.caption = ""
        st.session_state.prompt_log = []
        st.session_state.product_vision_desc = ""

        # ── Step 1: Analyse product images with vision LLM ──────────────────
        product_vision_desc = ""
        if product_images:
            with st.spinner(f"🔍 Analysing {len(product_images)} product image(s) with AI vision..."):
                product_vision_desc = analyse_product_images(product_images)
                st.session_state.product_vision_desc = product_vision_desc

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🔍 Product Vision Analysis")
                st.info(product_vision_desc)
                st.markdown('</div>', unsafe_allow_html=True)

        # Build logo hint for prompt if a logo was uploaded
        logo_hint = ""
        if logo_img is not None:
            logo_hint = "\nBrand logo provided: incorporate it subtly in the scene."

        # Build product image context block for the prompt
        product_image_context = ""
        if product_vision_desc:
            product_image_context = f"\nProduct Visual Description (from uploaded images):\n{product_vision_desc}"

        global_context = f"""
        Promotion: {prompt}
        Business Type: {business_type}
        Tone: {tone}
        Theme: {theme}
        Business Name: {business_name}
        Website: {website}
        Location: {location}
        Promo: {promo}
        Promo Details: {promo_details if promo_details else "None"}
        {logo_hint}
        {product_image_context}
        """

        caption = generate_caption(global_context)
        st.session_state.caption = caption

        slide_structures = [
            "Hero promotion, bold headline",
            "Features and benefits",
            "Strong call to action"
        ]

        # ── Step 2: Prompt enhancement + image generation ───────────────────
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Prompt Enhancement Pipeline")

            for i, structure in enumerate(slide_structures, 1):
                base_prompt = f"{global_context}\nObjective: {structure}"
                st.markdown(f"#### Slide {i}")
                st.write("**Base Prompt**")
                st.code(base_prompt)

                enhanced = enhance_prompt(base_prompt)
                st.write("**Enhanced Prompt**")
                st.code(enhanced)

                # Store prompt log
                st.session_state.prompt_log.append((base_prompt, enhanced))

                with st.spinner(f"Generating Slide {i}..."):
                    img = generate_image(enhanced)
                    img = add_overlay(img, business_name, website, location, logo_img=logo_img)

                st.session_state.images_bytes.append(img_to_bytes(img))

            st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.generated = True

# ---------------- OUTPUT (always shown when generated) ----------------
if st.session_state.generated and st.session_state.images_bytes:

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎉 Your marketing Package is Ready!")

        st.markdown("#### 📢 Caption")
        st.info(st.session_state.caption)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- IMAGES WITH DOWNLOAD BUTTONS --------
    st.markdown("#### 🖼️ Images")

    for i, img_bytes in enumerate(st.session_state.images_bytes, 1):
        st.image(img_bytes, caption=f"Slide {i}", width="stretch")

        st.download_button(
            label=f"💾 Save Slide {i}.png",
            data=img_bytes,
            file_name=f"slide_{i}.png",
            mime="image/png",
            key=f"download_{i}",
            help="Download image",
            width="stretch"
        )

    # -------- ZIP DOWNLOAD --------
    zip_file = create_zip(st.session_state.images_bytes)

    st.download_button(
        label="⬇️ Download All Slides",
        data=zip_file,
        file_name="carousel.zip",
        mime="application/zip",
        type="primary",
        width="stretch"
    )

# -------- FOOTER --------
st.markdown("""
<div style='text-align:center; font-size:12px; color:gray; margin-top:30px;'>
Made in 🇮🇳 with ❤️ by Chhetri
</div>
""", unsafe_allow_html=True)
