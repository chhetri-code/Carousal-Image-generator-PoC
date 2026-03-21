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
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("✨ AdLume.ai")
st.write("Ready-to-post high-converting Instagram Ads in seconds!")

# ---------------- PROMPTS ----------------
enhance_template = PromptTemplate.from_template("""
You are a prompt engineer specialising in AI image generation for commercial Instagram ads.
 
Convert the brief below into ONE image generation prompt for FLUX.1.1-pro.
Output ONLY the prompt — no labels, no JSON, no explanation.
 
Structure your prompt in this exact order:
 
1. SHOT: Experiment with Camera angle and framing — e.g. "Eye-level close-up", "45-degree overhead flat-lay", "Wide lifestyle shot", etc. 
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

enhance_chain = enhance_template | llm
caption_chain = caption_template | llm


# ---------------- HELPERS ----------------
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
# Store generated results so downloads don't trigger re-generation
if "generated" not in st.session_state:
    st.session_state.generated = False
if "images_bytes" not in st.session_state:
    st.session_state.images_bytes = []
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "prompt_log" not in st.session_state:
    st.session_state.prompt_log = []  # list of (base, enhanced) tuples per slide

# ---------------- INPUT CARD ----------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    prompt = st.text_area(
        "💡Creative Idea",
        placeholder="e.g. We sell fresh cold-pressed juices. Launching a new mango blast flavour this weekend."
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

    business_name = st.text_input("🏢Business Name", placeholder="e.g. Juice Bar by Priya")
    website = st.text_input("💻Website", placeholder="e.g. www.juicebar.in")
    location = st.text_input("📌Location", placeholder="e.g. Koramangala, Bengaluru")

    # ---- Promo Details ----
    promo_details = st.text_input(
        "🏷️ Promo Details *(optional)*",
        placeholder="e.g. Flat 30% Off | Use code SAVE30 | Valid till Sunday"
    )

    # ---- Logo / Product Image Upload ----
    st.markdown("🪪Brand Logo / Product Image *(optional)*")
    uploaded_file = st.file_uploader(
        "Upload your logo or product photo to include in all slides",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )
    logo_img = None
    if uploaded_file is not None:
        logo_img = Image.open(uploaded_file)
        st.image(logo_img, caption="Uploaded image will appear on all slides", width=160)

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

        # Build logo hint for prompt if an image was uploaded
        logo_hint = ""
        if logo_img is not None:
            logo_hint = "\nBrand logo/product image provided: incorporate it prominently and naturally into the scene."

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
        """

        caption = generate_caption(global_context)
        st.session_state.caption = caption

        slide_structures = [
            "Hero promotion, bold headline",
            "Features and benefits",
            "Strong call to action"
        ]

        # -------- PROMPT CARD --------
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

                # Immediately serialise to bytes and store — avoids re-run loss
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
        # Render from stored bytes — never regenerated on re-run
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

