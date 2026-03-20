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
st.title("✨ AdLume.aii")
st.write("Ready-to-post high-converting Instagram Ads in seconds!")

# ---------------- LLM ----------------
llm = ChatGroq(
    api_key=st.secrets["GROK_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0.6
)

# ---------------- PROMPTS ----------------
enhance_template = PromptTemplate.from_template("""
You are a senior ad creative director. Create a visually stunning Instagram ad image.

Requirements:
- Aspect ratio: 4:5 (1080x1440)
- Style: Realistic commercial photography, premium and professional
- Composition: Clear, focused, visually rich, conversion-oriented
- Text overlay: Include exact ad text on the image (headline, subtext, CTA)
- Typography: Strong, modern, highly legible, only text that fits naturally in the image
- Avoid clutter: Only essential and readable text
- Mood/Lighting: Bright, high-quality lighting, attention-grabbing
- Visual focus: Product or service central, clear branding
- Keep writeup on the pictures at the minimal side
- Do not add any offers if users have not specified
- If promo = True, then add appropriate offers

Output format:
- Headline: [Insert exact short headline text]
- Subtext: [Insert concise secondary message]
- CTA: [Insert clear call-to-action text]
- Background & style: [Describe realistic background, props, context for the product/service]
- Text placement: [Specify where headline, subtext, CTA appear in the composition]

Example:
Headline: "Flat 30% Off All Juices!"
Subtext: "Freshly squeezed sugarcane & fruit juices daily."
CTA: "Order Now"
Background & style: "Bright, modern juice bar, close-up on glass with condensation, warm natural lighting."
Text placement: "Headline at top center, subtext below product, CTA at bottom in contrasting button-style typography."

Base Input:
{base_prompt}
""")

caption_template = PromptTemplate.from_template("""
Write ONE engaging Instagram caption for a carousel post.

- Hook in first line
- CTA in last line
- Include 3-5 relevant hashtags
- Keep it concise, human sounding and natural

Context:
{context}
""")

enhance_chain = enhance_template | llm
caption_chain = caption_template | llm

# ---------------- IMAGE CLIENT ----------------
client = Together(api_key=st.secrets["TOGETHER_API_KEY"])

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

def add_overlay(img, business_name, website, location):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 1200

    if business_name:
        draw.text((30, y), business_name, fill="black", font=font)
        y -= 30
    if website:
        draw.text((30, y), website, fill="black", font=font)
        y -= 30
    if location:
        draw.text((30, y), location, fill="black", font=font)

    return img

def create_zip(images):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i, img in enumerate(images, 1):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            zf.writestr(f"slide_{i}.png", img_bytes.getvalue())
    buffer.seek(0)
    return buffer

# ---------------- INPUT CARD ----------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    prompt = st.text_area("Ad details (include promos you are running)")

    business_type = st.selectbox(
        "Business Type",
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
        "Ad Style",
        [
            "🔥 High Converting",
            "💎 Luxury Brand",
            "😎 Gen-Z Viral",
            "📈 Direct Response",
            "🧠 Emotional Story"
        ]
    )

    theme = st.selectbox(
    "Visual Style",
    [
        "📸 Realistic Photography",
        "🎨 Minimal Clean",
        "🌈 Bright & Colorful",
        "🖤 Dark Premium",
        "🏝 Lifestyle Aesthetic",
        "🪔 Festive & Fun"
    ]
    )

    business_name = st.text_input("Business Name")
    website = st.text_input("Website")
    location = st.text_input("Location")
    promo = st.toggle("Add automatic offers")
    generate = st.button("🪄 Create Ads!", type = "primary", width = "stretch")

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
        images = []

        global_context = f"""
        Promotion: {prompt}
        Business Type: {business_type}
        Tone: {tone}
        Theme: {theme}
        Business Name: {business_name}
        Website: {website}
        Location: {location}
        Promo: {promo}
        """

        caption = generate_caption(global_context)

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

                with st.spinner(f"Generating Slide {i}..."):
                    img = generate_image(enhanced)
                    img = add_overlay(img, business_name, website, location)

                images.append(img)

            st.markdown('</div>', unsafe_allow_html=True)

        # -------- OUTPUT CARD --------
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.markdown("### 🎉 Your marketing Package is Ready!")

            st.markdown("#### 📢 Caption")
            st.info(caption)

            st.markdown('</div>', unsafe_allow_html=True)

            # -------- IMAGES WITH DOWNLOAD ICON --------
            st.markdown("#### 🖼️ Images")

            for i, img in enumerate(images, 1):
                buf = io.BytesIO()
                img.save(buf, format="PNG")

                # Show image
                st.image(img, caption=f"Slide {i}", width="stretch")

                
                # Small icon-style download button
                st.download_button(
                    label=f"💾 Save Slide {i}.png",
                    data=buf.getvalue(),
                    file_name=f"slide_{i}.png",
                    mime="image/png",
                    key=f"download_{i}",
                    help="Download image",
                    width = "stretch"
                )

            # -------- ZIP DOWNLOAD --------
            zip_file = create_zip(images)

            
            st.download_button(
            label="Download All Slides",
            data=zip_file,
            file_name="carousel.zip",
            mime="application/zip",
            type = "primary",
            width = "stretch"
            )

            st.markdown('</div>', unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("""
<div style='text-align:center; font-size:12px; color:gray; margin-top:30px;'>
Made in 🇮🇳 with ❤️ by Chhetri
</div>
""", unsafe_allow_html=True)
