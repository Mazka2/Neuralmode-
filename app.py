import streamlit as st
import base64
import json
import os
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ====================== CONFIG ======================
st.set_page_config(
    page_title="AI Chart Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== PROMPT ======================
SYSTEM_PROMPT = """
You are an elite technical analyst specializing in pure price action, market structure, Smart Money Concepts (SMC), and high-probability trading setups.

Analyze ONLY what is clearly visible in the provided chart image. Never invent price levels, patterns, or indicators that are not readable.

Follow this strict Chain-of-Thought process:

1. Image Quality Check
   - Is the price axis legible?
   - Can you clearly see the candles and timeframe?
   - Flag any uncertainty.

2. Basic Info
   - Symbol (if visible)
   - Timeframe
   - Chart type (candlestick, etc.)

3. Market Structure
   - Overall trend: Bullish / Bearish / Ranging
   - Structure: Higher Highs & Higher Lows or Lower Highs & Lower Lows
   - Current phase (accumulation, expansion, distribution, etc.)

4. Key Levels
   - Strongest visible Support zones (approximate prices)
   - Strongest visible Resistance zones (approximate prices)
   - Any Order Blocks, Fair Value Gaps, or liquidity zones if clearly present

5. Patterns
   - Candlestick patterns (engulfing, pin bar, doji, etc.)
   - Chart patterns (flags, triangles, head & shoulders, double top/bottom, etc.)

6. Indicators (only if clearly visible)
   - Moving averages, RSI, MACD, Volume, Bollinger Bands, etc.
   - Interpret them in context of the price action

7. Trading Signal
   - BUY / SELL / WAIT / NO TRADE

8. Trade Plan (only if signal is BUY or SELL)
   - Precise Entry Zone
   - Stop Loss (with clear reason)
   - Target 1 and Target 2
   - Estimated Risk:Reward ratio

9. A+ Setup Grade
   Score the setup strictly against these criteria:
   - Clear higher-timeframe bias alignment
   - Price is at a high-quality key level / zone
   - Strong confirmation trigger present
   - Clean invalidation level + favorable Risk:Reward (≥ 1:2 preferred)
   - Overall confluence of factors

   Grade scale:
   - A+ → All criteria strongly met
   - A  → Almost all criteria met
   - B  → Decent setup but missing 1 important element
   - C  → Marginal / low probability
   - F  → Avoid

Output ONLY valid JSON in this exact structure (no markdown, no extra text):

{
  "symbol": "string or null",
  "timeframe": "string or null",
  "trend": "Bullish / Bearish / Ranging",
  "structure": "detailed structure description",
  "key_levels": {
    "supports": ["price1", "price2"],
    "resistances": ["price1", "price2"]
  },
  "patterns": ["pattern1", "pattern2"],
  "indicators_summary": "string",
  "signal": "BUY / SELL / WAIT / NO TRADE",
  "entry": "string or null",
  "stop_loss": "string or null",
  "targets": ["target1", "target2"],
  "risk_reward": "string or null",
  "a_plus_grade": "A+ / A / B / C / F",
  "grade_reasoning": "short explanation of the grade",
  "invalidation": "what would invalidate this setup",
  "confidence": "High / Medium / Low",
  "full_reasoning": "detailed step-by-step analysis"
}
"""

# ====================== HELPER FUNCTIONS ======================
def encode_image(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    # Convert to RGB if necessary (handles RGBA, etc.)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def analyze_chart(image: Image.Image, model: str = "gpt-4o") -> dict:
    """Send chart to OpenAI Vision and return parsed JSON"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    base64_image = encode_image(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this trading chart thoroughly and return the structured JSON."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=2500,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Clean possible markdown code blocks
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse JSON",
            "raw_response": raw
        }


# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("⚙️ Settings")
    
    model_choice = st.selectbox(
        "Model",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="gpt-4o is more accurate, gpt-4o-mini is cheaper/faster"
    )
    
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Upload a clear chart screenshot  
    2. Make sure the **price axis** is visible  
    3. Click **Analyze Chart**  
    4. Review the signal + A+ grade
    """)
    
    st.markdown("---")
    st.markdown("**Tip:** Higher resolution + clean charts = better results")


# ====================== MAIN UI ======================
st.title("📈 AI Chart Analyzer")
st.caption("Upload a chart screenshot → Get signal + A+ grade")

uploaded_file = st.file_uploader(
    "Upload Chart Screenshot",
    type=["png", "jpg", "jpeg", "webp"],
    help="Best results with high-resolution screenshots from TradingView, MT4/5, etc."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(image, caption="Uploaded Chart", use_container_width=True)
    
    with col2:
        st.info("Image loaded successfully")
        st.write(f"**Size:** {image.size[0]} × {image.size[1]} px")
        st.write(f"**Mode:** {image.mode}")

    # Analyze button
    if st.button("🔍 Analyze Chart", type="primary", use_container_width=True):
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OPENAI_API_KEY not found. Please set it in your environment or .env file.")
            st.stop()

        with st.spinner("Analyzing chart with vision model... This usually takes 5–15 seconds"):
            result = analyze_chart(image, model=model_choice)

        if "error" in result:
            st.error("Failed to parse the AI response as JSON")
            st.code(result.get("raw_response", "No response"), language="text")
        else:
            # ========== RESULTS ==========
            st.success("Analysis complete!")

            # Top metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Signal", result.get("signal", "N/A"))
            m2.metric("A+ Grade", result.get("a_plus_grade", "N/A"))
            m3.metric("Confidence", result.get("confidence", "N/A"))
            m4.metric("Trend", result.get("trend", "N/A"))

            st.markdown("---")

            # Trade Plan
            if result.get("signal") in ["BUY", "SELL"]:
                st.subheader("🎯 Trade Plan")
                t1, t2, t3 = st.columns(3)
                t1.write(f"**Entry:** {result.get('entry', 'N/A')}")
                t2.write(f"**Stop Loss:** {result.get('stop_loss', 'N/A')}")
                t3.write(f"**Risk:Reward:** {result.get('risk_reward', 'N/A')}")

                st.write("**Targets:**")
                for i, target in enumerate(result.get("targets", []), 1):
                    st.write(f"• TP{i}: {target}")

            # Key Levels
            st.subheader("📌 Key Levels")
            levels = result.get("key_levels", {})
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Supports**")
                for s in levels.get("supports", []):
                    st.write(f"• {s}")
            with c2:
                st.write("**Resistances**")
                for r in levels.get("resistances", []):
                    st.write(f"• {r}")

            # Patterns & Indicators
            st.subheader("🕯️ Patterns & Indicators")
            st.write("**Patterns:**", ", ".join(result.get("patterns", [])) or "None detected")
            st.write("**Indicators Summary:**", result.get("indicators_summary", "N/A"))

            # Grade Reasoning
            st.subheader("⭐ A+ Grade Reasoning")
            st.info(result.get("grade_reasoning", "No reasoning provided"))

            # Invalidation
            st.subheader("⚠️ Invalidation")
            st.warning(result.get("invalidation", "N/A"))

            # Full Reasoning
            with st.expander("📄 Full Detailed Reasoning", expanded=False):
                st.write(result.get("full_reasoning", "No detailed reasoning provided"))

            # Raw JSON
            with st.expander("🧾 Raw JSON Output"):
                st.json(result)

else:
    st.info("👆 Upload a chart screenshot to begin analysis")

# Footer
st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only. Not financial advice. Always do your own analysis and risk management.")