"""
app.py — ShopAI: Full Shopping Assistant
Login, Chat (text/image/voice), Browse & Search, Cart, Wishlist,
Orders (with coupons + simulated tracking), Price Alerts, Compare,
Size Finder, Outfit Builder, Reviews & Ratings, Analytics/Evaluation.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os, sys, base64


def html_block(raw: str, **kwargs):
    """
    Renders an HTML string via st.markdown, safely. Multi-line f-strings
    naturally pick up Python's own source-code indentation as literal
    leading spaces — and Markdown treats any line indented 4+ spaces as
    a CODE block, not HTML, which is what silently turns product cards
    into raw text. Stripping each line's leading/trailing whitespace
    before rendering removes that ambiguity entirely.
    """
    cleaned = "\n".join(line.strip() for line in raw.strip("\n").split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True, **kwargs)

from io import BytesIO
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.embedder import MultimodalEmbedder
from src.database import ProductDatabase
from src.pipeline import RAGPipeline
from src.voice import VoiceInput, VoiceOutput
from src.user_store import UserStore
from src.auth import validate_credentials
from src.size_recommender import recommend_size, height_from_feet_inches
from src.pdf_export import export_chat_to_pdf
from src.orders import apply_coupon, get_tracking_info, generate_otp, COUPONS, PAYMENT_METHODS, generate_upi_txn_id
from src.recommender import get_related_products
from src.sizes import get_size_options, default_size
from src.sms import send_otp_sms
from src.maps import build_picker_map, build_tracking_map, reverse_geocode
from src.trends import simulate_price_trend
from streamlit_folium import st_folium
from evaluation.evaluate import run_evaluation
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ShopAI — Smart Shopping Assistant", page_icon="🛍️",
                    layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# NAVY & GOLD THEME
# ─────────────────────────────────────────────
html_block("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ═══════════ GLOBAL ROYAL NAVY CANVAS ═══════════ */
.stApp { 
    background: linear-gradient(180deg, #0A1426 0%, #0D1B33 50%, #07101E 100%) !important; 
    color: #E8E4D6;
}
.stApp p, .stApp li, .stApp label { color: #E8E4D6; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #F5E7B8; }

/* Sidebar Styling */
[data-testid="stSidebar"] { 
    background: linear-gradient(180deg, #07101E, #0D1B33) !important; 
    border-right: 1px solid rgba(212, 175, 55, 0.25);
}

/* Container Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #10203B !important; 
    border: 1px solid rgba(212, 175, 55, 0.35) !important; 
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}

/* ═══════════ ANIMATIONS: WATER SLIDE & GOLD GLOW ═══════════ */
@keyframes slideBubbleIn {
    0% {
        opacity: 0;
        transform: translateY(35px) scale(0.92);
    }
    60% {
        opacity: 1;
        transform: translateY(-5px) scale(1.02);
    }
    100% {
        opacity: 1;
        transform: translateY(0px) scale(1);
    }
}

@keyframes goldPulse {
    0%, 100% {
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4), 0 0 10px rgba(212, 175, 55, 0.3);
        border-color: #D4AF37;
    }
    50% {
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.6), 0 0 22px rgba(255, 215, 0, 0.6);
        border-color: #FFD700;
    }
}

/* ═══════════ DARK BUBBLE CHAT OVERRIDES ═══════════ */

/* Bot & User Custom Bubbles with Water Slide Entrance */
.msg-bot, .msg-user {
    background: #142645 !important; 
    color: #F5E7B8 !important; 
    padding: 1.1rem 1.4rem !important;
    border-radius: 20px !important; 
    margin: 0.5rem 0 !important;
    border: 1.5px solid #D4AF37 !important;
    animation: slideBubbleIn 0.65s cubic-bezier(0.25, 1, 0.5, 1) forwards, goldPulse 3.5s ease-in-out infinite !important;
    font-size: 0.95rem; 
    line-height: 1.6;
}

.msg-bot *, .msg-bot p, .msg-bot li, .msg-bot span, .msg-bot div,
.msg-user *, .msg-user p, .msg-user li, .msg-user span, .msg-user div {
    color: #F5E7B8 !important;
    -webkit-text-fill-color: #F5E7B8 !important;
}

/* Native Streamlit Chat Message Wrapper Overrides */
[data-testid="stChatMessage"],
[data-testid="stChatMessageContent"] {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
}

/* ═══════════ PRODUCT CARD NAVY OVERRIDE ═══════════ */
.product-card {
    background: #10203B !important; 
    border-radius: 16px; 
    padding: 1rem; 
    margin: 0.5rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4); 
    border: 1.5px solid #D4AF37 !important; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
    position: relative; 
    overflow: hidden;
    animation: slideBubbleIn 0.6s cubic-bezier(0.25, 1, 0.5, 1) forwards !important;
}
.product-card *, .product-card p, .product-card span, .product-card div {
    color: #F5E7B8 !important;
    -webkit-text-fill-color: #F5E7B8 !important;
}
.product-card:hover { 
    transform: translateY(-4px); 
    box-shadow: 0 0 25px rgba(212,175,55,0.6); 
    border-color: #FFD700 !important; 
}

.product-name { font-weight: 700 !important; font-size: 1rem; color: #F5E7B8 !important; margin-bottom: 4px; }
.product-price { color: #FFD700 !important; font-size: 1.15rem; font-weight: 700; }
.product-meta { color: #D9C98A !important; font-size: 0.82rem; margin-top: 4px; }
.match-score { background: #142645 !important; color: #FFD700 !important; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; display: inline-block; border: 1px solid rgba(212,175,55,0.5); }
.in-stock { background: #0D2818 !important; color: #4ADE80 !important; padding: 2px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; display: inline-block; border: 1px solid #22C55E; }

/* ═══════════ FORM INPUTS & INTERACTIVE BUTTONS ═══════════ */
.stTextInput input, 
.stTextArea textarea, 
.stNumberInput input, 
.stChatInput textarea,
[data-testid="stChatInput"] textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background: #142645 !important; 
    color: #F5E7B8 !important; 
    -webkit-text-fill-color: #F5E7B8 !important;
    border: 1px solid #2A3F63 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease-in-out !important;
}

.stTextInput input:focus, 
.stTextArea textarea:focus, 
.stChatInput textarea:focus,
[data-testid="stChatInput"] textarea:focus,
.stSelectbox div[data-baseweb="select"] > div:focus-within {
    border-color: #D4AF37 !important;
    box-shadow: 0 0 14px rgba(212, 175, 55, 0.4) !important;
    color: #F5E7B8 !important;
    -webkit-text-fill-color: #F5E7B8 !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background-color: #0A1F44 !important; 
    border: 1px solid rgba(212, 175, 55, 0.35) !important;
    border-radius: 14px !important;
}

[data-baseweb="textarea"],
[data-baseweb="base-input"] {
    background-color: transparent !important;
}

[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stChatInputContainer"] {
    background-color: transparent !important;
    background: transparent !important;
}

input:-webkit-autofill,
input:-webkit-autofill:hover, 
input:-webkit-autofill:focus,
input:-webkit-autofill:active {
    -webkit-box-shadow: 0 0 0 1000px #142645 inset !important;
    -webkit-text-fill-color: #F5E7B8 !important;
    transition: background-color 5000s ease-in-out 0s !important;
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid #2A3F63 !important;
    background: linear-gradient(135deg, #142645 0%, #1B3A6B 100%) !important;
    color: #E8E4D6 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.015) !important;
    box-shadow: 0 0 22px rgba(212, 175, 55, 0.5), 0 8px 25px rgba(0, 0, 0, 0.4) !important;
    border-color: #D4AF37 !important;
    color: #F5E7B8 !important;
}

.stButton > button:active {
    transform: translateY(-1px) scale(0.98) !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #D4AF37 0%, #AA820A 100%) !important;
    color: #07101E !important;
    border: 1px solid #F5E7B8 !important;
    font-weight: 700 !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #F5E7B8 0%, #D4AF37 100%) !important;
    box-shadow: 0 0 30px rgba(212, 175, 55, 0.75) !important;
    color: #07101E !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    gap: 6px; background: #0A1F44; padding: 6px; border-radius: 14px; border: 1px solid #2A3F63;
}
[data-baseweb="tab"] {
    border-radius: 10px !important; padding: 8px 16px !important; font-weight: 600 !important;
    color: #E8E4D6 !important; transition: all 0.2s ease !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, #142645, #1B3A6B) !important; color: #F5E7B8 !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35) !important; border: 1px solid #D4AF37 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ═══════════ FILE UPLOADER BOX STYLING ═══════════ */
[data-testid="stFileUploader"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    animation: none !important;
    padding: 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #142645 !important;
    border: 1.5px solid #D4AF37 !important;
    border-radius: 14px !important;
    animation: goldPulse 3.5s infinite ease-in-out !important;
}

[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] small {
    color: #F5E7B8 !important;
    -webkit-text-fill-color: #F5E7B8 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background-color: #0A1F44 !important;
    color: #D4AF37 !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 8px !important;
}
/* ═══════════ ROYAL LOGIN SCREEN ═══════════ */
.royal-topbar {
    background: linear-gradient(160deg, #030714 0%, #0A1F44 60%, #030714 100%);
    border-radius: 20px 20px 0 0; padding: 1.6rem 1rem 0.8rem 1rem; text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}
.royal-logo {
    font-size: 2.4rem; font-weight: 800; color: #F5E7B8; letter-spacing: -1px;
    text-shadow: 0 0 30px rgba(212,175,55,0.5); animation: fadeInDown 0.8s ease;
}
.royal-tagline {
    color: #D9C98A; font-size: 1.02rem; margin: 0.4rem 0 1rem 0; font-style: italic;
    animation: fadeInDown 0.8s ease 0.15s both;
}
.promo-ticker {
    width: 100%; overflow: hidden; white-space: nowrap; border-top: 1px solid rgba(212,175,55,0.35);
    padding: 8px 0;
}
.promo-ticker span {
    display: inline-block; padding-left: 100%; color: #F5E7B8; font-size: 0.86rem;
    font-weight: 600; animation: tickerScroll 22s linear infinite;
}
@keyframes tickerScroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}
/* Seamless video + form panel, scoped via a marker element so it
   never affects any other column layout elsewhere in the app */
div[data-testid="stMarkdown"]:has(.login-panel-marker) ~ div[data-testid="stHorizontalBlock"] {
    box-shadow: 0 15px 45px rgba(0,0,0,0.35); border-radius: 20px; overflow: hidden;
    background: #0A1F44;
}
.login-video-wrap {
    position: relative; height: 100%; min-height: 420px; background: #0A1F44;
    display: flex; align-items: center; justify-content: center; overflow: hidden;
}
.login-video-wrap video {
    width: 100%; height: 100%; object-fit: cover; display: block;
}
/* Fades the video's right edge into the dark panel color, so however much
   white margin the animation happens to show at any moment (the character's
   arm/bags swing, so it varies frame to frame) always blends smoothly
   instead of showing as a hard white patch. */
.login-video-wrap::after {
    content: ''; position: absolute; top: 0; right: 0; bottom: 0; width: 110px;
    background: linear-gradient(to right, rgba(10,31,68,0) 0%, #0A1F44 88%);
    pointer-events: none;
}
div[data-testid="stMarkdown"]:has(.login-panel-marker)
    ~ div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(2) {
    background: #0A1F44; padding: 1.8rem 2rem;
}
.login-required-banner {
    background: linear-gradient(135deg, #1B3A6B, #0A1F44); color: #F5E7B8;
    border: 1px solid #D4AF37; border-radius: 14px; padding: 0.9rem 1.2rem;
    font-size: 0.9rem; margin-bottom: 0.8rem; text-align: center;
    box-shadow: 0 6px 20px rgba(10,31,68,0.3); animation: fadeInUp 0.5s ease;
}
/* Glowing, interactive primary buttons everywhere */
button[kind="primary"] { transition: all 0.2s ease !important; }
button[kind="primary"]:hover {
    box-shadow: 0 0 18px rgba(212,175,55,0.55) !important;
    transform: translateY(-1px) scale(1.01);
}

</style>
<script>
window.addEventListener('load', () => { document.title = "ShopAI — Premium Shopping Assistant"; });
</script>

""")


# ─────────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────────
@st.cache_resource
def load_system():
    with st.spinner("🚀 Loading AI models..."):
        embedder = MultimodalEmbedder()
        db = ProductDatabase(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
        if db.get_count() == 0:
            df = pd.read_csv(os.getenv("CATALOG_PATH", "./data/abo_products.csv"))
            embed_images = os.getenv("EMBED_CATALOG_IMAGES", "false").lower() == "true"
            db.index_catalog(df, embedder, embed_images=embed_images)
        pipe = RAGPipeline(embedder, db)
        vin = VoiceInput()
        vout = VoiceOutput(language=os.getenv("VOICE_LANGUAGE", "en-IN"))
    return pipe, vin, vout


@st.cache_resource
def load_user_store():
    return UserStore(os.getenv("USER_DB_PATH", "./data/users.db"))


pipeline, voice_in, voice_out = load_system()
store = load_user_store()


def get_price_bounds():
    try:
        meta = pipeline.database.collection.get(include=["metadatas"])["metadatas"]
        prices = [m["price"] for m in meta if "price" in m]
        return (min(prices), max(prices)) if prices else (0, 10000)
    except Exception:
        return (0, 10000)


def get_all_products_df():
    try:
        meta = pipeline.database.collection.get(include=["metadatas"])["metadatas"]
        return pd.DataFrame(meta)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [("user", None), ("auth_mode", "login"), ("compare_ids", []),
                      ("messages", []), ("applied_coupon", None), ("checkout_step", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# LOGIN / REGISTER
# ─────────────────────────────────────────────
@st.cache_resource
def get_login_video_base64():
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "login_video.mp4")
    try:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


def render_auth_screen():
    video_b64 = get_login_video_base64()

    html_block("""
    <div class="royal-topbar">
        <div class="royal-logo">🛍️ ShopAI</div>
        <div class="royal-tagline">Where Intelligent Shopping Meets Royal Style</div>
        <div class="promo-ticker">
            <span>✨ FLAT 20% OFF your first order &nbsp;&nbsp;•&nbsp;&nbsp;
            🚚 FREE shipping over ₹999 &nbsp;&nbsp;•&nbsp;&nbsp;
            🔥 New arrivals every week &nbsp;&nbsp;•&nbsp;&nbsp;
            🤖 AI-powered search, powered by CLIP + LLaMA &nbsp;&nbsp;•&nbsp;&nbsp;
            ✨ FLAT 20% OFF your first order &nbsp;&nbsp;•&nbsp;&nbsp;
            🚚 FREE shipping over ₹999</span>
        </div>
    </div>
    """)

    st.markdown('<span class="login-panel-marker"></span>', unsafe_allow_html=True)

    col_video, col_form = st.columns([0.8, 1.2])

    with col_video:
        if video_b64:
            html_block(f"""
            <div class="login-video-wrap">
                <video autoplay loop muted playsinline>
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                </video>
            </div>
            """)
        else:
            st.info("Add a video at assets/login_video.mp4 to show it here.")

    with col_form:
        html_block("""
        <div class="login-required-banner">
            🔒 <b>Sign in required</b> — create a free account to unlock your personal cart,
            wishlist, price alerts, order tracking, and AI-personalized recommendations.
        </div>
        """)

        tab_login, tab_register = st.tabs(["🔑 Log In", "✨ Create Account"])

        with tab_login:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")
            if submitted:
                user = store.authenticate(u, p)
                if user:
                    st.session_state.user = user
                    st.session_state.messages = [
                        {"role": m["role"], "content": m["content"], "products": m["products"]}
                        for m in store.get_history(user["id"])
                    ]
                    if not st.session_state.messages:
                        welcome = (
                            "Hello! I'm ShopAI, your smart shopping assistant. 👋\n\n"
                            "Type your query, upload a photo, or click **🎤 Speak**. "
                            "Try: *'Show me a blue jacket under ₹2000'*"
                        )
                        st.session_state.messages.append({"role": "assistant", "content": welcome, "products": []})
                        store.save_message(user["id"], "assistant", welcome, [])
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        with tab_register:
            with st.form("register_form"):
                ru = st.text_input("Choose a username")
                rp = st.text_input("Choose a password", type="password")
                rp2 = st.text_input("Confirm password", type="password")
                r_submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if r_submitted:
                err = validate_credentials(ru, rp)
                if err:
                    st.error(err)
                elif rp != rp2:
                    st.error("Passwords don't match.")
                else:
                    ok, msg = store.create_user(ru, rp)
                    if ok:
                        st.success(msg + " Switch to the Log In tab.")
                    else:
                        st.error(msg)


if not st.session_state.user:
    render_auth_screen()
    st.stop()

user = st.session_state.user
user_id = user["id"]

# ─────────────────────────────────────────────
# PRICE ALERT CHECK (once per session load)
# ─────────────────────────────────────────────
def get_current_price_lookup():
    try:
        all_meta = pipeline.database.collection.get(include=["metadatas"])
        return {m["product_id"]: m["price"] for m in all_meta["metadatas"]}
    except Exception:
        return {}


if "alerts_checked" not in st.session_state:
    st.session_state.newly_triggered_alerts = store.check_alerts(user_id, get_current_price_lookup())
    st.session_state.alerts_checked = True

if st.session_state.get("newly_triggered_alerts"):
    for a in st.session_state.newly_triggered_alerts:
        p = a["product"]
        html_block(f"""<div class="alert-banner">🔔 Price drop! <b>{p.get('name')}</b> is now
        ₹{a.get('current_price')} (your threshold was ₹{a['threshold_price']:.0f}).</div>""")

# ─────────────────────────────────────────────
# NAVBAR (Amazon/Flipkart style — dark brand strip + toolbar)
# ─────────────────────────────────────────────
cart_count = len(store.get_cart(user_id))
wishlist_count = len(store.get_wishlist(user_id))

if "active_page" not in st.session_state:
    st.session_state.active_page = "chat"

html_block(f"""
<div class="navbar-strip">
    <span class="navbar-logo">🛍️ ShopAI</span>
    <span class="navbar-tagline">Text · Image · Voice powered shopping</span>
</div>
""")

nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([3, 1, 1, 1, 1.2])
with nav_col1:
    quick_search = st.text_input("Search", placeholder="🔍 Search for jackets, dresses, shoes...",
                                  label_visibility="collapsed", key="navbar_search")
    if quick_search:
        st.session_state["pending_query"] = quick_search
        st.session_state.active_page = "chat"
        st.rerun()
with nav_col2:
    if st.button(f"🤖 Assistant", use_container_width=True,
                 type="primary" if st.session_state.active_page == "chat" else "secondary"):
        st.session_state.active_page = "chat"
        st.rerun()
with nav_col3:
    if st.button(f"❤️ Wishlist ({wishlist_count})", use_container_width=True,
                 type="primary" if st.session_state.active_page == "wishlist" else "secondary"):
        st.session_state.active_page = "wishlist"
        st.rerun()
with nav_col4:
    if st.button(f"🛒 Cart ({cart_count})", use_container_width=True,
                 type="primary" if st.session_state.active_page == "cart" else "secondary"):
        st.session_state.active_page = "cart"
        st.rerun()
with nav_col5:
    if st.button(f"👤 {user['username']}", use_container_width=True,
                 type="primary" if st.session_state.active_page == "account" else "secondary"):
        st.session_state.active_page = "account"
        st.rerun()

SECONDARY_NAV = [
    ("browse", "🗂️ Browse"), ("orders", "📦 Orders"), ("alerts", "🔔 Alerts"),
    ("compare", "⚖️ Compare"), ("size", "📏 Size Finder"),
    ("outfit", "👗 Outfit Builder"), ("analytics", "📊 Analytics"),
    ("trends", "📈 Trends"),
]
if user.get("is_admin"):
    SECONDARY_NAV.append(("admin", "🛠️ Admin"))
sec_cols = st.columns(len(SECONDARY_NAV))
for i, (pid, label) in enumerate(SECONDARY_NAV):
    with sec_cols[i]:
        if st.button(label, key=f"secnav_{pid}", use_container_width=True,
                     type="primary" if st.session_state.active_page == pid else "secondary"):
            st.session_state.active_page = pid
            st.rerun()

st.markdown("<div style='margin-bottom:0.6rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
price_min, price_max = get_price_bounds()
if "price_range" not in st.session_state:
    st.session_state.price_range = (int(price_min), int(price_max))

PRICE_PRESETS = {
    "All": (int(price_min), int(price_max)),
    "Under ₹1,000": (int(price_min), 1000),
    "₹1,000 – ₹3,000": (1000, 3000),
    "Above ₹3,000": (3000, int(price_max)),
}

with st.sidebar:
    with st.container(border=True):
        st.markdown(f"**👤 {user['username']}**")
        st.caption("Signed in")
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.pop("alerts_checked", None)
            st.rerun()

    with st.container(border=True):
        st.markdown("**🎤 Voice**")
        voice_input_on = st.toggle("Voice Input", value=True)
        voice_output_on = st.toggle("🔊 Voice Output (reads answers aloud)", value=False)
        if voice_output_on:
            if st.button("⏹️ Stop Speaking Now", use_container_width=True, type="primary"):
                voice_out.stop()

    with st.container(border=True):
        st.markdown("**🔍 Filters**")
        preset_cols = st.columns(2)
        preset_names = list(PRICE_PRESETS.keys())
        for i, name in enumerate(preset_names):
            with preset_cols[i % 2]:
                if st.button(name, key=f"preset_{name}", use_container_width=True):
                    st.session_state.price_range = PRICE_PRESETS[name]
                    st.rerun()

        price_range = st.slider(
            "Custom range (₹)", min_value=int(price_min), max_value=int(price_max),
            value=st.session_state.price_range, step=50, key="price_range_slider"
        )
        st.session_state.price_range = price_range

        sort_by = st.selectbox(
            "Sort by", ["Best Match", "Price: Low to High", "Price: High to Low", "Name: A-Z"]
        )

        colors = ["Any", "Blue", "Red", "Black", "White", "Grey", "Green", "Yellow", "Pink",
                  "Brown", "Multicolor", "Navy Blue", "Olive Green", "Maroon", "Beige"]
        color_filter = st.selectbox("Color", colors)
        if color_filter == "Any":
            color_filter = None

    with st.container(border=True):
        st.markdown("**🛠️ Tools**")
        if st.button("🔄 New Conversation", use_container_width=True):
            pipeline.reset()
            st.session_state.messages = []
            store.clear_history(user_id)
            st.rerun()
        if st.button("🔇 Stop Voice", use_container_width=True):
            voice_out.stop()
        if st.button("📄 Export Chat as PDF", use_container_width=True):
            out_dir = "/tmp/shopai_exports"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"shopai_chat_{user['username']}.pdf")
            export_chat_to_pdf(user["username"], st.session_state.messages, out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name="shopai_conversation.pdf",
                                    mime="application/pdf", use_container_width=True)

    st.caption(f"📦 {pipeline.database.get_count()} products · CLIP + ChromaDB + LLaMA 3")
    st.caption("Dataset: Amazon Berkeley Objects (ABO)")


def sort_products(products, sort_by):
    if sort_by == "Price: Low to High":
        return sorted(products, key=lambda p: p.get("price", 0))
    if sort_by == "Price: High to Low":
        return sorted(products, key=lambda p: p.get("price", 0), reverse=True)
    if sort_by == "Name: A-Z":
        return sorted(products, key=lambda p: p.get("name", ""))
    return products  # Best Match = keep retrieval order


def max_price_filter():
    return price_range[1] if price_range[1] < price_max else None


def min_price_ok(p):
    return price_range[0] <= p.get("price", 0) <= price_range[1]


# ─────────────────────────────────────────────
# SHARED: PRODUCT CARD
# ─────────────────────────────────────────────
def render_stars(avg):
    full = int(round(avg))
    return "★" * full + "☆" * (5 - full)


import hashlib as _hashlib

BADGE_POOL = [
    ("🔥 Bestseller", "#D4453B"),
    ("🆕 New Arrival", "#1B8A5A"),
    ("⭐ Top Rated", "#A9791F"),
    ("⚡ Trending", "#7B4FCC"),
    (None, None),  # some products get no badge — keeps it believable
]


def get_product_badge(product_id: str):
    seed = int(_hashlib.md5(str(product_id).encode()).hexdigest(), 16)
    return BADGE_POOL[seed % len(BADGE_POOL)]


def get_discount_pricing(product_id: str, price: float):
    """
    Amazon/Flipkart-style 'M.R.P ₹X, now ₹Y (Z% off)' display. The
    discount percentage is derived deterministically from the product
    id (10-40%), so it's stable across reruns instead of flickering.
    """
    seed = int(_hashlib.md5((str(product_id) + "disc").encode()).hexdigest(), 16)
    pct = 10 + (seed % 31)  # 10–40%
    mrp = round(price / (1 - pct / 100))
    return mrp, pct


def render_product_card(p, key_prefix):
    pid = str(p.get("product_id"))
    summary = store.review_summary(pid)
    badge_label, badge_color = get_product_badge(pid)

    badge_html = (
        f'<div class="ribbon" style="background:{badge_color}">{badge_label}</div>'
        if badge_label else ""
    )

    html_block(f"""
<div class="product-card">
    {badge_html}
    <div class="product-name">{p.get('name','')}</div>
    <div class="product-meta">🎨 {p.get('color','')} &nbsp;|&nbsp; 🏷️ {p.get('brand','')}</div>
    <div style="margin-top:6px">
        <span class="stars">{render_stars(summary['average'])}</span>
        <span style="color:#888;font-size:0.8rem"> ({summary['count']} reviews)</span>
    </div>
    <div style="margin-top:8px">
        <span class="in-stock">✓ {p.get('stock','')}</span>
        &nbsp;
        <span class="match-score">⚡ {p.get('similarity_score', '')}{'%' if p.get('similarity_score') is not None else ''} match</span>
    </div>
</div>
""")
    if p.get("image_url"):
        st.image(p["image_url"], use_container_width=True)

    # ── Size selector with live per-size pricing ──
    size_options = get_size_options(p)
    size_labels = [f"{s}  ·  ₹{price:,}" for s, price in size_options]
    size_key = f"size_{key_prefix}_{pid}"
    default_idx = len(size_options) // 2
    chosen_label = st.selectbox("Size", size_labels, index=default_idx, key=size_key)
    chosen_size, chosen_price = size_options[size_labels.index(chosen_label)]

    mrp, pct = get_discount_pricing(pid, chosen_price)
    html_block(f"""
    <div class="price-row">
        <span class="mrp">M.R.P: <s>₹{mrp:,}</s></span>
        <span class="discount-pct">{pct}% off</span>
    </div>
    <div class="live-price">₹{chosen_price:,}</div>
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🛒 Add to Cart", key=f"cart_{key_prefix}_{pid}", use_container_width=True):
            item = {**p, "price": chosen_price, "size": chosen_size}
            store.add_to_cart(user_id, item)
            st.toast(f"Added {p.get('name')} (Size {chosen_size}) to cart")
    with c2:
        if st.button("❤️ Save", key=f"wish_{key_prefix}_{pid}", use_container_width=True):
            item = {**p, "price": chosen_price, "size": chosen_size}
            store.add_to_wishlist(user_id, item)
            st.toast(f"Saved {p.get('name')} to wishlist")
    with c3:
        in_compare = pid in st.session_state.compare_ids
        label = "✓ Comparing" if in_compare else "⚖️ Compare"
        if st.button(label, key=f"cmp_{key_prefix}_{pid}", use_container_width=True):
            if in_compare:
                st.session_state.compare_ids.remove(pid)
            elif len(st.session_state.compare_ids) < 4:
                st.session_state.compare_ids.append(pid)
            else:
                st.warning("You can compare up to 4 products at a time.")
            st.rerun()

    with st.expander("📖 Details, Reviews & Similar Items"):
        store.log_view(user_id, p)
        st.write(p.get("description", ""))

        threshold = st.number_input("🔔 Alert me when price drops below (₹)", min_value=0,
                                     value=max(int(chosen_price) - 100, 0), step=50,
                                     key=f"alertnum_{key_prefix}_{pid}")
        if st.button("Set Price Alert", key=f"alertbtn_{key_prefix}_{pid}"):
            store.add_price_alert(user_id, {**p, "price": chosen_price, "size": chosen_size}, threshold)
            st.toast(f"🔔 Alert set for ₹{threshold} — check the 🔔 Alerts page anytime.")
            st.success(f"Alert set — you'll be notified when price drops below ₹{threshold}. "
                       f"View it under the 🔔 Alerts page.")

        st.markdown("---")
        st.markdown(f"**⭐ Reviews** ({summary['count']}, avg {summary['average']}/5)")
        reviews = store.get_reviews(pid)
        for r in reviews[:5]:
            st.markdown(f"<span class='stars'>{render_stars(r['rating'])}</span> "
                        f"**{r['reviewer_name']}** — {r['comment']}", unsafe_allow_html=True)

        with st.form(key=f"reviewform_{key_prefix}_{pid}"):
            rating = st.slider("Your rating", 1, 5, 5, key=f"ratingslider_{key_prefix}_{pid}")
            comment = st.text_input("Your review", key=f"reviewtext_{key_prefix}_{pid}")
            if st.form_submit_button("Submit Review"):
                store.add_review(pid, user["username"], rating, comment, user_id=user_id)
                st.success("Thanks for your review!")
                st.rerun()

        related = get_related_products(pipeline.database, p, k=4)
        if related:
            st.markdown("---")
            st.markdown("**🔎 You might also like**")
            rcols = st.columns(len(related))
            for i, rp in enumerate(related):
                with rcols[i]:
                    if rp.get("image_url"):
                        st.image(rp["image_url"], use_container_width=True)
                    st.caption(f"{rp.get('name','')[:28]}\n₹{rp.get('price','')}")


def render_line_item(p, actions: list):
    """Compact row for Cart/Wishlist — shows the size/price already
    chosen, instead of re-showing a full interactive product card."""
    img_col, info_col, action_col = st.columns([1, 3, 2])
    with img_col:
        if p.get("image_url"):
            st.image(p["image_url"], use_container_width=True)
    with info_col:
        size_tag = f' <span class="size-tag">Size: {p.get("size")}</span>' if p.get("size") else ""
        html_block(f"""
        <div class="line-item-name">{p.get('name','')}</div>
        <div class="product-meta">🎨 {p.get('color','')} &nbsp;|&nbsp; 🏷️ {p.get('brand','')}{size_tag}</div>
        <div class="live-price">₹{p.get('price', 0):,}</div>
        """)
    with action_col:
        for label, key, callback in actions:
            if st.button(label, key=key, use_container_width=True):
                callback()
                st.rerun()
    st.markdown('<hr class="line-divider">', unsafe_allow_html=True)


def render_product_grid(products, key_prefix):
    if not products:
        return
    cols = st.columns(min(len(products), 3))
    for i, p in enumerate(products[:3]):
        with cols[i % 3]:
            render_product_card(p, f"{key_prefix}_{i}")


# ─────────────────────────────────────────────
# PAGE ROUTER (session-state driven, so navbar clicks can jump pages)
# ─────────────────────────────────────────────
page = st.session_state.active_page

# ══════════════════════ CHAT ══════════════════════
if page == "chat":
    QUICK_CHIPS = [
        ("🧥 Jackets", "Show me jackets"),
        ("👗 Dresses", "Show me dresses"),
        ("👟 Shoes", "Show me shoes"),
        ("💰 Under ₹1,000", "Show me products under ₹1000"),
        ("🔥 Trending", "What are your most popular products right now?"),
        ("🎉 Party Wear", "Show me something for a party"),
    ]
    st.markdown('<div class="chip-row-label">Try asking:</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(QUICK_CHIPS))
    for i, (label, query) in enumerate(QUICK_CHIPS):
        with chip_cols[i]:
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                st.session_state["pending_query"] = query
                st.rerun()

    recently_viewed = store.get_recently_viewed(user_id, limit=6)
    if recently_viewed:
        with st.expander(f"🕒 Recently Viewed ({len(recently_viewed)})"):
            rcols = st.columns(min(len(recently_viewed), 6))
            for i, rp in enumerate(recently_viewed):
                with rcols[i % 6]:
                    if rp.get("image_url"):
                        st.image(rp["image_url"], use_container_width=True)
                    st.caption(f"{rp.get('name','')[:22]}\n₹{rp.get('price','')}")

    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown('<div class="msg-icon-user">👤</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("image"):
                try:
                    img = Image.open(BytesIO(base64.b64decode(msg["image"])))
                    c1, c2, c3 = st.columns([3, 2, 3])
                    with c2:
                        st.image(img, caption="Your photo", use_container_width=True)
                except Exception:
                    pass
        else:
            st.markdown('<div class="msg-icon-bot">🤖</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("products"):
                render_product_grid(msg["products"], key_prefix=f"hist{idx}")

    st.markdown("---")
    col_upload, col_voice = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader("📷 Upload product photo (optional)",
                                     type=["jpg", "jpeg", "png", "webp"], key="chat_uploader")
        if uploaded:
            img_preview_col, _ = st.columns([1, 3])
            with img_preview_col:
                st.image(uploaded, caption="Your photo", width=120)

    with col_voice:
        st.write("")
        st.write("")
        if voice_input_on:
            if st.button("🎤  Speak", use_container_width=True, type="primary"):
                with st.spinner("🎤 Listening... speak now!"):
                    result = voice_in.listen(timeout=6)
                if result["success"]:
                    st.session_state["vq"] = result["text"]
                    st.success(f'Heard: "{result["text"]}"')
                    st.rerun()
                else:
                    st.error(result["error"])

    def process(user_query, uploaded_file, max_price, color_filter):
        query_image = None
        img_b64 = None
        if uploaded_file:
            query_image = Image.open(uploaded_file).convert("RGB")
            buf = BytesIO()
            query_image.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

        st.session_state.messages.append({"role": "user", "content": user_query, "image": img_b64})
        store.save_message(user_id, "user", user_query, [])

        st.markdown('<div class="msg-icon-user">👤</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-user">{user_query}</div>', unsafe_allow_html=True)

        thinking = st.empty()
        thinking.markdown("""
        <div class="speaking-indicator">
            🤖 <span class="dot"></span><span class="dot"></span><span class="dot"></span>
            Finding the best matches...
        </div>
        """, unsafe_allow_html=True)
        result = pipeline.chat(query=user_query, image=query_image,
                                max_price=max_price, color=color_filter)
        thinking.empty()

        products = [p for p in result["products"] if min_price_ok(p)]
        products = sort_products(products, sort_by)
        answer = result["answer"]

        st.markdown('<div class="msg-icon-bot">🤖</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-bot">{answer}</div>', unsafe_allow_html=True)

        if voice_output_on:
            voice_out.speak(answer)
            sp_col1, sp_col2 = st.columns([4, 1])
            with sp_col1:
                html_block("""<div class="speaking-indicator"><span class="dot"></span><span class="dot"></span><span class="dot"></span>Speaking...</div>""")
            with sp_col2:
                if st.button("⏹️ Stop", key=f"stop_speak_{len(st.session_state.messages)}"):
                    voice_out.stop()

        render_product_grid(products, key_prefix=f"new{len(st.session_state.messages)}")
        st.session_state.messages.append({"role": "assistant", "content": answer, "products": products})
        store.save_message(user_id, "assistant", answer, products)

    if st.session_state.get("vq"):
        vq = st.session_state.pop("vq")
        process(vq, uploaded, max_price_filter(), color_filter)
        st.rerun()

    if st.session_state.get("pending_query"):
        pq = st.session_state.pop("pending_query")
        process(pq, uploaded, max_price_filter(), color_filter)
        st.rerun()

    if prompt := st.chat_input("Ask me anything about products..."):
        process(prompt, uploaded, max_price_filter(), color_filter)
        st.rerun()

# ══════════════════════ BROWSE & SEARCH ══════════════════════
if page == "browse":
    st.markdown('<div class="section-title">🗂️ Browse the Catalog</div>', unsafe_allow_html=True)
    all_df = get_all_products_df()
    if all_df.empty:
        st.info("Catalog not loaded yet.")
    else:
        search_text = st.text_input("🔎 Search by name or description")
        categories = ["All"] + sorted(all_df["category"].dropna().unique().tolist())
        selected_cat = st.selectbox("Category", categories)

        filtered = all_df.copy()
        if selected_cat != "All":
            filtered = filtered[filtered["category"] == selected_cat]
        if search_text:
            s = search_text.lower()
            filtered = filtered[
                filtered["name"].str.lower().str.contains(s, na=False) |
                filtered["description"].str.lower().str.contains(s, na=False)
            ]
        filtered = filtered[filtered["price"].between(price_range[0], price_range[1])]

        st.caption(f"{len(filtered)} products found")
        products_list = filtered.to_dict("records")
        products_list = sort_products(products_list, sort_by)
        cols = st.columns(3)
        for i, p in enumerate(products_list[:30]):
            with cols[i % 3]:
                render_product_card(p, f"browse_{i}")

# ══════════════════════ CART ══════════════════════
if page == "cart":
    st.markdown('<div class="section-title">🛒 Your Cart</div>', unsafe_allow_html=True)

    # Show a real confirmation screen for the order just placed — set
    # in session_state right BEFORE the rerun below, so it survives
    # into this fresh page load instead of flashing and disappearing.
    just_placed = st.session_state.pop("order_confirmation", None)
    if just_placed:
        html_block(f"""
        <div class="order-confirm">
            <div class="order-confirm-icon">✅</div>
            <h2>Order Placed Successfully!</h2>
            <p>Order <b>#{just_placed['order_id']}</b> · {just_placed['item_count']} item(s) ·
            Total <b>₹{just_placed['total']:,.0f}</b></p>
            <p class="order-confirm-sub">Estimated delivery in 4–6 days. Track progress under 📦 Orders.</p>
        </div>
        """)
        oc1, oc2 = st.columns(2)
        with oc1:
            if st.button("📦 View Order Status", use_container_width=True):
                st.session_state.active_page = "orders"
                st.rerun()
        with oc2:
            if st.button("🛍️ Continue Shopping", use_container_width=True):
                st.session_state.active_page = "chat"
                st.rerun()
        st.markdown("---")

    cart_items = store.get_cart(user_id)
    checkout_step = st.session_state.get("checkout_step")

    if not cart_items and not checkout_step:
        if not just_placed:
            st.info("Your cart is empty. Add products from Chat or Browse!")

    elif checkout_step == "address":
        st.markdown("### 📍 Shipping Details")
        st.caption("Step 1 of 3 — click your delivery location on the map, then confirm the details below.")

        picked = st.session_state.get("checkout_map_pick")
        map_obj = build_picker_map(marker=picked)
        map_result = st_folium(map_obj, height=350, width=None, key="address_picker_map")

        if map_result and map_result.get("last_clicked"):
            lat = map_result["last_clicked"]["lat"]
            lng = map_result["last_clicked"]["lng"]
            if st.session_state.get("checkout_map_pick") != (lat, lng):
                st.session_state.checkout_map_pick = (lat, lng)
                with st.spinner("Looking up address..."):
                    st.session_state.checkout_map_address = reverse_geocode(lat, lng)
                st.rerun()

        if st.session_state.get("checkout_map_address"):
            st.success(f"📍 Selected: {st.session_state.checkout_map_address}")
        else:
            st.info("👆 Click anywhere on the map to set your delivery location.")

        with st.form("shipping_form"):
            recipient_name = st.text_input("Full Name")
            address = st.text_area(
                "Delivery Address (auto-filled from map — edit if needed)",
                value=st.session_state.get("checkout_map_address", ""),
                placeholder="Click the map above, or type your address manually",
            )
            phone = st.text_input("Phone Number", placeholder="10-digit mobile number")
            proceed = st.form_submit_button("Send OTP →", type="primary", use_container_width=True)
        if proceed:
            phone_digits = "".join(ch for ch in phone if ch.isdigit())
            if not recipient_name.strip():
                st.error("Please enter the recipient's name.")
            elif not address.strip():
                st.error("Please enter a delivery address (click the map or type one).")
            elif len(phone_digits) != 10:
                st.error("Please enter a valid 10-digit phone number.")
            else:
                pick = st.session_state.get("checkout_map_pick")
                st.session_state.checkout_shipping = {
                    "recipient_name": recipient_name.strip(),
                    "address": address.strip(),
                    "phone": phone_digits,
                    "lat": pick[0] if pick else None,
                    "lng": pick[1] if pick else None,
                }
                otp = generate_otp()
                st.session_state.checkout_otp = otp
                sent, sms_msg = send_otp_sms(phone_digits, otp)
                st.session_state.checkout_sms_status = (sent, sms_msg)
                st.session_state.checkout_step = "otp"
                st.rerun()
        if st.button("← Back to Cart"):
            st.session_state.checkout_step = None
            st.rerun()

    elif checkout_step == "otp":
        shipping = st.session_state.checkout_shipping
        sms_sent, sms_msg = st.session_state.get("checkout_sms_status", (False, ""))
        st.markdown("### 🔐 Verify OTP")
        st.caption(f"Step 2 of 3 — code sent to +91 {shipping['phone']}")
        if sms_sent:
            html_block(f"""
            <div class="login-required-banner">
                📲 <b>Real SMS sent!</b> {sms_msg} Check your phone for the code.
            </div>
            """)
        else:
            html_block(f"""
            <div class="login-required-banner">
                📱 <b>Demo mode</b> — {sms_msg}<br>Your OTP: 
                <b style="font-size:1.2rem; letter-spacing:3px;">{st.session_state.checkout_otp}</b>
            </div>
            """)
        with st.form("otp_form"):
            entered_otp = st.text_input("Enter 6-digit OTP", max_chars=6)
            confirm = st.form_submit_button("Verify →", type="primary", use_container_width=True)
        if confirm:
            if entered_otp.strip() == st.session_state.checkout_otp:
                st.session_state.checkout_step = "payment"
                st.rerun()
            else:
                st.error("Incorrect OTP. Please check and try again.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Resend OTP"):
                otp = generate_otp()
                st.session_state.checkout_otp = otp
                sent, sms_msg = send_otp_sms(shipping["phone"], otp)
                st.session_state.checkout_sms_status = (sent, sms_msg)
                st.rerun()
        with c2:
            if st.button("← Edit Address"):
                st.session_state.checkout_step = "address"
                st.rerun()

    elif checkout_step == "payment":
        shipping = st.session_state.checkout_shipping
        st.markdown("### 💳 Payment Method")
        st.caption("Step 3 of 3 — how would you like to pay?")

        subtotal = store.cart_total(user_id)
        discount = st.session_state.applied_coupon["discount"] if st.session_state.applied_coupon else 0
        coupon_code = st.session_state.applied_coupon["code"] if st.session_state.applied_coupon else None
        total = max(subtotal - discount, 0)
        st.markdown(f"**Amount payable: ₹{total:,.0f}**")

        method_key = st.radio(
            "Select a payment method",
            options=list(PAYMENT_METHODS.keys()),
            format_func=lambda k: PAYMENT_METHODS[k],
            key="payment_method_choice",
        )

        payment_reference = None
        ready_to_place = False

        if method_key == "upi":
            upi_id = st.text_input("Enter your UPI ID", placeholder="yourname@upi")
            if st.button("Pay with UPI", type="primary", use_container_width=True):
                if "@" not in (upi_id or ""):
                    st.error("Please enter a valid UPI ID (e.g. yourname@upi).")
                else:
                    payment_reference = generate_upi_txn_id()
                    st.session_state.checkout_payment = {
                        "method": method_key, "reference": payment_reference,
                    }
                    st.toast(f"✅ UPI payment successful — Ref {payment_reference}")
                    ready_to_place = True
        else:
            label = "Confirm Cash on Delivery" if method_key == "cod" else "Confirm Cash Payment"
            if st.button(label, type="primary", use_container_width=True):
                st.session_state.checkout_payment = {"method": method_key, "reference": None}
                ready_to_place = True

        if ready_to_place:
            order_id = store.place_order(
                user_id, cart_items, subtotal, discount, coupon_code,
                recipient_name=shipping["recipient_name"],
                shipping_address=shipping["address"],
                phone_number=shipping["phone"],
                delivery_lat=shipping.get("lat"),
                delivery_lng=shipping.get("lng"),
                payment_method=PAYMENT_METHODS[st.session_state.checkout_payment["method"]],
                payment_reference=st.session_state.checkout_payment["reference"],
            )
            store.clear_cart(user_id)
            st.session_state.applied_coupon = None
            st.session_state.checkout_step = None
            for k in ("checkout_shipping", "checkout_otp", "checkout_map_pick",
                      "checkout_map_address", "checkout_sms_status", "checkout_payment"):
                st.session_state.pop(k, None)
            st.session_state["order_confirmation"] = {
                "order_id": order_id, "item_count": len(cart_items), "total": total,
            }
            st.rerun()

        if st.button("← Back"):
            st.session_state.checkout_step = "otp"
            st.rerun()

    else:
        for p in cart_items:
            render_line_item(p, actions=[
                ("🗑️ Remove", f"rm_cart_{p['product_id']}",
                 lambda pid=p["product_id"]: store.remove_from_cart(user_id, pid)),
            ])

        st.markdown("---")
        subtotal = store.cart_total(user_id)

        st.markdown("**🎟️ Have a coupon?**")
        cc1, cc2 = st.columns([3, 1])
        with cc1:
            coupon_input = st.text_input("Coupon code", label_visibility="collapsed",
                                          placeholder="Try WELCOME10, SHOP20, or MCA50")
        with cc2:
            if st.button("Apply", use_container_width=True):
                discount, msg = apply_coupon(subtotal, coupon_input)
                if discount > 0:
                    st.session_state.applied_coupon = {"code": coupon_input.strip().upper(), "discount": discount}
                    st.success(msg)
                else:
                    st.session_state.applied_coupon = None
                    if msg:
                        st.error(msg)

        discount = st.session_state.applied_coupon["discount"] if st.session_state.applied_coupon else 0
        total = max(subtotal - discount, 0)

        st.markdown(f"Subtotal: ₹{subtotal:,.0f}")
        if discount:
            st.markdown(f"Discount: −₹{discount:,.0f}")
        st.markdown(f"### Total: ₹{total:,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Clear Cart", use_container_width=True):
                store.clear_cart(user_id)
                st.session_state.applied_coupon = None
                st.rerun()
        with c2:
            if st.button("✅ Proceed to Checkout", type="primary", use_container_width=True):
                st.session_state.checkout_step = "address"
                st.rerun()

# ══════════════════════ WISHLIST ══════════════════════
if page == "wishlist":
    st.markdown('<div class="section-title">❤️ Your Wishlist</div>', unsafe_allow_html=True)
    st.caption("Items you're interested in but not ready to buy yet.")
    wishlist_items = store.get_wishlist(user_id)
    if not wishlist_items:
        st.info("Your wishlist is empty. Tap ❤️ Save on any product to add it here.")
    else:
        for p in wishlist_items:
            render_line_item(p, actions=[
                ("🛒 Move to Cart", f"move_{p['product_id']}",
                 lambda pid=p["product_id"]: store.move_to_cart(user_id, pid)),
                ("🗑️ Remove", f"rm_wish_{p['product_id']}",
                 lambda pid=p["product_id"]: store.remove_from_wishlist(user_id, pid)),
            ])

# ══════════════════════ ORDERS ══════════════════════
if page == "orders":
    st.markdown('<div class="section-title">📦 Order History</div>', unsafe_allow_html=True)
    orders = store.get_orders(user_id)
    if not orders:
        st.info("No orders yet — place one from your Cart!")
    else:
        for o in orders:
            tracking = get_tracking_info(o["placed_at"])
            placed_dt = datetime.fromisoformat(o["placed_at"]).strftime("%d %b %Y, %I:%M %p")

            stage_html = ""
            for i, (_, label, icon) in enumerate(tracking["stages"]):
                state = "done" if i <= tracking["current_index"] else "pending"
                stage_html += f"""
                <div class="track-stage {state}">
                    <div class="track-dot">{icon}</div>
                    <div class="track-label">{label}</div>
                </div>"""
                if i < len(tracking["stages"]) - 1:
                    line_state = "done" if i < tracking["current_index"] else "pending"
                    stage_html += f'<div class="track-line {line_state}"></div>'

            html_block(f"""
            <div class="order-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b>Order #{o['id']}</b>
                    <span class="status-pill">{tracking['current_icon']} {tracking['current_label']}</span>
                </div>
                <div style="color:#8A94A6; font-size:0.85rem; margin-top:2px;">
                    Placed {placed_dt} &nbsp;|&nbsp; {len(o['items'])} item(s) &nbsp;|&nbsp;
                    Total ₹{o['total']:,.0f}{f" (coupon {o['coupon_code']})" if o['coupon_code'] else ""}
                </div>
                <div class="track-row">{stage_html}</div>
                <div style="color:#8A94A6; font-size:0.82rem; margin-top:6px;">
                    {"🎉 Delivered!" if tracking["is_delivered"] else f"📅 Estimated delivery: {tracking['estimated_delivery']}"}
                </div>
            </div>
            """)

            with st.expander(f"View details — Order #{o['id']}"):
                if o.get("recipient_name") or o.get("shipping_address"):
                    st.markdown(f"**Shipping to:** {o.get('recipient_name', '—')}")
                    st.markdown(f"**Address:** {o.get('shipping_address', '—')}")
                    st.markdown(f"**Phone:** {o.get('phone_number', '—')}")
                if o.get("payment_method"):
                    pay_line = f"**Payment:** {o['payment_method']}"
                    if o.get("payment_reference"):
                        pay_line += f" (Ref: {o['payment_reference']})"
                    st.markdown(pay_line)
                st.markdown("---")
                for item in o["items"]:
                    size_txt = f", Size {item.get('size')}" if item.get("size") else ""
                    st.write(f"• {item.get('name')} — ₹{item.get('price')} ({item.get('color', '')}{size_txt})")

                if o.get("delivery_lat") and o.get("delivery_lng"):
                    st.markdown("---")
                    st.markdown("**📍 Live Route Tracking**")
                    st.caption("Straight-line route from our fulfillment center to your address — "
                               "the truck marker's position reflects delivery progress.")
                    delivery_loc = (float(o["delivery_lat"]), float(o["delivery_lng"]))
                    tracking_map = build_tracking_map(delivery_loc, tracking["percent"] / 100)
                    if tracking_map:
                        st_folium(tracking_map, height=320, width=None, key=f"track_map_{o['id']}")

# ══════════════════════ PRICE ALERTS ══════════════════════
if page == "alerts":
    st.markdown('<div class="section-title">🔔 Price Drop Alerts</div>', unsafe_allow_html=True)
    st.caption("We check your alerts every time you open the app.")

    just_triggered = st.session_state.pop("alert_just_triggered", None)
    if just_triggered:
        st.success(f"🔔 Price drop! {just_triggered['product'].get('name')} is now "
                   f"₹{just_triggered['current_price']:.0f} "
                   f"(your threshold was ₹{just_triggered['threshold_price']:.0f}).")

    if st.button("🔄 Check Alerts Now"):
        triggered = store.check_alerts(user_id, get_current_price_lookup())
        if triggered:
            st.session_state["alert_just_triggered"] = triggered[0]
        else:
            st.session_state["alert_no_drop"] = True
        st.rerun()

    if st.session_state.pop("alert_no_drop", False):
        st.info("No new price drops right now — catalog prices haven't changed since your alerts were set.")

    alerts = store.get_price_alerts(user_id)
    if not alerts:
        st.info("No alerts yet. Set one from any product's 'Details, Reviews & Similar Items' panel.")
    else:
        for a in alerts:
            p = a["product"]
            status = "✅ Triggered" if a["triggered"] else "⏳ Watching"
            html_block(f"""
            <div class="filter-card">
                <b>{p.get('name')}</b><br>
                Alert threshold: ₹{a['threshold_price']:.0f} &nbsp;|&nbsp;
                Current price: ₹{p.get('price')} &nbsp;|&nbsp; {status}
            </div>
            """)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🗑️ Remove Alert", key=f"rm_alert_{a['id']}", use_container_width=True):
                    store.remove_price_alert(user_id, a["id"])
                    st.rerun()
            with b2:
                if not a["triggered"]:
                    if st.button("🧪 Simulate Price Drop (Demo)", key=f"sim_alert_{a['id']}",
                                 use_container_width=True):
                        result = store.simulate_price_drop(user_id, a["id"])
                        st.session_state["alert_just_triggered"] = result
                        st.rerun()

# ══════════════════════ COMPARE ══════════════════════
if page == "compare":
    st.markdown('<div class="section-title">⚖️ Product Comparison</div>', unsafe_allow_html=True)
    st.caption("Tap '⚖️ Compare' on up to 4 products (from Chat, Browse, Cart, or Wishlist) to add them here.")

    if not st.session_state.compare_ids:
        st.info("No products selected for comparison yet.")
    else:
        pool = {}
        for msg in st.session_state.messages:
            for p in msg.get("products", []):
                pool[str(p["product_id"])] = p
        for p in store.get_cart(user_id):
            pool[str(p["product_id"])] = p
        for p in store.get_wishlist(user_id):
            pool[str(p["product_id"])] = p
        for p in get_all_products_df().to_dict("records"):
            pool[str(p["product_id"])] = p

        compare_products = [pool[pid] for pid in st.session_state.compare_ids if pid in pool]

        if compare_products:
            rows = ["Price (₹)", "Color", "Brand", "Material", "Category", "Stock"]
            table = {"Attribute": rows}
            for p in compare_products:
                table[p.get("name", "")] = [
                    p.get("price", ""), p.get("color", ""), p.get("brand", ""),
                    p.get("material", ""), p.get("category", ""), p.get("stock", ""),
                ]
            st.dataframe(pd.DataFrame(table).set_index("Attribute"), use_container_width=True)

            cols = st.columns(len(compare_products))
            for i, p in enumerate(compare_products):
                with cols[i]:
                    if p.get("image_url"):
                        st.image(p["image_url"], use_container_width=True)
                    st.caption(p.get("name", ""))

        if st.button("Clear Comparison"):
            st.session_state.compare_ids = []
            st.rerun()

# ══════════════════════ SIZE FINDER ══════════════════════
if page == "size":
    st.markdown('<div class="section-title">📏 Size & Fit Recommender</div>', unsafe_allow_html=True)
    unit = st.radio("Height unit", ["cm", "ft / in"], horizontal=True)
    if unit == "cm":
        height_cm = st.number_input("Height (cm)", min_value=120, max_value=220, value=165)
    else:
        c1, c2 = st.columns(2)
        with c1:
            feet = st.number_input("Feet", min_value=3, max_value=7, value=5)
        with c2:
            inches = st.number_input("Inches", min_value=0, max_value=11, value=4)
        height_cm = height_from_feet_inches(feet, inches)

    build = st.selectbox("Build", ["Slim", "Medium", "Athletic", "Heavy / Broad"])
    category = st.selectbox("Category", ["Top", "Jacket", "Dress", "Shirt", "Hoodie", "Kurta", "Blazer"])

    if st.button("Get My Size", type="primary"):
        result = recommend_size(height_cm, build, category)
        html_block(f"""<div class="filter-card"><h3 style="margin-top:0">Recommended Size: {result['size']}</h3><p>{result['note']}</p></div>""")

# ══════════════════════ OUTFIT BUILDER ══════════════════════
if page == "outfit":
    st.markdown('<div class="section-title">👗 Outfit Builder</div>', unsafe_allow_html=True)
    st.caption("Upload a photo of something you already own, and ask what would pair well with it.")

    outfit_img = st.file_uploader("Upload your outfit photo", type=["jpg", "jpeg", "png", "webp"], key="outfit_uploader")
    outfit_query = st.text_input("What are you looking for?", placeholder="e.g. what top goes with these jeans?")
    if outfit_img:
        st.image(outfit_img, caption="Your photo", width=200)

    if st.button("Find Matching Items", type="primary"):
        if not outfit_img or not outfit_query.strip():
            st.warning("Please upload a photo and describe what you're looking for.")
        else:
            query_image = Image.open(outfit_img).convert("RGB")
            enhanced_query = f"Suggest complementary clothing items that would pair well with this outfit. {outfit_query}"
            with st.spinner("🔍 Finding matching pieces..."):
                result = pipeline.chat(query=enhanced_query, image=query_image)
            st.markdown(f'<div class="msg-bot">{result["answer"]}</div>', unsafe_allow_html=True)
            render_product_grid(result["products"], key_prefix="outfit")

# ══════════════════════ ANALYTICS ══════════════════════
if page == "analytics":
    st.markdown('<div class="section-title">📊 Catalog Analytics</div>', unsafe_allow_html=True)
    all_df = get_all_products_df()

    if all_df.empty:
        st.info("Catalog not loaded yet.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Products", len(all_df))
        m2.metric("Categories", all_df["category"].nunique())
        m3.metric("Brands", all_df["brand"].nunique())
        m4.metric("Avg Price", f"₹{all_df['price'].mean():,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            cat_counts = all_df["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.bar(cat_counts.head(15), x="category", y="count",
                         color_discrete_sequence=["#0A1F44"], title="Products by Category")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(all_df, x="price", nbins=20, color_discrete_sequence=["#D4AF37"],
                                title="Price Distribution")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">🧪 RAG Retrieval Evaluation</div>', unsafe_allow_html=True)
        st.caption("Runs a fixed set of test queries against the retriever and scores retrieval quality — "
                   "useful for demonstrating the RAG pipeline's precision in your project report.")

        if st.button("▶️ Run Evaluation Now", type="primary"):
            with st.spinner("Running evaluation..."):
                results = run_evaluation(pipeline, k=5)
            e1, e2, e3 = st.columns(3)
            e1.metric("Mean Precision@5", results["mean_precision_at_k"])
            e2.metric("Category Hit Rate", results["category_hit_rate"])
            e3.metric("Mean Similarity", f"{results['mean_similarity']}%")
            st.dataframe(pd.DataFrame(results["per_query"]), use_container_width=True)

# ══════════════════════ TRENDS & INSIGHTS ══════════════════════
if page == "trends":
    st.markdown('<div class="section-title">📈 Trends & Insights</div>', unsafe_allow_html=True)
    all_df = get_all_products_df()

    st.markdown("#### 🔥 Trending Now")
    st.caption("Based on real view activity across all shoppers on this platform.")
    trending = store.get_most_viewed_products(limit=6)
    if not trending:
        st.info("Not enough browsing activity yet — view a few products to start building trends.")
    else:
        cols = st.columns(min(len(trending), 3))
        for i, p in enumerate(trending):
            with cols[i % 3]:
                html_block(f"""
                <div class="product-card">
                    <div class="product-name">{p.get('name', '')}</div>
                    <div class="product-meta">👁️ {p.get('view_count', 0)} views this period</div>
                    <div class="live-price">₹{p.get('price', 0):,}</div>
                </div>
                """)
                if p.get("image_url"):
                    st.image(p["image_url"], use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💹 Price Trend & Forecast by Category")
    st.caption("⚠️ Illustrative simulation anchored to today's real average catalog price — "
               "there's no historical price record for this dataset, so history/forecast lines "
               "are a seeded random-walk model, not real past market data.")

    if not all_df.empty:
        categories = sorted(all_df["category"].dropna().unique().tolist())
        selected_cat = st.selectbox("Category", categories)
        cat_avg_price = all_df[all_df["category"] == selected_cat]["price"].mean()
        trend_df = simulate_price_trend(selected_cat, cat_avg_price)

        fig = px.line(trend_df, x="date", y="price", color="type",
                      color_discrete_map={"Historical (simulated)": "#0A1F44",
                                          "Forecast (simulated)": "#D4AF37"},
                      title=f"{selected_cat} — Price Trend & 4-Week Forecast")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ✨ Recommended for You")
    st.caption("Based on your recently viewed items.")
    recently_viewed = store.get_recently_viewed(user_id, limit=3)
    if not recently_viewed:
        st.info("Browse a few products and personalized suggestions will appear here.")
    else:
        suggestions = []
        seen_ids = {str(p["product_id"]) for p in recently_viewed}
        for p in recently_viewed:
            for rel in get_related_products(pipeline.database, p, k=3):
                if str(rel.get("product_id")) not in seen_ids:
                    suggestions.append(rel)
                    seen_ids.add(str(rel.get("product_id")))
        if suggestions:
            cols = st.columns(min(len(suggestions[:6]), 3))
            for i, p in enumerate(suggestions[:6]):
                with cols[i % 3]:
                    html_block(f"""
                    <div class="product-card">
                        <div class="product-name">{p.get('name', '')}</div>
                        <div class="live-price">₹{p.get('price', 0):,}</div>
                    </div>
                    """)
                    if p.get("image_url"):
                        st.image(p["image_url"], use_container_width=True)
        else:
            st.info("No new suggestions yet — try viewing a few more products.")

# ══════════════════════ ADMIN DASHBOARD ══════════════════════
if page == "admin" and user.get("is_admin"):
    st.markdown('<div class="section-title">🛠️ Admin Dashboard</div>', unsafe_allow_html=True)

    summary = store.admin_summary()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Users", summary["total_users"])
    m2.metric("Total Orders", summary["total_orders"])
    m3.metric("Platform Revenue", f"₹{summary['total_revenue']:,.0f}")
    m4.metric("Reviews Written", summary["total_reviews"])
    m5.metric("Active Alerts", summary["active_alerts"])

    st.markdown("---")
    all_orders = store.admin_all_orders()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📦 Order Status Breakdown")
        if all_orders:
            status_counts = {}
            for o in all_orders:
                t = get_tracking_info(o["placed_at"])
                status_counts[t["current_label"]] = status_counts.get(t["current_label"], 0) + 1
            status_df = pd.DataFrame(
                {"status": list(status_counts.keys()), "count": list(status_counts.values())}
            )
            fig = px.pie(status_df, names="status", values="count",
                        color_discrete_sequence=["#0A1F44", "#1B3A6B", "#D4AF37", "#A9791F", "#1B8A5A"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No orders placed yet.")
    with c2:
        st.markdown("#### 📈 Signups Over Time")
        signups = store.admin_signup_counts_by_day()
        if signups:
            signup_df = pd.DataFrame(signups)
            fig = px.bar(signup_df, x="day", y="count", color_discrete_sequence=["#D4AF37"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No signups recorded yet.")

    st.markdown("---")
    st.markdown("#### 🏆 Best-Selling Products")
    best_sellers = store.get_best_selling_products(limit=8)
    if best_sellers:
        bs_df = pd.DataFrame([
            {"Product": b["product"].get("name", ""), "Units Sold": b["count"],
             "Price": f"₹{b['product'].get('price', 0):,}"}
            for b in best_sellers
        ])
        st.dataframe(bs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No sales data yet.")

    st.markdown("---")
    st.markdown("#### 🧾 All Orders")
    if all_orders:
        orders_df = pd.DataFrame([
            {"Order ID": o["id"], "User": o["username"],
             "Placed": datetime.fromisoformat(o["placed_at"]).strftime("%d %b %Y, %I:%M %p"),
             "Items": len(o["items"]), "Total": f"₹{o['total']:,.0f}",
             "Payment": o.get("payment_method") or "—",
             "Status": get_tracking_info(o["placed_at"])["current_label"]}
            for o in all_orders
        ])
        st.dataframe(orders_df, use_container_width=True, hide_index=True)
    else:
        st.info("No orders yet.")

    st.markdown("---")
    st.markdown("#### 🎟️ Active Coupon Codes")
    coupon_df = pd.DataFrame(
        [{"Code": code, "Discount": f"{pct}% off"} for code, pct in COUPONS.items()]
    )
    st.dataframe(coupon_df, use_container_width=True, hide_index=True)

elif page == "admin":
    st.warning("Admin access only. Log in with the 'admin' account to view this page.")

# ══════════════════════ MY ACCOUNT ══════════════════════
if page == "account":
    st.markdown('<div class="section-title">👤 My Account</div>', unsafe_allow_html=True)

    initial = user["username"][0].upper() if user["username"] else "U"
    member_since = user.get("created_at", "")[:10] if user.get("created_at") else "—"

    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        html_block(f"""
        <div style="width:72px;height:72px;border-radius:50%;
                    background:linear-gradient(135deg,#0A1F44,#1B3A6B);
                    color:#F5E7B8;display:flex;align-items:center;justify-content:center;
                    font-size:1.8rem;font-weight:700;border:2px solid #D4AF37;">
            {initial}
        </div>
        """)
    with header_col2:
        st.markdown(f"### {user['username']}")
        st.caption(f"Member since {member_since}")

    st.markdown("---")

    stats = store.get_account_stats(user_id)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Orders", stats["orders_count"])
    s2.metric("Total Spent", f"₹{stats['total_spent']:,.0f}")
    s3.metric("Cart Items", stats["cart_count"])
    s4.metric("Wishlist", stats["wishlist_count"])
    s5.metric("Active Alerts", stats["active_alerts"])
    s6.metric("Reviews Written", stats["reviews_written"])

    st.markdown("---")
    st.markdown('<div class="section-title">🔒 Change Password</div>', unsafe_allow_html=True)
    with st.form("change_password_form"):
        old_pw = st.text_input("Current password", type="password")
        new_pw = st.text_input("New password", type="password")
        new_pw2 = st.text_input("Confirm new password", type="password")
        pw_submitted = st.form_submit_button("Update Password", type="primary")
    if pw_submitted:
        if new_pw != new_pw2:
            st.error("New passwords don't match.")
        elif len(new_pw) < 6:
            st.error("New password must be at least 6 characters.")
        else:
            ok, msg = store.change_password(user_id, old_pw, new_pw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
