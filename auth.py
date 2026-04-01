import json
import os
import hashlib
import base64
import streamlit as st
from datetime import datetime

USERS_FILE = "users.json"
LOGO_PATH  = "logo.png"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def get_logo_base64() -> str:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def register_user(username: str, password: str, email: str) -> tuple:
    if not username or not password or not email:
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if "@" not in email:
        return False, "Invalid email address."
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "watchlist": ["AAPL", "MSFT", "TSLA"],
    }
    save_users(users)
    return True, "Account created successfully!"


def login_user(username: str, password: str) -> tuple:
    if not username or not password:
        return False, "Please enter username and password."
    users = load_users()
    if username not in users:
        return False, "Username not found."
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, users[username]


def get_user_watchlist(username: str) -> list:
    users = load_users()
    if username in users:
        return users[username].get("watchlist", ["AAPL", "MSFT", "TSLA"])
    return ["AAPL", "MSFT", "TSLA"]


def update_watchlist(username: str, watchlist: list):
    users = load_users()
    if username in users:
        users[username]["watchlist"] = watchlist
        save_users(users)


def logout():
    for key in ["logged_in", "username", "user_data"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def show_login_page():
    logo_b64 = get_logo_base64()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:220px;margin:0 auto 8px;display:block;" />' if logo_b64 else ""

    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
      html, body, [class*="css"] {{ font-family: 'JetBrains Mono', monospace !important; }}
      .stApp {{ background: #0d0f14; }}
      .block-container {{ padding: 2rem; }}
      .login-wrap {{
        max-width: 420px; margin: 30px auto;
        background: #1a1f2e;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px; padding: 36px 32px;
      }}
      .login-sub {{
        text-align: center; color: #8b93a8;
        font-size: 12px; margin-bottom: 28px; line-height: 1.6;
      }}
      .stTextInput > div > div > input {{
        background: #0d0f14 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #e8eaf0 !important;
        font-family: 'JetBrains Mono', monospace !important;
      }}
      .stTextInput > div > div > input:focus {{
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
      }}
      .stButton > button {{
        background: #00d4ff !important; color: #000 !important;
        font-weight: 700 !important; border: none !important;
        border-radius: 8px !important; width: 100% !important;
        padding: 12px !important; font-size: 14px !important;
        font-family: 'JetBrains Mono', monospace !important;
        margin-top: 8px;
      }}
      .stButton > button:hover {{ background: #00b8e6 !important; }}
      .stTabs [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.07) !important;
        gap: 0;
      }}
      .stTabs [data-baseweb="tab"] {{
        color: #8b93a8 !important; font-size: 13px !important;
        padding: 10px 24px !important; background: transparent !important;
        font-family: 'JetBrains Mono', monospace !important;
      }}
      .stTabs [aria-selected="true"] {{
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
      }}
      .demo-hint {{
        text-align: center; color: #4a5168;
        font-size: 11px; margin-top: 16px;
      }}
      .divider {{
        border: none; border-top: 1px solid rgba(255,255,255,0.07);
        margin: 20px 0;
      }}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Logo
        if logo_html:
            st.markdown(logo_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;font-size:28px;font-weight:700;color:#00d4ff;margin-bottom:8px;">↗ PredictTrade</div>', unsafe_allow_html=True)

        st.markdown('<div class="login-sub">AI-powered stock market predictions<br>and real-time trading intelligence</div>', unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐  Login", "📝  Register"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("Password", placeholder="Enter your password", type="password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Login →", key="login_btn"):
                if username and password:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state.logged_in  = True
                        st.session_state.username   = username
                        st.session_state.user_data  = result
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.warning("⚠ Please fill in all fields.")

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="demo-hint">Demo account → username: <b>demo</b> &nbsp;|&nbsp; password: <b>demo123</b></div>', unsafe_allow_html=True)

        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            reg_user    = st.text_input("Username", placeholder="Choose a username", key="reg_user")
            reg_email   = st.text_input("Email", placeholder="Enter your email", key="reg_email")
            reg_pass    = st.text_input("Password", placeholder="Min 6 characters", type="password", key="reg_pass")
            reg_confirm = st.text_input("Confirm Password", placeholder="Repeat password", type="password", key="reg_confirm")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account →", key="reg_btn"):
                if reg_pass != reg_confirm:
                    st.error("❌ Passwords do not match.")
                else:
                    success, msg = register_user(reg_user, reg_pass, reg_email)
                    if success:
                        st.success(f"✅ {msg} Please login.")
                    else:
                        st.error(f"❌ {msg}")


def init_demo_user():
    users = load_users()
    if "demo" not in users:
        users["demo"] = {
            "password": hash_password("demo123"),
            "email": "demo@predicttrade.com",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "watchlist": ["AAPL", "MSFT", "TSLA", "GOOGL"],
        }
        save_users(users)


def check_auth() -> bool:
    init_demo_user()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in