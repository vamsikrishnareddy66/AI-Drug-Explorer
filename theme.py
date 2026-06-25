"""
theme.py
--------
All CSS injection lives here. app.py calls inject_theme() once at startup;
no other module should st.markdown raw <style> blocks.
"""

import streamlit as st

from config import TEAL_DARK, TEAL_MID, TEAL_LIGHT, GOLD, GREEN_OK, PINK


def inject_theme() -> None:
    """Inject the global dark-glass / neon theme used across all tabs."""
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: white; }}
.stApp {{
    background: linear-gradient(135deg,{TEAL_DARK} 0%,#142850 30%,{TEAL_MID} 65%,#6A11CB 100%);
    color: white;
}}
.stat-card {{
    background: rgba(20,27,58,.88); border: 1px solid rgba(0,212,255,.35);
    border-radius: 16px; padding: 22px 18px; text-align: center;
    backdrop-filter: blur(12px); transition: transform .2s, box-shadow .2s;
}}
.stat-card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 28px rgba(0,212,255,.25); }}
.stat-card .icon {{ font-size: 2rem; margin-bottom: 6px; }}
.stat-card .val  {{ font-size: 1.9rem; font-weight: 800; color: {TEAL_LIGHT}; }}
.stat-card .lbl  {{ color: #DCE9FF; font-size: .82rem; margin-top: 4px; }}
.metric-card {{
    background: rgba(20,27,58,.88); border: 1px solid rgba(0,212,255,.35);
    border-radius: 14px; padding: 18px; text-align: center; backdrop-filter: blur(12px);
}}
.metric-card .val {{ font-size: 1.8rem; font-weight: 800; color: {TEAL_LIGHT}; }}
.metric-card .lbl {{ color: #DCE9FF; font-size: .82rem; }}
.section-header {{
    border-left: 5px solid {TEAL_LIGHT}; padding-left: 12px; color: white;
    font-size: 1.2rem; font-weight: 700; margin: 25px 0 15px;
}}
.info-box {{
    background: rgba(0,212,255,.08); border: 1px solid rgba(0,212,255,.35);
    border-radius: 12px; padding: 16px; color: #EAF7FF;
}}
.disclaimer {{
    background: rgba(255,209,102,.08); border: 1px solid {GOLD};
    border-radius: 12px; padding: 14px; color: #FFE7A0;
}}
.ref-lig-card {{
    background: rgba(20,27,58,.9); border: 1px solid rgba(0,212,255,.25);
    border-radius: 14px; padding: 18px; color: white;
}}
.compound-card {{
    background: rgba(20,27,58,.88); border: 1px solid rgba(123,97,255,.35);
    border-radius: 14px; padding: 18px; margin-bottom: 12px; color: white;
    backdrop-filter: blur(12px);
}}
.compound-card b {{ color: {TEAL_LIGHT}; }}
.roadmap-item {{
    background: rgba(20,27,58,.85); border-left: 4px solid {TEAL_LIGHT};
    border-radius: 10px; padding: 12px 16px; margin: 10px 0; color: white;
}}
.ai-bubble {{
    background: rgba(106,17,203,.25); border: 1px solid rgba(0,212,255,.3);
    border-radius: 14px 14px 14px 4px; padding: 14px 18px; color: white; margin-bottom: 10px;
}}
.user-bubble {{
    background: rgba(0,229,255,.12); border: 1px solid rgba(0,229,255,.3);
    border-radius: 14px 14px 4px 14px; padding: 14px 18px; color: white;
    margin-bottom: 10px; text-align: right;
}}
.progress-step {{
    background: rgba(20,27,58,.85); border: 1px solid rgba(0,212,255,.2);
    border-radius: 10px; padding: 10px 16px; margin: 6px 0; color: #DCE9FF; font-size: .9rem;
}}
.progress-step.done   {{ border-color: {GREEN_OK}; color: {GREEN_OK}; }}
.progress-step.active {{ border-color: {TEAL_LIGHT}; color: white; font-weight: 700; }}
.admet-pass {{ color: {GREEN_OK}; font-weight: 700; }}
.admet-warn {{ color: {GOLD}; font-weight: 700; }}
.admet-fail {{ color: {PINK}; font-weight: 700; }}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,{TEAL_DARK},{TEAL_MID} 120%);
}}
section[data-testid="stSidebar"] * {{ color: white !important; }}
section[data-testid="stSidebar"] .stTextInput input {{ color: black !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(20,27,58,.8); color: white;
    border-radius: 12px 12px 0 0; font-weight: 700; padding: 10px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg,#6A11CB,#00D4FF) !important; color: white !important;
}}
.stButton>button {{
    border-radius: 12px; border: none;
    background: linear-gradient(90deg,#6A11CB,#00D4FF);
    color: white; font-weight: 700; transition: .3s;
}}
.stButton>button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,212,255,.4); }}
.site-footer {{
    background: rgba(7,26,47,.95); border-top: 1px solid rgba(0,212,255,.25);
    border-radius: 16px; padding: 28px 32px; margin-top: 40px;
    text-align: center; color: #9DB4CC; font-size: .85rem;
}}
.site-footer b {{ color: {TEAL_LIGHT}; }}
.site-footer a  {{ color: {GOLD}; text-decoration: none; }}
.stTextInput input, .stSelectbox div, .stNumberInput input {{ border-radius: 10px !important; }}
.selectivity-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: .78rem; font-weight: 700; margin: 2px;
}}
</style>
""", unsafe_allow_html=True)
