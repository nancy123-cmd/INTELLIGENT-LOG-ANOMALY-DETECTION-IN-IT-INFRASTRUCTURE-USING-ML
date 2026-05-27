# -*- coding: utf-8 -*-
"""
IT Infrastructure Log Anomaly Detection & Root Cause Analysis System
Run:
    python generate_dataset.py      ← create dataset first
    streamlit run it_log_monitor.py ← launch dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import time
import datetime
import os
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Log Anomaly Detection",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #060d18;
    color: #cdd9e5;
}
section[data-testid="stSidebar"] {
    background: #080f1c;
    border-right: 1px solid #112240;
}
h1 {
    font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 2.1rem;
    background: linear-gradient(90deg, #00c6ff, #00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
h2 { font-family:'Rajdhani',sans-serif; font-weight:700; color:#00c6ff; }
h3 { font-family:'Rajdhani',sans-serif; font-weight:600; color:#00ff88; }

div[data-testid="metric-container"] {
    background:#0a1628; border:1px solid #1a3a5c; border-radius:10px;
    padding:14px 18px; box-shadow:0 0 15px rgba(0,198,255,0.08);
}
div[data-testid="metric-container"] label {
    color:#00c6ff !important; font-size:0.72rem;
    letter-spacing:2px; text-transform:uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family:'Share Tech Mono',monospace; font-size:1.9rem; color:#e6f1ff;
}
div.stButton > button {
    background: linear-gradient(135deg,#005f8e,#007a4d);
    color:white; border:1px solid #00c6ff44; border-radius:8px;
    padding:10px 26px; font-family:'Rajdhani',sans-serif;
    font-weight:700; font-size:1rem; transition:all 0.2s;
}
div.stButton > button:hover {
    background:linear-gradient(135deg,#0077b6,#00b36b);
    box-shadow:0 0 20px rgba(0,198,255,0.4); transform:translateY(-2px);
}
.info-box {
    background:#0a1628; border-left:4px solid #00c6ff;
    border-radius:8px; padding:14px 18px; margin:10px 0;
    font-size:0.95rem; line-height:1.7;
}
.tamil-box {
    background:#091520; border-left:4px solid #00ff88;
    border-radius:8px; padding:12px 16px; margin:8px 0;
    font-size:0.9rem; color:#a8d8b0;
}
.step-box {
    background:#0a1a2e; border:1px solid #1a3a5c;
    border-radius:10px; padding:16px 20px; margin:8px 0;
}
.log-line { font-family:'Share Tech Mono',monospace; font-size:0.77rem;
            padding:3px 8px; border-radius:4px; margin:2px 0; }
.log-error { background:#1a0808; color:#ff6b6b; }
.log-warn  { background:#1a1208; color:#ffd166; }
.log-info  { background:#081a10; color:#06d6a0; }
.log-debug { background:#080d1a; color:#74b9ff; }
.anomaly-badge {
    background:#2d0a0a; border:1px solid #ff4444; color:#ff6b6b;
    border-radius:6px; padding:2px 8px;
    font-family:'Share Tech Mono',monospace; font-size:0.73rem;
}
.normal-badge {
    background:#0a2d1a; border:1px solid #00ff88; color:#00ff88;
    border-radius:6px; padding:2px 8px;
    font-family:'Share Tech Mono',monospace; font-size:0.73rem;
}
.section-hr { border:none; border-top:1px solid #112240; margin:20px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (fallback / used by generate_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────
ROOT_CAUSES = ["Disk Failure","Memory Overflow","Network Issue",
               "Security Attack","Application Crash","Normal"]

LOG_TEMPLATES = {
    "Disk Failure":      ["Disk I/O error on /dev/sda1","SMART error disk health critical",
                          "Write failed no space left on device","Bad sector detected at block 4096",
                          "Filesystem corruption detected","Disk timeout after 30 seconds"],
    "Memory Overflow":   ["OutOfMemoryError Java heap space","Memory usage exceeded 95 percent",
                          "Kernel OOM killer activated","Swap usage at 100 percent",
                          "Process killed memory limit reached","GC overhead limit exceeded"],
    "Network Issue":     ["Connection timeout to 192.168.1.1","Packet loss 40 percent on eth0",
                          "DNS resolution failed for db.internal","TCP retransmit storm detected",
                          "Network interface eth1 link down","High latency 2500ms detected"],
    "Security Attack":   ["Failed login attempt 50 tries in 10 seconds","Unauthorized access detected",
                          "SQL injection pattern detected","Port scan detected from remote host",
                          "Brute force attack on SSH port 22","DDoS attack high traffic volume"],
    "Application Crash": ["NullPointerException in UserService","Segmentation fault core dumped",
                          "Unhandled exception IndexOutOfBounds","Service health check failed 3 times",
                          "Thread pool exhausted all workers busy","Database connection pool exhausted"],
    "Normal":            ["User login successful","Backup completed successfully",
                          "Health check OK all services running","Database query executed in 12ms",
                          "Cache refresh completed","Scheduled job completed",
                          "API request processed 200 OK","Config reload successful",
                          "Memory usage normal 45 percent","Network throughput normal"],
}

SERVERS    = ["server-01","server-02","db-primary","db-replica",
              "web-01","web-02","app-server","cache-01"]
COMPONENTS = ["kernel","sshd","mysqld","nginx","java-app",
              "cron","systemd","network-manager"]

def generate_row(idx, ts, cause=None):
    if cause is None:
        cause = random.choices(ROOT_CAUSES, weights=[3,4,4,3,3,83])[0]
    msg    = random.choice(LOG_TEMPLATES[cause])
    level  = "ERROR" if cause != "Normal" else random.choice(["INFO","INFO","INFO","WARN","DEBUG"])
    server = random.choice(SERVERS)
    comp   = random.choice(COMPONENTS)
    pid    = random.randint(1000, 65535)
    block  = f"blk_{random.randint(100000000,999999999)}"
    return {
        "LogId": f"LOG{idx:07d}", "BlockId": block,
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Level": level, "Server": server, "Component": comp,
        "PID": pid, "Message": msg,
        "FullLog": f"{ts.strftime('%Y-%m-%d %H:%M:%S')} {level} [{server}] {comp}[{pid}]: {msg}",
        "RootCause": cause, "Label": 0 if cause=="Normal" else 1,
    }

def generate_dataset(n=10000):
    rows, base = [], datetime.datetime(2024,1,1)
    for i in range(n):
        ts = base + datetime.timedelta(seconds=i*5+random.randint(0,4))
        rows.append(generate_row(i+1, ts))
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean_log(text):
    text = str(text).lower()
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}','IP_ADDR',text)
    text = re.sub(r'\d+','NUM',text)
    text = re.sub(r'[^a-zA-Z_ ]',' ',text)
    return re.sub(r'\s+',' ',text).strip()

def dark_fig(figsize=(8,4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#060d18")
    ax.set_facecolor("#060d18")
    return fig, ax

def smetric(label, value, color="#00c6ff"):
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1a3a5c;border-radius:10px;
                padding:14px 18px;text-align:center;margin:4px 0;">
        <div style="color:{color};font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;">
            {label}</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:1.9rem;color:#e6f1ff;">
            {value}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🖥️ IT Log Monitor")
    st.markdown("**Anomaly Detection System**")
    st.markdown("---")
    st.markdown("### Model Settings")
    contamination = st.slider("IF Contamination",  0.05, 0.30, 0.17, 0.01)
    n_estimators  = st.slider("RF Trees",          50,   300,  150,  25)
    test_size     = st.slider("Test Split",        0.1,  0.4,  0.2,  0.05)
    max_features  = st.selectbox("TF-IDF Max Features", [1000,3000,5000,10000], index=1)
    st.markdown("---")
    st.markdown("### Live Simulation")
    sim_speed = st.slider("Lines per second", 1, 15, 4)
    st.markdown("---")
    st.caption("Python · Streamlit · scikit-learn")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🖥️ IT Infrastructure Log Anomaly Detection System")
st.markdown("> Real-time monitoring · Root cause analysis · Isolation Forest + Random Forest")
st.markdown('<hr class="section-hr">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_ingest, tab_dash, tab_train, tab_eval, tab_live, tab_predict = st.tabs([
    "📂 Data Ingestion",
    "📊 Dashboard",
    "🤖 Train Models",
    "📈 Evaluation",
    "🔴 Live Monitor",
    "🔍 Predict",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("Data Ingestion")

    st.markdown("""
    <div class="info-box">
        <b>Step 1:</b> Generate or upload your log dataset here before using other tabs.<br>
        The dataset is the foundation for training, evaluation, and prediction.
    </div>
    <div class="tamil-box">
        💬 <b>Data Ingestion என்னன்னா?</b> — முதல்ல data load பண்ணணும்.
        "Generate Dataset" button click பண்ணா 20,000 synthetic IT logs உருவாகும்.
        அல்லது உங்க own CSV file upload பண்ணலாம். Data ready-ஆனா மட்டும்
        மத்த tabs work பண்ணும்.
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-hr">', unsafe_allow_html=True)

    # ── Option A: Generate ────────────────────────────────────────────────
    st.subheader("Option A — Generate Synthetic Dataset")
    st.markdown("""
    <div class="step-box">
        Generates realistic IT infrastructure logs with 6 root cause categories:<br>
        <b>Disk Failure · Memory Overflow · Network Issue ·
        Security Attack · Application Crash · Normal</b>
    </div>""", unsafe_allow_html=True)

    col_n, col_btn = st.columns([2,1])
    with col_n:
        gen_n = st.slider("Number of log lines to generate",
                          1000, 50000, 20000, step=1000)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_gen = st.button("⚡ Generate Dataset", key="btn_gen")

    if btn_gen:
        with st.spinner(f"Generating {gen_n:,} synthetic IT infrastructure logs..."):
            df_gen = generate_dataset(gen_n)
            st.session_state["df"] = df_gen
            # Auto-save CSV
            df_gen.to_csv("logs_dataset.csv", index=False)
            lbl = df_gen[["BlockId","RootCause","Label"]].copy()
            lbl["Label"] = lbl["Label"].map({0:"Normal",1:"Anomaly"})
            lbl.to_csv("anomaly_label.csv", index=False)

        st.success(f"✅ Dataset generated and saved as **logs_dataset.csv** ({gen_n:,} rows)")

        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: smetric("Total Rows",  f"{len(df_gen):,}",                          "#00c6ff")
        with c2: smetric("Normal",      f"{(df_gen['Label']==0).sum():,}",            "#00ff88")
        with c3: smetric("Anomaly",     f"{(df_gen['Label']==1).sum():,}",            "#ff6b6b")
        with c4: smetric("Servers",     str(df_gen['Server'].nunique()),              "#ffd166")
        with c5: smetric("Root Causes", str(df_gen['RootCause'].nunique()),           "#a29bfe")

        st.markdown("<br>", unsafe_allow_html=True)

        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.subheader("Root Cause Distribution")
            rc = df_gen["RootCause"].value_counts()
            rc_clr = {"Normal":"#06d6a0","Disk Failure":"#ff6b6b",
                      "Memory Overflow":"#ff9f43","Network Issue":"#74b9ff",
                      "Security Attack":"#ee5a24","Application Crash":"#a29bfe"}
            fig, ax = dark_fig((6,4))
            bars = ax.barh(rc.index, rc.values,
                           color=[rc_clr.get(r,"#888") for r in rc.index],
                           edgecolor="#060d18", height=0.6)
            for bar, val in zip(bars, rc.values):
                ax.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                        f"{val:,}", va="center", color="#cdd9e5", fontsize=9)
            ax.set_xlabel("Count", color="#94a3b8")
            ax.tick_params(colors="#94a3b8", labelsize=9)
            for sp in ax.spines.values(): sp.set_edgecolor("#112240")
            ax.set_title("Root Cause Breakdown", color="#00c6ff", fontweight="bold")
            st.pyplot(fig, use_container_width=True)

        with col_dist2:
            st.subheader("Log Level Distribution")
            lv = df_gen["Level"].value_counts()
            lv_clr = {"ERROR":"#ff6b6b","WARN":"#ffd166","INFO":"#06d6a0","DEBUG":"#74b9ff"}
            fig, ax = dark_fig((6,4))
            wedges,_,autotexts = ax.pie(
                lv.values, labels=lv.index,
                colors=[lv_clr.get(l,"#888") for l in lv.index],
                autopct='%1.1f%%', startangle=90,
                textprops={"color":"#cdd9e5","fontsize":11},
                wedgeprops={"edgecolor":"#060d18","linewidth":2})
            for at in autotexts: at.set_color("#060d18"); at.set_fontweight("bold")
            ax.set_title("Log Levels", color="#00c6ff", fontweight="bold")
            st.pyplot(fig, use_container_width=True)

        with st.expander("Preview generated data (first 100 rows)"):
            st.dataframe(df_gen[["LogId","Timestamp","Level","Server",
                                  "Component","Message","RootCause","Label"]].head(100),
                         use_container_width=True)

        # Download buttons
        st.subheader("Download Files")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button(
                label="⬇️ Download logs_dataset.csv",
                data=df_gen.to_csv(index=False).encode("utf-8"),
                file_name="logs_dataset.csv",
                mime="text/csv")
        with dc2:
            lbl_dl = df_gen[["BlockId","RootCause","Label"]].copy()
            lbl_dl["Label"] = lbl_dl["Label"].map({0:"Normal",1:"Anomaly"})
            st.download_button(
                label="⬇️ Download anomaly_label.csv",
                data=lbl_dl.to_csv(index=False).encode("utf-8"),
                file_name="anomaly_label.csv",
                mime="text/csv")

    st.markdown('<hr class="section-hr">', unsafe_allow_html=True)

    # ── Option B: Upload ──────────────────────────────────────────────────
    st.subheader("Option B — Upload Existing Dataset")
    st.markdown("""
    <div class="step-box">
        Upload your own <b>logs_dataset.csv</b> (generated previously) or any CSV with
        <b>Message</b>, <b>Label</b> (0/1), and <b>RootCause</b> columns.
    </div>""", unsafe_allow_html=True)

    up_csv = st.file_uploader("Upload logs_dataset.csv", type=["csv"], key="up_main")
    if up_csv:
        with st.spinner("Loading uploaded dataset..."):
            df_up = pd.read_csv(up_csv)
        # Normalise column names
        df_up.columns = [c.strip() for c in df_up.columns]
        if "Message" not in df_up.columns and "message" in df_up.columns:
            df_up = df_up.rename(columns={"message":"Message","label":"Label",
                                           "root_cause":"RootCause","server":"Server",
                                           "level":"Level","timestamp":"Timestamp"})
        if "Label" not in df_up.columns:
            df_up["Label"] = 0
        if "RootCause" not in df_up.columns:
            df_up["RootCause"] = "Unknown"
        if "Server" not in df_up.columns:
            df_up["Server"] = "unknown"
        if "Level" not in df_up.columns:
            df_up["Level"] = "INFO"

        st.session_state["df"] = df_up
        st.success(f"✅ Uploaded dataset loaded: {len(df_up):,} rows, {df_up.columns.tolist()}")
        c1,c2,c3 = st.columns(3)
        with c1: smetric("Total Rows", f"{len(df_up):,}",                  "#00c6ff")
        with c2: smetric("Normal",     f"{(df_up['Label']==0).sum():,}",    "#00ff88")
        with c3: smetric("Anomaly",    f"{(df_up['Label']==1).sum():,}",    "#ff6b6b")
        with st.expander("Preview uploaded data"):
            st.dataframe(df_up.head(50), use_container_width=True)

    # ── Option C: Load saved CSV ──────────────────────────────────────────
    st.markdown('<hr class="section-hr">', unsafe_allow_html=True)
    st.subheader("Option C — Load Saved logs_dataset.csv from Disk")
    if st.button("📁 Load logs_dataset.csv from current folder", key="btn_load_disk"):
        if os.path.exists("logs_dataset.csv"):
            with st.spinner("Loading..."):
                df_disk = pd.read_csv("logs_dataset.csv")
            st.session_state["df"] = df_disk
            st.success(f"✅ Loaded logs_dataset.csv — {len(df_disk):,} rows")
            c1,c2,c3 = st.columns(3)
            with c1: smetric("Total Rows", f"{len(df_disk):,}",                   "#00c6ff")
            with c2: smetric("Normal",     f"{(df_disk['Label']==0).sum():,}",     "#00ff88")
            with c3: smetric("Anomaly",    f"{(df_disk['Label']==1).sum():,}",     "#ff6b6b")
        else:
            st.error("logs_dataset.csv not found. Run 'Generate Dataset' first or upload one.")

    # Status indicator
    st.markdown('<hr class="section-hr">', unsafe_allow_html=True)
    if "df" in st.session_state:
        df_loaded = st.session_state["df"]
        st.success(f"✅ **Data is ready!** {len(df_loaded):,} rows loaded. "
                   f"You can now use all other tabs.")
    else:
        st.warning("⚠️ No data loaded yet. Use one of the options above to load data first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.header("System Overview Dashboard")

    if "df" not in st.session_state:
        st.warning("⚠️ Go to **📂 Data Ingestion** tab and load your dataset first.")
        st.stop()

    df = st.session_state["df"]

    # Normalise column access
    msg_col = "Message"   if "Message"   in df.columns else "message"
    lbl_col = "Label"     if "Label"     in df.columns else "label"
    rc_col  = "RootCause" if "RootCause" in df.columns else "root_cause"
    sv_col  = "Server"    if "Server"    in df.columns else "server"
    lv_col  = "Level"     if "Level"     in df.columns else "level"
    ts_col  = "Timestamp" if "Timestamp" in df.columns else "timestamp"

    total     = len(df)
    n_anomaly = int((df[lbl_col]==1).sum())
    n_normal  = total - n_anomaly
    n_servers = df[sv_col].nunique()
    n_errors  = int((df[lv_col]=="ERROR").sum())

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: smetric("Total Logs",    f"{total:,}",    "#00c6ff")
    with c2: smetric("Anomalies",     f"{n_anomaly:,}","#ff6b6b")
    with c3: smetric("Normal",        f"{n_normal:,}", "#00ff88")
    with c4: smetric("Servers",       str(n_servers),  "#ffd166")
    with c5: smetric("ERROR Lines",   f"{n_errors:,}", "#a29bfe")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Log Level Distribution")
        lv_cnt = df[lv_col].value_counts()
        clr_m  = {"ERROR":"#ff6b6b","WARN":"#ffd166","INFO":"#06d6a0","DEBUG":"#74b9ff"}
        fig, ax = dark_fig((5,3.5))
        _,_,auts = ax.pie(lv_cnt.values, labels=lv_cnt.index,
                           colors=[clr_m.get(l,"#888") for l in lv_cnt.index],
                           autopct='%1.1f%%', startangle=90,
                           textprops={"color":"#cdd9e5","fontsize":10},
                           wedgeprops={"edgecolor":"#060d18","linewidth":2})
        for at in auts: at.set_color("#060d18"); at.set_fontweight("bold")
        ax.set_title("Log Levels", color="#00c6ff", fontweight="bold")
        st.pyplot(fig, use_container_width=True)

    with col_b:
        st.subheader("Root Cause Breakdown")
        rc_cnt = df[rc_col].value_counts()
        rc_clr = {"Normal":"#06d6a0","Disk Failure":"#ff6b6b","Memory Overflow":"#ff9f43",
                  "Network Issue":"#74b9ff","Security Attack":"#ee5a24",
                  "Application Crash":"#a29bfe","Unknown":"#636e72"}
        fig, ax = dark_fig((5,3.5))
        bars = ax.barh(rc_cnt.index, rc_cnt.values,
                       color=[rc_clr.get(r,"#888") for r in rc_cnt.index],
                       edgecolor="#060d18", height=0.6)
        for bar, val in zip(bars, rc_cnt.values):
            ax.text(bar.get_width()+8, bar.get_y()+bar.get_height()/2,
                    f"{val:,}", va="center", color="#cdd9e5", fontsize=9)
        ax.set_xlabel("Count", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#112240")
        ax.set_title("Root Causes", color="#00c6ff", fontweight="bold")
        st.pyplot(fig, use_container_width=True)

    with col_c:
        st.subheader("Anomaly vs Normal")
        vals = [n_normal, n_anomaly]
        fig, ax = dark_fig((5,3.5))
        bars = ax.bar(["Normal","Anomaly"], vals, color=["#00ff88","#ff6b6b"],
                      width=0.45, edgecolor="#060d18", linewidth=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                    f"{v:,}", ha="center", color="#e6f1ff", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_edgecolor("#112240")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_title("Class Balance", color="#00c6ff", fontweight="bold")
        st.pyplot(fig, use_container_width=True)

    # Timeline
    st.subheader("Anomaly Timeline")
    try:
        df["_ts"] = pd.to_datetime(df[ts_col])
        df_t = df.set_index("_ts").resample("30T")[lbl_col].sum().reset_index()
        fig, ax = dark_fig((12,3))
        ax.fill_between(df_t.iloc[:,0], df_t.iloc[:,1], color="#ff6b6b", alpha=0.25)
        ax.plot(df_t.iloc[:,0], df_t.iloc[:,1], color="#ff6b6b", lw=1.8)
        ax.set_ylabel("Anomaly Count", color="#94a3b8")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#112240")
        ax.set_title("Anomalies Over Time (30-min bins)", color="#00c6ff", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
    except Exception:
        st.info("Timestamp column not parseable — timeline skipped.")

    # Server Heatmap
    st.subheader("Server x Log Level Heatmap")
    heat = df.groupby([sv_col, lv_col]).size().unstack(fill_value=0)
    fig, ax = dark_fig((11,3.5))
    sns.heatmap(heat, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, linecolor="#060d18", ax=ax, cbar_kws={"shrink":0.6})
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.set_xlabel("Log Level", color="#94a3b8"); ax.set_ylabel("Server", color="#94a3b8")
    ax.set_title("Error Distribution per Server", color="#00c6ff", fontweight="bold")
    st.pyplot(fig, use_container_width=True)

    with st.expander("View Raw Data (first 200 rows)"):
        st.dataframe(df.head(200), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.header("Train Anomaly Detection Models")

    if "df" not in st.session_state:
        st.warning("⚠️ Go to **📂 Data Ingestion** tab and load your dataset first.")
        st.stop()

    if not ML_AVAILABLE:
        st.error("scikit-learn not installed. Run: pip install scikit-learn")
        st.stop()

    st.markdown("""
    <div class="info-box">
        <b>ML Pipeline:</b><br>
        1️⃣ <b>TF-IDF Vectorization</b> — Log text → numerical feature matrix<br>
        2️⃣ <b>Isolation Forest</b> — Unsupervised anomaly detection<br>
        3️⃣ <b>Random Forest</b> — Root cause classification (6 classes)
    </div>
    <div class="tamil-box">
        💬 Log text TF-IDF மூலம் numbers-ஆ மாறும். Isolation Forest normal behavior கத்துக்கும்.
        Random Forest exact root cause (Disk/Memory/Network/Security/App) predict பண்ணும்.
    </div>""", unsafe_allow_html=True)

    df = st.session_state["df"]
    msg_col = "Message"   if "Message"   in df.columns else "message"
    lbl_col = "Label"     if "Label"     in df.columns else "label"
    rc_col  = "RootCause" if "RootCause" in df.columns else "root_cause"

    col_iso, col_rf = st.columns(2, gap="large")
    with col_iso:
        st.subheader("Isolation Forest")
        st.caption("Unsupervised — learns normal patterns, flags outliers.")
        btn_iso = st.button("Train Isolation Forest", key="train_iso")
    with col_rf:
        st.subheader("Random Forest")
        st.caption("Supervised — classifies root cause from labelled data.")
        btn_rf = st.button("Train Random Forest", key="train_rf")

    def build_features(dataframe):
        dataframe = dataframe.copy()
        dataframe["_clean"] = dataframe[msg_col].apply(clean_log)
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words="english")
        X = tfidf.fit_transform(dataframe["_clean"])
        y = dataframe[lbl_col].fillna(0).astype(int)
        return X, y, tfidf

    if btn_iso:
        with st.spinner("Training Isolation Forest..."):
            X, y, tfidf_iso = build_features(df)
            iso = IsolationForest(n_estimators=100, contamination=contamination,
                                  random_state=42, n_jobs=-1)
            t0 = time.time()
            iso.fit(X)
            elapsed    = time.time() - t0
            iso_preds  = [1 if p==-1 else 0 for p in iso.predict(X)]
            iso_scores = iso.decision_function(X)

        st.session_state.update({"iso_model":iso,"iso_preds":iso_preds,
                                  "iso_y":y,"tfidf_iso":tfidf_iso,"iso_scores":iso_scores})
        n_det = sum(iso_preds)
        c1,c2,c3 = st.columns(3)
        with c1: smetric("Training Time",  f"{elapsed:.1f}s",             "#00c6ff")
        with c2: smetric("Anomalies Found",f"{n_det:,}",                  "#ff6b6b")
        with c3: smetric("Detection Rate", f"{n_det/len(y)*100:.1f}%",    "#ffd166")
        st.success(f"Isolation Forest trained in {elapsed:.2f}s — {n_det:,} anomalies flagged.")

    if btn_rf:
        with st.spinner("Training Random Forest..."):
            df_rc = df.copy()
            df_rc["_clean"] = df_rc[msg_col].apply(clean_log)
            tfidf_rf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words="english")
            X_all    = tfidf_rf.fit_transform(df_rc["_clean"])
            le       = LabelEncoder()
            y_all    = le.fit_transform(df_rc[rc_col].fillna("Normal"))

            X_tr,X_te,y_tr,y_te = train_test_split(
                X_all, y_all, test_size=test_size, random_state=42, stratify=y_all)
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            t0 = time.time()
            rf.fit(X_tr, y_tr)
            elapsed  = time.time() - t0
            rf_preds = rf.predict(X_te)
            rf_proba = rf.predict_proba(X_te)

        st.session_state.update({"rf_model":rf,"rf_preds":rf_preds,"rf_y_test":y_te,
                                  "tfidf_rf":tfidf_rf,"le":le,"rf_proba":rf_proba})
        acc = (rf_preds == y_te).mean()
        c1,c2,c3 = st.columns(3)
        with c1: smetric("Training Time",    f"{elapsed:.1f}s",          "#00c6ff")
        with c2: smetric("Test Accuracy",     f"{acc*100:.1f}%",          "#00ff88")
        with c3: smetric("Root Cause Classes",str(len(le.classes_)),      "#ffd166")
        st.success(f"Random Forest trained in {elapsed:.2f}s — Accuracy: {acc*100:.1f}%")
        with st.expander("Classification Report"):
            rpt = classification_report(y_te,rf_preds,target_names=le.classes_,output_dict=True)
            st.dataframe(pd.DataFrame(rpt).transpose().round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("Model Evaluation")

    has_iso = "iso_preds" in st.session_state
    has_rf  = "rf_preds"  in st.session_state

    if not has_iso and not has_rf:
        st.warning("⚠️ Train at least one model in the **Train Models** tab first.")
        st.stop()

    ev1, ev2 = st.tabs(["Isolation Forest", "Random Forest"])

    with ev1:
        if not has_iso:
            st.info("Train Isolation Forest first.")
        else:
            ip = st.session_state["iso_preds"]
            iy = st.session_state["iso_y"]
            sc = st.session_state["iso_scores"]
            rp = classification_report(iy, ip, output_dict=True)
            cm = confusion_matrix(iy, ip)

            c1,c2,c3,c4 = st.columns(4)
            with c1: smetric("Precision",f"{rp.get('1',{}).get('precision',0):.3f}","#00c6ff")
            with c2: smetric("Recall",   f"{rp.get('1',{}).get('recall',0):.3f}",   "#00ff88")
            with c3: smetric("F1-Score", f"{rp.get('1',{}).get('f1-score',0):.3f}", "#ffd166")
            with c4: smetric("Accuracy", f"{rp.get('accuracy',0):.3f}",             "#a29bfe")

            ca, cb = st.columns(2)
            with ca:
                fig, ax = dark_fig((5,4))
                sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                            xticklabels=["Normal","Anomaly"],
                            yticklabels=["Normal","Anomaly"],
                            linewidths=0.5,linecolor="#060d18",ax=ax,cbar=False)
                ax.tick_params(colors="#94a3b8")
                ax.set_xlabel("Predicted",color="#94a3b8")
                ax.set_ylabel("Actual",color="#94a3b8")
                ax.set_title("Confusion Matrix — IF",color="#00c6ff",fontweight="bold")
                st.pyplot(fig, use_container_width=True)
            with cb:
                ya = np.array(iy)
                fig, ax = dark_fig((5,4))
                ax.hist(sc[ya==0],bins=50,color="#00ff88",alpha=0.6,label="Normal",density=True)
                ax.hist(sc[ya==1],bins=50,color="#ff6b6b",alpha=0.6,label="Anomaly",density=True)
                ax.set_xlabel("Score",color="#94a3b8"); ax.set_ylabel("Density",color="#94a3b8")
                ax.tick_params(colors="#94a3b8")
                for sp in ax.spines.values(): sp.set_edgecolor("#112240")
                ax.legend(facecolor="#0a1628",edgecolor="#1a3a5c",labelcolor="#cdd9e5")
                ax.set_title("Anomaly Score Distribution",color="#00c6ff",fontweight="bold")
                st.pyplot(fig, use_container_width=True)

    with ev2:
        if not has_rf:
            st.info("Train Random Forest first.")
        else:
            rp2  = st.session_state["rf_preds"]
            yt   = st.session_state["rf_y_test"]
            prob = st.session_state["rf_proba"]
            le   = st.session_state["le"]
            rpt  = classification_report(yt,rp2,output_dict=True)
            cm2  = confusion_matrix(yt,rp2)
            acc  = (rp2==yt).mean()

            c1,c2,c3,c4 = st.columns(4)
            with c1: smetric("Accuracy",     f"{acc:.3f}",                            "#00ff88")
            with c2: smetric("Macro F1",     f"{rpt['macro avg']['f1-score']:.3f}",   "#00c6ff")
            with c3: smetric("Test Samples", f"{len(yt):,}",                          "#ffd166")
            with c4: smetric("Classes",      str(len(le.classes_)),                   "#a29bfe")

            ca, cb = st.columns(2)
            with ca:
                fig, ax = dark_fig((6,5))
                sns.heatmap(cm2,annot=True,fmt="d",cmap="Purples",
                            xticklabels=le.classes_,yticklabels=le.classes_,
                            linewidths=0.5,linecolor="#060d18",ax=ax,cbar=False)
                ax.tick_params(colors="#94a3b8",labelsize=8)
                ax.set_xlabel("Predicted",color="#94a3b8")
                ax.set_ylabel("Actual",color="#94a3b8")
                plt.xticks(rotation=30,ha="right")
                ax.set_title("Root Cause Confusion Matrix",color="#00c6ff",fontweight="bold")
                st.pyplot(fig, use_container_width=True)
            with cb:
                classes = [c for c in rpt if c not in ["accuracy","macro avg","weighted avg"]]
                f1s     = [rpt[c]["f1-score"] for c in classes]
                clrs    = ["#ff6b6b" if f<0.7 else "#ffd166" if f<0.9 else "#00ff88" for f in f1s]
                fig, ax = dark_fig((6,5))
                bars    = ax.barh(classes, f1s, color=clrs, edgecolor="#060d18")
                for bar, val in zip(bars,f1s):
                    ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,
                            f"{val:.3f}",va="center",color="#e6f1ff",fontsize=9)
                ax.set_xlim(0,1.1); ax.set_xlabel("F1 Score",color="#94a3b8")
                ax.tick_params(colors="#94a3b8",labelsize=9)
                for sp in ax.spines.values(): sp.set_edgecolor("#112240")
                ax.set_title("F1 per Root Cause",color="#00c6ff",fontweight="bold")
                st.pyplot(fig, use_container_width=True)

            st.subheader("Top 20 Feature Importances")
            fn  = np.array(st.session_state["tfidf_rf"].get_feature_names_out())
            imp = st.session_state["rf_model"].feature_importances_
            top = np.argsort(imp)[-20:]
            fig, ax = dark_fig((10,4))
            ax.barh(fn[top], imp[top], color="#00c6ff", edgecolor="#060d18")
            ax.tick_params(colors="#94a3b8",labelsize=9)
            for sp in ax.spines.values(): sp.set_edgecolor("#112240")
            ax.set_xlabel("Importance",color="#94a3b8")
            ax.set_title("Feature Importances",color="#00c6ff",fontweight="bold")
            st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.header("Real-Time Log Simulation Monitor")
    st.markdown("""
    <div class="tamil-box">
        💬 Real company-ல servers continuous-ஆ logs generate பண்ணும்.
        நாம் அதை simulate பண்ணி real-time-ல anomaly detect பண்றோம். 🚨
    </div>""", unsafe_allow_html=True)

    cb1,cb2,_ = st.columns([1,1,5])
    start_sim = cb1.button("▶ Start", key="start_sim")
    stop_sim  = cb2.button("⏹ Stop",  key="stop_sim")

    if stop_sim:  st.session_state["sim_running"] = False
    if start_sim:
        st.session_state["sim_running"] = True
        st.session_state["sim_logs"]    = []

    if st.session_state.get("sim_running", False):
        ph_logs, ph_stats = st.empty(), st.empty()
        sim_logs, anom_cnt = [], 0

        for _ in range(200):
            if not st.session_state.get("sim_running", False): break
            entry = generate_row(len(sim_logs)+1, datetime.datetime.now())
            sim_logs.append(entry)
            if entry["Label"]==1: anom_cnt+=1

            html=""
            for e in sim_logs[-18:]:
                css   = ("log-error" if e["Level"]=="ERROR" else
                         "log-warn"  if e["Level"]=="WARN"  else
                         "log-debug" if e["Level"]=="DEBUG" else "log-info")
                badge = (f'<span class="anomaly-badge">ANOMALY · {e["RootCause"]}</span>'
                         if e["Label"]==1 else '<span class="normal-badge">NORMAL</span>')
                html += f'<div class="log-line {css}">{e["FullLog"]} &nbsp;{badge}</div>'

            ph_logs.markdown(
                f'<div style="background:#060d18;border:1px solid #112240;border-radius:8px;'
                f'padding:10px;height:340px;overflow-y:auto;">{html}</div>',
                unsafe_allow_html=True)

            ts = len(sim_logs)
            with ph_stats.container():
                s1,s2,s3,s4 = st.columns(4)
                with s1: smetric("Processed",  str(ts),                          "#00c6ff")
                with s2: smetric("Anomalies",  str(anom_cnt),                    "#ff6b6b")
                with s3: smetric("Normal",     str(ts-anom_cnt),                 "#00ff88")
                with s4: smetric("Anomaly %",  f"{anom_cnt/max(ts,1)*100:.1f}%", "#ffd166")

            time.sleep(1.0/max(sim_speed,1))

        st.session_state["sim_running"]=False
        st.success(f"Simulation complete — {len(sim_logs)} lines, {anom_cnt} anomalies.")
    else:
        st.info("Click **Start** to begin live log monitoring simulation.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Predict Anomaly & Root Cause")
    st.markdown("""
    <div class="tamil-box">
        💬 எந்த log line-யும் paste பண்ணு. AI உடனே Anomaly/Normal + Root Cause சொல்லும். 🎯
    </div>""", unsafe_allow_html=True)

    sc1,sc2,sc3 = st.columns(3)
    samples = {
        "Disk Error":     "2024-01-01 10:00:00 ERROR [server-01] kernel[1234]: Disk I/O error on /dev/sda1 bad sector detected",
        "Security Alert": "2024-01-01 10:01:00 ERROR [web-01] sshd[5678]: Failed login attempt 55 tries in 10 seconds",
        "Memory Issue":   "2024-01-01 10:02:00 ERROR [app-server] java-app[9012]: OutOfMemoryError Java heap space GC overhead",
    }
    if sc1.button("💾 Disk Error"):     st.session_state["pi"] = samples["Disk Error"]
    if sc2.button("🔐 Security Alert"): st.session_state["pi"] = samples["Security Alert"]
    if sc3.button("🧠 Memory Issue"):   st.session_state["pi"] = samples["Memory Issue"]

    log_input    = st.text_area("Paste log lines (one per line)",
                                value=st.session_state.get("pi",""), height=180)
    model_choice = st.radio("Model",
                            ["Random Forest — Root Cause","Isolation Forest — Anomaly Only"],
                            horizontal=True)

    if st.button("Analyse Logs", key="btn_predict"):
        lines = [l.strip() for l in log_input.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("No lines entered.")
        else:
            use_rf = "Random Forest" in model_choice
            tk,mk  = ("tfidf_rf","rf_model") if use_rf else ("tfidf_iso","iso_model")
            if tk not in st.session_state or mk not in st.session_state:
                st.error("Model not trained yet — go to Train Models tab first.")
            else:
                tfidf = st.session_state[tk]
                model = st.session_state[mk]
                le    = st.session_state.get("le")
                X_new = tfidf.transform([clean_log(l) for l in lines])

                if use_rf and le:
                    enc      = model.predict(X_new)
                    preds_rc = le.inverse_transform(enc)
                    probas   = model.predict_proba(X_new).max(axis=1)
                    preds_b  = [0 if p=="Normal" else 1 for p in preds_rc]
                else:
                    raw      = model.predict(X_new)
                    preds_b  = [1 if p==-1 else 0 for p in raw]
                    sc_      = model.decision_function(X_new)
                    probas   = 1-(sc_-sc_.min())/(sc_.max()-sc_.min()+1e-9)
                    preds_rc = ["Anomaly" if p==1 else "Normal" for p in preds_b]

                na = sum(preds_b)
                r1,r2,r3 = st.columns(3)
                with r1: smetric("Lines Analysed",str(len(lines)),    "#00c6ff")
                with r2: smetric("Anomalies",     str(na),            "#ff6b6b")
                with r3: smetric("Normal",         str(len(lines)-na),"#00ff88")
                st.markdown("<br>",unsafe_allow_html=True)

                for line, pred, rc, prob in zip(lines, preds_b, preds_rc, probas):
                    if pred==1:
                        st.markdown(f"""
                        <div style="background:#1a0808;border:1px solid #ff4444;
                                    border-radius:8px;padding:12px 16px;margin:6px 0;">
                          <div style="font-family:'Share Tech Mono',monospace;
                                      font-size:0.77rem;color:#ff6b6b;">{line}</div>
                          <div style="margin-top:6px;">
                            <span style="color:#ff4444;font-weight:bold;">ANOMALY DETECTED</span>
                            <span style="color:#ffd166;margin-left:16px;">Root Cause: <b>{rc}</b></span>
                            <span style="color:#94a3b8;margin-left:12px;font-size:0.85rem;">
                              Confidence: {prob*100:.1f}%</span>
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#081a10;border:1px solid #00ff8844;
                                    border-radius:8px;padding:12px 16px;margin:6px 0;">
                          <div style="font-family:'Share Tech Mono',monospace;
                                      font-size:0.77rem;color:#4ade80;">{line}</div>
                          <div style="margin-top:6px;">
                            <span style="color:#00ff88;font-weight:bold;">NORMAL</span>
                            <span style="color:#94a3b8;margin-left:16px;font-size:0.85rem;">
                              Confidence: {prob*100:.1f}%</span>
                          </div>
                        </div>""", unsafe_allow_html=True)

# FOOTER
st.markdown('<hr class="section-hr">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.85rem;padding:10px 0 20px;">
    IT Infrastructure Log Anomaly Detection System &nbsp;·&nbsp;
    Isolation Forest + Random Forest &nbsp;·&nbsp;
    Python · Streamlit · scikit-learn
</div>""", unsafe_allow_html=True)
