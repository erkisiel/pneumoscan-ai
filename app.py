"""PneumoScan AI: a radiology decision-support workstation (Streamlit).

Clinical workflow
  sign in (role-based)  ->  import X-ray (PNG/JPG or DICOM, de-identified on import)
  ->  AI analysis (probability, triage, Grad-CAM, uncertainty)  ->  compare with prior studies
  ->  physician edits and signs the report  ->  PDF export.  Every step is written to an audit log.

Run:  streamlit run app.py        Demo accounts: dr.demo / demo123 (physician), admin / admin123 (admin)
"""
import hashlib
import io
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
import db  # noqa: E402
from data import build_transforms  # noqa: E402
from dicom_io import anonymize, read_dicom  # noqa: E402
from gradcam import GradCAM, overlay  # noqa: E402
from model import load_model  # noqa: E402
from uncertainty import confidence_level, predict_with_uncertainty  # noqa: E402

CKPT = ROOT / "checkpoints" / "best_model.pt"
RES = ROOT / "results"
SAMPLES = ROOT / "samples"
MODEL_VERSION = "ResNet18 · v1.1"
COLORS = {"urgent": "#C0392B", "review": "#D68910", "routine": "#1E8449"}
LEVEL_TXT = {"urgent": "High suspicion", "review": "Possible pneumonia", "routine": "No pneumonia detected"}

st.set_page_config(page_title="PneumoScan AI", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
:root { --navy:#0F2A44; --teal:#0B7A8A; --red:#C0392B; --amber:#D68910; --green:#1E8449;
        --bg:#F4F6F9; --card:#FFFFFF; --muted:#6B7684; --line:#E3E7ED; }
.stApp { background: var(--bg); }
.block-container { padding-top: 4.6rem; padding-bottom: 2rem; max-width: 1500px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"] { visibility: hidden; }

.topbar { position: fixed; top: 0; left: 0; right: 0; height: 56px; z-index: 1000100;
          background: linear-gradient(90deg, var(--navy), #163B5E); color: #fff;
          display: flex; align-items: center; justify-content: space-between; padding: 0 28px;
          box-shadow: 0 2px 8px rgba(15,42,68,.25); }
.topbar .brand { font-size: 20px; font-weight: 700; letter-spacing: .3px; }
.topbar .brand span.hl { color: #6FD3E0; }
.topbar .sub { font-size: 12.5px; opacity: .8; margin-left: 14px; font-weight: 400; }
.topbar .right { font-size: 13px; opacity: .95; display:flex; gap: 14px; align-items:center; }
.pill { background: rgba(255,255,255,.14); padding: 4px 10px; border-radius: 20px; }
.pill.role { background: #6FD3E0; color: var(--navy); font-weight: 700; }

.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(16,24,40,.05); margin-bottom: 14px; color:#1F2933; }
.card h4 { margin: 0 0 10px 0; font-size: 13px; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); }
.kv { display:grid; grid-template-columns: 118px 1fr; row-gap: 6px; font-size: 14px; }
.kv .k { color: var(--muted); } .kv .v { color: #1F2933; font-weight: 600; word-break: break-word; }
.sec { font-size: 12.5px; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); font-weight: 700; margin: 4px 0 6px 0; }

.gauge { width: 164px; height: 164px; border-radius: 50%; margin: 6px auto 12px auto; position: relative;
         display:flex; align-items:center; justify-content:center; }
.gauge::before { content:""; position:absolute; inset: 16px; background:#fff; border-radius:50%; }
.gauge .val { position: relative; text-align:center; }
.gauge .val b { font-size: 33px; color: var(--navy); display:block; line-height: 1.1; }
.gauge .val small { font-size: 12px; color: var(--muted); }

.triage { border-radius: 10px; padding: 11px 12px; font-weight: 700; font-size: 14px; text-align:center; margin: 4px 0 12px 0; }
.triage.urgent { background: #FDECEA; color: var(--red); border: 1px solid #F5B7B1; }
.triage.review { background: #FEF5E7; color: var(--amber); border: 1px solid #F8C471; }
.triage.routine { background: #E9F7EF; color: var(--green); border: 1px solid #A9DFBF; }
.conf { border-radius: 10px; padding: 10px 12px; font-size: 13px; margin: 0 0 12px 0; }
.conf.high { background:#EEF6F7; border:1px solid #B9DDE2; color:#0B5F6B; }
.conf.low { background:#FFF8E6; border:1px dashed #E0A800; color:#8A6100; }
.conf b { display:block; font-size: 13.5px; margin-bottom: 2px; }
.small { font-size: 12.5px; color: var(--muted); }
.reco { padding-left: 18px; margin: 0; } .reco li { margin-bottom: 5px; font-size: 14px; }
.viewer-caption { color: #6B7684; font-size: 12px; text-align:center; margin-top: -4px; }
[data-testid="stImage"] img { border-radius: 10px; }

.kpi { background: var(--card); border:1px solid var(--line); border-radius: 12px; padding: 16px; text-align:center; }
.kpi b { display:block; font-size: 30px; color: var(--navy); }
.kpi span { font-size: 13px; color: var(--muted); }
.kpi.accent b { color: var(--teal); } .kpi.red b { color: var(--red); } .kpi.amber b { color: var(--amber); }

.badge { display:inline-block; padding: 2px 9px; border-radius: 20px; font-size: 12px; font-weight: 700; }
.badge.urgent { background:#FDECEA; color:var(--red); } .badge.review { background:#FEF5E7; color:var(--amber); }
.badge.routine { background:#E9F7EF; color:var(--green); } .badge.low { background:#FFF8E6; color:#8A6100; }
.badge.signed { background:#E8EEF6; color:var(--navy); } .badge.pending { background:#F1F3F6; color:#6B7684; }
.badge.dicom { background:#E6F4F6; color:var(--teal); }

.wl-item { background:#fff; border:1px solid var(--line); border-left: 4px solid var(--muted); border-radius: 8px;
           padding: 8px 10px; margin-bottom: 8px; font-size: 13px; color:#1F2933; }
.wl-item.urgent { border-left-color: var(--red); } .wl-item.review { border-left-color: var(--amber); }
.wl-item.routine { border-left-color: var(--green); }
.prior { display:flex; gap: 10px; align-items:center; }
.tl { border-left: 2px solid var(--line); margin-left: 6px; padding-left: 14px; }

.login-hero { background: linear-gradient(135deg, #0F2A44 0%, #0B7A8A 100%); color:#fff; border-radius: 16px;
              padding: 34px 32px; height: 100%; }
.login-hero h1 { color:#fff; font-size: 30px; margin: 0 0 6px 0; } .login-hero h1 span { color:#6FD3E0; }
.login-hero p { opacity:.9; font-size: 15px; } .login-hero li { margin-bottom: 8px; opacity:.95; }
.demo-cred { background:#F1F5F9; border:1px dashed #B8C4D0; border-radius:10px; padding:10px 12px; font-size:13px; color:#1F2933; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #E3E8EE; border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #fff !important; color: var(--teal) !important; }
section[data-testid="stSidebar"] { background: #EAEFF4; top: 56px !important; height: calc(100vh - 56px) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


def topbar(user=None):
    right = f"<span class='pill'>{MODEL_VERSION}</span><span>{datetime.now(db.TZ):%d %b %Y · %H:%M}</span>"
    if user:
        right += f"<span>👤 {user['full_name']}</span><span class='pill role'>{db.ROLES[user['role']]}</span>"
    st.markdown(f"""<div class="topbar">
      <div class="brand">🫁 Pneumo<span class="hl">Scan</span> AI <span class="sub">Chest X-ray decision support · Radiology workstation</span></div>
      <div class="right">{right}</div></div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_resource
def get_model():
    model, img_size, thr = load_model(str(CKPT))
    return model, img_size, thr, GradCAM(model, model.layer4)


@st.cache_resource
def get_db():
    return db.connect()


@st.cache_data
def load_json(name):
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else None


if not CKPT.exists():
    st.error(f"Model file not found: {CKPT}. Train it first with `python src/train.py`.")
    st.stop()

model, img_size, default_thr, cam = get_model()
con = get_db()
metrics, baseline, unc_metrics = load_json("metrics.json"), load_json("baseline_metrics.json"), load_json("uncertainty.json")


# ------------------------------------------------------------------ analysis helpers
def triage(p, thr):
    if p >= max(0.8, thr):
        return "urgent", "🔴 HIGH SUSPICION · prioritise for review"
    if p >= thr:
        return "review", "🟠 POSSIBLE PNEUMONIA · review recommended"
    return "routine", "🟢 NO PNEUMONIA DETECTED · routine reading"


def analyse(img):
    """Point prediction + Grad-CAM + MC-dropout/TTA uncertainty for one grayscale PIL image."""
    x = build_transforms(img_size, train=False)(img).unsqueeze(0).requires_grad_(True)
    heat, probs = cam(x, class_idx=1)
    _, std, samples = predict_with_uncertainty(model, img, img_size, n=20)
    return float(probs[1]), std, heat, samples


def analyse_cached(key, img):
    cache = st.session_state.setdefault("analysis_cache", {})
    if key not in cache:
        cache[key] = analyse(img)
    return cache[key]


def to_png(img):
    b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()


def recommendations(level, conf):
    rec = {
        "urgent": ["Move study to the top of the reading worklist", "Correlate with symptoms, SpO₂, CRP / WBC",
                   "Start treatment per guidelines once confirmed"],
        "review": ["Radiologist review within normal turnaround", "Correlate with clinical findings and labs",
                   "Consider follow-up imaging if symptoms persist"],
        "routine": ["Routine reading, no AI flag raised", "A negative AI result does not exclude pneumonia",
                    "Re-evaluate clinically if symptoms worsen"],
    }[level]
    if conf == "low":
        rec = ["Low AI confidence: a radiologist must read this study"] + rec[:2]
    return rec


def make_report_text(info, p, std, thr, level, conf, priors):
    impression = {
        "urgent": "Findings highly suggestive of pneumonia (AI). Prompt radiologist confirmation advised.",
        "review": "Findings possibly consistent with pneumonia (AI). Radiologist confirmation advised.",
        "routine": "No radiographic evidence of pneumonia detected by the AI model.",
    }[level]
    comp = "None available."
    if priors:
        pr = priors[0]
        comp = (f"Prior study {pr['created_at'][:10]}: pneumonia probability {pr['prob']*100:.0f}% "
                f"({LEVEL_TXT[pr['level']].lower()}). Current: {p*100:.0f}%.")
    return (
        f"EXAMINATION: Chest radiograph, frontal view\n"
        f"CLINICAL INDICATION: {info['indication']}\n"
        f"TECHNIQUE: Digital radiograph analysed by {MODEL_VERSION} (input 128x128, Grad-CAM, MC-dropout + TTA uncertainty)\n"
        f"COMPARISON: {comp}\n\n"
        f"AI FINDINGS: Pneumonia probability {p*100:.1f}% (±{std*100:.1f}%, decision threshold {thr:.2f}). "
        f"AI confidence: {conf.upper()}. Grad-CAM highlights the regions that contributed most to the score.\n\n"
        f"IMPRESSION: {impression}\n\n"
        f"PHYSICIAN COMMENT: "
    )


def make_pdf(info, gray, ov, report_text, p, level, conf, signed_by=None, signed_at=None):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    ax = fig.add_axes([0, 0.93, 1, 0.07]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0F2A44", transform=ax.transAxes))
    ax.text(0.04, 0.5, "PneumoScan AI  ·  Chest X-ray Report", color="white", fontsize=15, weight="bold", va="center")
    ax.text(0.96, 0.5, datetime.now(db.TZ).strftime("%d.%m.%Y %H:%M"), color="white", fontsize=10, va="center", ha="right")
    t = fig.add_axes([0.06, 0.80, 0.88, 0.11]); t.axis("off")
    rows = [("Patient ID", info["pid"]), ("Birth year / Sex", f"{info['birth_year'] or '-'} / {info['sex']}"),
            ("Referring physician", info["ref"] or "-"), ("Indication", info["indication"])]
    for i, (k, v) in enumerate(rows):
        t.text(0, 1 - i * 0.26, k, fontsize=10, color="#6B7684", va="top")
        t.text(0.28, 1 - i * 0.26, v, fontsize=10.5, weight="bold", va="top")
    t.text(1, 1, f"{p*100:.1f}%", fontsize=28, weight="bold", color=COLORS[level], ha="right", va="top")
    t.text(1, 0.45, "pneumonia probability", fontsize=9, color="#6B7684", ha="right", va="top")
    t.text(1, 0.22, f"AI confidence: {conf.upper()}", fontsize=9, weight="bold",
           color="#8A6100" if conf == "low" else "#0B5F6B", ha="right", va="top")
    a1 = fig.add_axes([0.06, 0.50, 0.42, 0.28]); a1.imshow(gray, cmap="gray"); a1.axis("off"); a1.set_title("Original", fontsize=10)
    a2 = fig.add_axes([0.52, 0.50, 0.42, 0.28]); a2.imshow(ov); a2.axis("off"); a2.set_title("Grad-CAM", fontsize=10)
    r = fig.add_axes([0.06, 0.13, 0.88, 0.34]); r.axis("off")
    wrapped = "\n".join(textwrap.fill(line, 95) if line else "" for line in report_text.split("\n"))
    r.text(0, 1, wrapped, fontsize=9.5, va="top", linespacing=1.5)
    s = fig.add_axes([0.06, 0.08, 0.88, 0.04]); s.axis("off")
    s.text(0, 0.5, f"Electronically signed by {signed_by} on {signed_at}" if signed_by else "DRAFT - not signed",
           fontsize=10, weight="bold", color="#0F2A44", va="center")
    f = fig.add_axes([0.06, 0.02, 0.88, 0.05]); f.axis("off")
    f.text(0, 0.5, "Generated by an academic prototype (COMP 463). Not a medical device.\n"
                   "The final diagnosis must be made by a qualified physician.", fontsize=8, color="#6B7684", va="center")
    buf = io.BytesIO(); fig.savefig(buf, format="pdf"); plt.close(fig)
    return buf.getvalue()


def badge(cls, txt):
    return f"<span class='badge {cls}'>{txt}</span>"


def study_overlay(row, alpha):
    img = Image.open(io.BytesIO(row["image_png"])).convert("L")
    p, std, heat, _ = analyse_cached(f"study-{row['id']}", img)
    gray = np.array(img.resize((img_size, img_size)))
    ov = np.array(Image.fromarray(overlay(gray, heat, alpha)).resize((512, 512), Image.BICUBIC))
    return np.array(img.resize((512, 512))), ov


def seed_demo():
    """First run only: imports two demo studies so the worklist and patient history are not empty."""
    if con.execute("SELECT COUNT(*) FROM studies").fetchone()[0] or not (SAMPLES / "dicom").exists():
        return
    seeds = [("TR-100231_2026-10-02_prior.dcm", "Fever and cough", "2026-10-02 09:14:00", True),
             ("TR-100344_2026-11-23.dcm", "Pre-operative screening", "2026-11-23 16:40:00", False)]
    for fn, ind, ts, signed in seeds:
        f = SAMPLES / "dicom" / fn
        if not f.exists():
            continue
        img, meta, ds = read_dicom(f.read_bytes())
        anon, _ = anonymize(ds)
        pid = str(anon.PatientID)
        p, std, _, _ = analyse(img)
        level, _ = triage(p, default_thr)
        conf, _ = confidence_level(p, std, default_thr)
        db.upsert_patient(con, pid, None, meta["BirthYear"], meta["PatientSex"])
        sid = db.add_study(con, pid=pid, created_at=ts, created_by="dr.demo", source="DICOM", file_name=fn,
                           indication=ind, image_png=to_png(img.resize((512, 512))), prob=p, threshold=default_thr,
                           level=level, unc_std=std, confidence=conf)
        if signed:
            con.execute("UPDATE studies SET status='signed', signed_by='dr.demo', signed_at=?, report=? WHERE id=?",
                        (ts, "IMPRESSION: Right lower lobe consolidation consistent with pneumonia. Antibiotics started.", sid))
            con.commit()
    db.log(con, "system", "SEED_DEMO", detail="demo studies imported on first run")


seed_demo()


# ------------------------------------------------------------------ login
def login_page():
    topbar()
    st.write("")
    _, hero, form, _ = st.columns([0.25, 1.25, 1, 0.25], gap="large")
    with hero:
        st.markdown("""<div class='login-hero'>
          <h1>🫁 Pneumo<span>Scan</span> AI</h1>
          <p>AI-assisted chest X-ray triage for paediatric pneumonia.</p>
          <ul>
            <li><b>DICOM import</b> with automatic de-identification</li>
            <li><b>Explainable AI</b>: Grad-CAM heatmap for every result</li>
            <li><b>Confidence flag</b> when the model is unsure</li>
            <li><b>Patient history</b> and comparison with prior studies</li>
            <li><b>Role-based access</b> and a full audit trail</li>
          </ul>
          <p class='small' style='color:#D6EEF2'>COMP 463 · Introduction to Medical Informatics · MEF University</p>
        </div>""", unsafe_allow_html=True)
    with form:
        with st.container(border=True):
            st.markdown("<div class='sec'>Sign in</div>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin:0 0 8px 0;color:#0F2A44'>Welcome back</h3>", unsafe_allow_html=True)
            with st.form("login", border=False):
                u = st.text_input("Username", placeholder="e.g. dr.demo")
                pw = st.text_input("Password", type="password")
                ok = st.form_submit_button("Sign in", type="primary", width="stretch")
            if ok:
                user = db.authenticate(con, u.strip(), pw)
                if user:
                    st.session_state.user = user
                    db.log(con, user["username"], "LOGIN", detail=f"role={user['role']}")
                    st.rerun()
                else:
                    db.log(con, u.strip() or "?", "LOGIN_FAILED")
                    st.error("Wrong username or password.")
            st.markdown("""<div class='demo-cred'><b>Demo accounts</b><br>
              Physician: <code>dr.demo</code> / <code>demo123</code><br>
              Administrator: <code>admin</code> / <code>admin123</code></div>""", unsafe_allow_html=True)
            st.caption("🔒 Passwords are stored as salted PBKDF2-SHA256 hashes. All sign-ins are logged.")


user = st.session_state.get("user")
if not user:
    login_page()
    st.stop()

topbar(user)
U = user["username"]
is_admin = user["role"] == "admin"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"""<div class='card' style='padding:12px 14px'><div class='small'>Signed in as</div>
      <b style='font-size:15px'>{user['full_name']}</b><br>{badge('signed', db.ROLES[user['role']])}</div>""",
                unsafe_allow_html=True)
    if st.button("Sign out", width="stretch"):
        db.log(con, U, "LOGOUT")
        st.session_state.clear()
        st.rerun()
    if not is_admin:
        st.markdown("### ⚙️ Settings")
        thr = st.slider("Decision threshold (τ)", 0.05, 0.95, float(default_thr), 0.01,
                        help="Lower τ = higher sensitivity (fewer missed patients) but more false alarms. "
                             f"Default {default_thr:.2f} was selected on the validation set (Youden's J).")
        alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)
    else:
        thr, alpha = default_thr, 0.45
    st.markdown("### 📋 Worklist status")
    stats_box = st.empty()
    st.markdown("---")
    st.caption("🔒 DICOM files are de-identified on import. Only pseudonymised IDs are stored. "
               "Every access is recorded in the audit log.")


# ------------------------------------------------------------------ tabs
def tab_new_study():
    left, center, right = st.columns([1.05, 1.9, 1.15], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("<div class='sec'>1 · Image</div>", unsafe_allow_html=True)
            up = st.file_uploader("Upload chest X-ray (DICOM, PNG, JPG)", type=["dcm", "dicom", "png", "jpg", "jpeg"])
            demo = sorted(p.name for p in (SAMPLES / "dicom").glob("*.dcm")) + sorted(p.name for p in SAMPLES.glob("*.png"))
            sample = st.selectbox("…or load a demo case", ["—"] + demo,
                                  help="DICOM demo cases carry synthetic patient data and test de-identification.")
            anon_on = st.toggle("De-identify DICOM on import", value=True,
                                help="Removes name, birth date, institution, physicians and accession number; "
                                     "replaces the patient ID with a pseudonym.")

    src_bytes, fname = None, None
    if up is not None:
        src_bytes, fname = up.getvalue(), up.name
    elif sample != "—":
        f = (SAMPLES / "dicom" / sample) if sample.endswith(".dcm") else (SAMPLES / sample)
        src_bytes, fname = f.read_bytes(), sample

    if src_bytes is None:
        with center:
            st.markdown("""<div class='card' style='text-align:center;padding:80px 30px'>
              <div style='font-size:56px'>🩻</div>
              <h3 style='color:#0F2A44;margin:8px 0'>No study loaded</h3>
              <p class='small' style='font-size:14px'>Upload a frontal chest X-ray (DICOM or image) or pick a demo case.<br>
              The AI estimates the probability of pneumonia, assigns a triage level, shows where the evidence
              comes from and warns you when it is not confident.</p></div>""", unsafe_allow_html=True)
        with right:
            st.markdown("""<div class='card'><h4>Workflow</h4><ol class='reco'>
              <li>Import DICOM, header read and de-identified</li><li>ResNet18 CNN outputs P(pneumonia)</li>
              <li>20 MC-dropout + TTA passes estimate uncertainty</li><li>Grad-CAM shows the supporting regions</li>
              <li>Compare with the patient's prior studies</li><li>Physician edits and signs the report</li></ol></div>""",
                        unsafe_allow_html=True)
        return

    key = hashlib.sha1(src_bytes).hexdigest()
    is_dicom = fname.lower().endswith((".dcm", ".dicom")) or src_bytes[128:132] == b"DICM"
    meta, removed = None, []
    try:
        if is_dicom:
            img, meta, ds = read_dicom(src_bytes)
            pid_default = meta["PatientID"] or "ANON"
            if anon_on:
                anon, removed = anonymize(ds)
                pid_default = str(anon.PatientID)
        else:
            img = Image.open(io.BytesIO(src_bytes)).convert("L")
            pid_default = st.session_state.setdefault("pid_" + key, f"ANON-{key[:5].upper()}")
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    with left:
        with st.container(border=True):
            st.markdown("<div class='sec'>2 · Patient & study</div>", unsafe_allow_html=True)
            known = db.get_patient(con, pid_default)
            pid = st.text_input("Patient ID (pseudonymised)" if anon_on or not is_dicom else "Patient ID", pid_default,
                                disabled=is_dicom)
            c_a, c_s = st.columns(2)
            by_def = (meta or {}).get("BirthYear") or (known["birth_year"] if known else 2022)
            birth_year = c_a.number_input("Birth year", 1900, 2030, int(by_def))
            sex_opts = ["F", "M", "O"]
            sx = (meta or {}).get("PatientSex") or (known["sex"] if known else "F")
            sex = c_s.selectbox("Sex", sex_opts, index=sex_opts.index(sx) if sx in sex_opts else 0)
            indication = st.selectbox("Clinical indication",
                                      ["Fever and cough", "Shortness of breath", "Suspected pneumonia",
                                       "Follow-up of known pneumonia", "Pre-operative screening", "Other"])
            ref_def = "" if (anon_on or not meta) else meta["ReferringPhysician"].replace("^", " ")
            ref = st.text_input("Referring physician", ref_def)
            if known:
                st.markdown(f"<div class='small'>✔ Known patient · {(n_ := len(db.patient_studies(con, pid)))} prior {'study' if n_ == 1 else 'studies'} on file</div>",
                            unsafe_allow_html=True)
        if meta:
            with st.container(border=True):
                st.markdown("<div class='sec'>DICOM header</div>", unsafe_allow_html=True)
                shown = {"Modality": meta["Modality"], "Body part": meta["BodyPart"], "View": meta["View"],
                         "Study date": meta["StudyDate"], "Matrix": meta["Size"], "Description": meta["StudyDescription"]}
                if not anon_on:
                    shown = {"Name": meta["PatientName"].replace("^", " "), "Institution": meta["Institution"]} | shown
                st.markdown("<div class='kv'>" + "".join(f"<div class='k'>{k}</div><div class='v'>{v or '-'}</div>"
                                                         for k, v in shown.items()) + "</div>", unsafe_allow_html=True)
                if anon_on:
                    st.markdown(f"<div class='small' style='margin-top:8px'>🛡️ De-identified: removed "
                                f"{', '.join(removed) or 'no identifiers found'}; ID → <b>{pid}</b></div>",
                                unsafe_allow_html=True)

    p, std, heat, samples = analyse_cached(key, img)
    level, label = triage(p, thr)
    conf, conf_why = confidence_level(p, std, thr)
    if conf == "low" and level == "routine":  # safety rule: an uncertain negative is never "routine"
        level, label = "review", "🟠 INDETERMINATE · radiologist read required"
    gray = np.array(img.resize((img_size, img_size)))
    gray_big = np.array(img.resize((512, 512)))
    ov_big = np.array(Image.fromarray(overlay(gray, heat, alpha)).resize((512, 512), Image.BICUBIC))

    # store the study once per (file, patient); later edits update the same row
    saved = st.session_state.setdefault("saved", {})
    skey = f"{key}|{pid}"
    priors = [r for r in db.patient_studies(con, pid) if r["id"] != saved.get(skey)]
    if skey not in saved:
        db.upsert_patient(con, pid, None, int(birth_year), sex)
        saved[skey] = db.add_study(con, pid=pid, created_by=U, source="DICOM" if is_dicom else "Image", file_name=fname,
                                   indication=indication, image_png=to_png(img.resize((512, 512))), prob=p,
                                   threshold=thr, level=level, unc_std=std, confidence=conf)
        if is_dicom:
            db.log(con, U, "IMPORT_DICOM", pid, f"{fname}; de-identified={anon_on}; removed={len(removed)} tags")
        db.log(con, U, "ANALYSE", pid, f"study #{saved[skey]} p={p:.3f} ±{std:.3f} level={level} conf={conf}")
    else:
        db.update_study(con, saved[skey], threshold=thr, level=level, confidence=conf, indication=indication)
    study_id = saved[skey]
    study = db.get_study(con, study_id)

    with center:
        with st.container(border=True):
            mode = st.radio("View", ["Side by side", "Original", "Grad-CAM"], horizontal=True, label_visibility="collapsed")
            if mode == "Side by side":
                v1, v2 = st.columns(2)
                v1.image(gray_big, width="stretch"); v1.markdown("<div class='viewer-caption'>Original · frontal view</div>", unsafe_allow_html=True)
                v2.image(ov_big, width="stretch"); v2.markdown("<div class='viewer-caption'>Grad-CAM · evidence for pneumonia</div>", unsafe_allow_html=True)
            elif mode == "Original":
                st.image(gray_big, width="stretch")
            else:
                st.image(ov_big, width="stretch")
            st.caption(f"Study #{study_id} · {fname} · red = regions that raised the pneumonia score, blue = little contribution")

        if priors:
            with st.container(border=True):
                st.markdown("<div class='sec'>Comparison with prior studies</div>", unsafe_allow_html=True)
                cols = st.columns(min(len(priors), 3) + 1)
                for col, pr in zip(cols, priors[:3]):
                    col.image(pr["image_png"], width="stretch")
                    col.markdown(f"<div class='small' style='text-align:center'><b>{pr['created_at'][:10]}</b><br>"
                                 f"{pr['prob']*100:.0f}% · {badge(pr['level'], LEVEL_TXT[pr['level']])}</div>",
                                 unsafe_allow_html=True)
                delta = p - priors[0]["prob"]
                trend = ("⬇ Improving" if delta < -0.2 else "⬆ Worsening" if delta > 0.2 else "➡ Stable")
                cols[-1].markdown(f"""<div class='card' style='margin:0'><h4>Trend</h4>
                    <b style='font-size:22px;color:#0F2A44'>{trend}</b>
                    <div class='small'>{priors[0]['prob']*100:.0f}% → {p*100:.0f}%<br>since {priors[0]['created_at'][:10]}</div></div>""",
                                  unsafe_allow_html=True)

    with right:
        auc = f"{metrics['auc']:.3f}" if metrics else "–"
        conf_html = (f"<div class='conf low'><b>⚠️ Low AI confidence · second read required</b>{conf_why}.</div>"
                     if conf == "low" else
                     f"<div class='conf high'><b>✔ High AI confidence</b>{conf_why}.</div>")
        st.markdown(f"""<div class='card'><h4>AI findings</h4>
          <div class='gauge' style='background: conic-gradient({COLORS[level]} {p*360:.1f}deg, #E8ECF1 0deg)'>
            <div class='val'><b>{p*100:.1f}%</b><small>± {std*100:.1f}% · pneumonia</small></div></div>
          <div class='triage {level}'>{label}</div>{conf_html}
          <div class='kv'><div class='k'>Threshold τ</div><div class='v'>{thr:.2f}</div>
            <div class='k'>Model</div><div class='v'>{MODEL_VERSION}</div>
            <div class='k'>Test AUC</div><div class='v'>{auc}</div></div></div>""", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div class='sec'>Uncertainty · 20 stochastic passes</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(3.6, 1.25))
            ax.hist(samples, bins=np.linspace(0, 1, 26), color=COLORS[level], alpha=.85)
            ax.axvline(thr, ls="--", c="#0F2A44", lw=1); ax.set_yticks([]); ax.set_xlim(0, 1)
            ax.tick_params(labelsize=7)
            for s_ in ["top", "right", "left"]:
                ax.spines[s_].set_visible(False)
            fig.tight_layout(pad=0.2); st.pyplot(fig, width="stretch"); plt.close(fig)
            st.caption("Each bar = predictions under MC dropout + small image perturbations. Dashed line = τ.")
        st.markdown("<div class='card'><h4>Suggested next steps</h4><ul class='reco'>" +
                    "".join(f"<li>{r}</li>" for r in recommendations(level, conf)) + "</ul></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='sec'>3 · Report · edited and signed by the physician</div>", unsafe_allow_html=True)
        info = {"pid": pid, "birth_year": birth_year, "sex": sex, "indication": indication, "ref": ref}
        signed = study["status"] == "signed"
        default_txt = study["report"] if signed else make_report_text(info, p, std, thr, level, conf, priors)
        report = st.text_area("Report", default_txt, height=230, label_visibility="collapsed", disabled=signed,
                              key=f"report_{study_id}_{level}_{indication}_{conf}")
        c1, c2, c3 = st.columns([1.4, 1.2, 1.6])
        if signed:
            c1.markdown(f"{badge('signed', '✔ Signed')} <span class='small'>by {study['signed_by']} · {study['signed_at']}</span>",
                        unsafe_allow_html=True)
        else:
            reviewed = c1.checkbox("I have reviewed the images and the AI result")
            if c2.button("✍️  Sign report", type="primary", disabled=not reviewed):
                db.sign_study(con, study_id, U, report)
                db.log(con, U, "SIGN_REPORT", pid, f"study #{study_id}")
                st.rerun()
        study = db.get_study(con, study_id)
        c3.download_button("⬇️  Download PDF report",
                           data=make_pdf(info, gray_big, ov_big, study["report"] or report, p, level, conf,
                                         study["signed_by"], study["signed_at"]),
                           file_name=f"{pid}_study{study_id}_report.pdf", mime="application/pdf", disabled=not signed,
                           on_click=lambda: db.log(con, U, "EXPORT_PDF", pid, f"study #{study_id}"))
        st.caption("⚠️ Decision support only. The AI result must be confirmed by a qualified physician. "
                   "The PDF can be exported after the report is signed.")


def tab_worklist():
    st.markdown("#### Reading worklist")
    st.caption("Sorted by urgency, then by AI confidence: urgent and uncertain studies come first.")
    f1, f2, _ = st.columns([1, 1, 3])
    status = f1.selectbox("Status", ["pending", "signed", "all"], index=0)
    only_low = f2.toggle("Only low confidence", False)
    rows = db.worklist(con, None if status == "all" else status)
    if only_low:
        rows = [r for r in rows if r["confidence"] == "low"]
    if not rows:
        st.info("No studies match this filter.")
        return
    df = pd.DataFrame([{"Study": f"#{r['id']}", "Priority": {"urgent": "🔴 Urgent", "review": "🟠 Review", "routine": "🟢 Routine"}[r["level"]],
                        "Patient": r["pid"], "P(pneumonia)": r["prob"], "± Uncertainty": r["unc_std"],
                        "Confidence": "⚠️ Low" if r["confidence"] == "low" else "High",
                        "Source": r["source"], "Indication": r["indication"], "Received": r["created_at"][:16],
                        "Status": "✔ Signed" if r["status"] == "signed" else "Pending"} for r in rows])
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"P(pneumonia)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                "± Uncertainty": st.column_config.NumberColumn(format="percent")})
    sel = st.selectbox("Open study", [f"#{r['id']} · {r['pid']}" for r in rows], index=None, placeholder="Choose a study…")
    if sel:
        show_study(db.get_study(con, int(sel.split()[0][1:])))


def show_study(r):
    if st.session_state.get("last_viewed") != r["id"]:
        db.log(con, U, "VIEW_STUDY", r["pid"], f"study #{r['id']}")
        st.session_state.last_viewed = r["id"]
    a, b = st.columns([1.6, 1])
    g, ov = study_overlay(r, alpha)
    with a, st.container(border=True):
        v1, v2 = st.columns(2)
        v1.image(g, width="stretch"); v2.image(ov, width="stretch")
    with b:
        st.markdown(f"""<div class='card'><h4>Study #{r['id']}</h4><div class='kv'>
          <div class='k'>Patient</div><div class='v'>{r['pid']}</div>
          <div class='k'>Received</div><div class='v'>{r['created_at'][:16]}</div>
          <div class='k'>AI result</div><div class='v'>{r['prob']*100:.1f}% ± {r['unc_std']*100:.1f}% {badge(r['level'], LEVEL_TXT[r['level']])}</div>
          <div class='k'>Confidence</div><div class='v'>{badge('low' if r['confidence']=='low' else 'signed', r['confidence'].upper())}</div>
          <div class='k'>Status</div><div class='v'>{badge(r['status'], r['status'].upper())} {('by ' + r['signed_by']) if r['signed_by'] else ''}</div>
          </div></div>""", unsafe_allow_html=True)
        if r["report"]:
            st.text_area("Signed report", r["report"], height=160, disabled=True)
        else:
            st.caption("Report not signed yet. Load the same image in New study to complete it.")


def tab_patients():
    st.markdown("#### Patient records")
    q = st.text_input("Search patient ID", placeholder="PS-… or ANON-…")
    pts = db.list_patients(con, q)
    if not pts:
        st.info("No patients yet.")
        return
    df = pd.DataFrame([{"Patient ID": p_["pid"], "Birth year": p_["birth_year"], "Sex": p_["sex"], "Studies": p_["n_studies"],
                        "Last study": (p_["last_study"] or "")[:16],
                        "Last result": LEVEL_TXT.get(p_["last_level"], "-")} for p_ in pts])
    st.dataframe(df, hide_index=True, width="stretch")
    pid = st.selectbox("Open patient record", [p_["pid"] for p_ in pts], index=None, placeholder="Choose a patient…")
    if not pid:
        return
    if st.session_state.get("last_patient") != pid:
        db.log(con, U, "VIEW_PATIENT", pid)
        st.session_state.last_patient = pid
    studies = db.patient_studies(con, pid)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("<div class='sec'>Probability over time</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 2.6))
            xs = [r["created_at"][:10] for r in reversed(studies)]; ys = [r["prob"] * 100 for r in reversed(studies)]
            ax.plot(xs, ys, "-o", color="#0B7A8A"); ax.axhline(thr * 100, ls="--", c="#C0392B", lw=1, label=f"τ = {thr:.2f}")
            ax.set_ylim(0, 100); ax.set_ylabel("P(pneumonia) %"); ax.legend(fontsize=8); ax.tick_params(labelsize=8)
            for s_ in ["top", "right"]:
                ax.spines[s_].set_visible(False)
            fig.tight_layout(); st.pyplot(fig, width="stretch"); plt.close(fig)
    with right:
        st.markdown("<div class='sec'>Study timeline</div>", unsafe_allow_html=True)
        for r in studies:
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                c1.image(r["image_png"], width="stretch")
                c2.markdown(f"<b>{r['created_at'][:16]}</b> · Study #{r['id']} · {badge('dicom', r['source'])}<br>"
                            f"{r['indication']}<br>AI: <b>{r['prob']*100:.0f}%</b> {badge(r['level'], LEVEL_TXT[r['level']])} "
                            f"{badge('low', 'Low confidence') if r['confidence']=='low' else ''} {badge(r['status'], r['status'])}",
                            unsafe_allow_html=True)


def tab_performance():
    if not metrics:
        st.info("No results yet. Run `python src/evaluate.py --data <data>` to create them.")
        return
    a5, av = metrics["at_0.5"], metrics["at_val_threshold"]
    st.markdown(f"#### Evaluation on the independent test set ({metrics['n_test']} images)")
    cards = [("ROC AUC", f"{metrics['auc']:.3f}", ""), ("Accuracy · τ 0.50", f"{a5['accuracy']*100:.1f}%", ""),
             ("Specificity · τ 0.50", f"{a5['specificity']*100:.1f}%", ""),
             (f"Sensitivity · τ {av['threshold']:.2f}", f"{av['sensitivity_recall']*100:.1f}%", "accent"),
             ("Missed pneumonia cases", f"{av['fn']} / {av['fn'] + av['tp']}", "accent")]
    for col, (lab, val, cls) in zip(st.columns(5), cards):
        col.markdown(f"<div class='kpi {cls}'><b>{val}</b><span>{lab}</span></div>", unsafe_allow_html=True)
    st.write("")
    rows = []
    if baseline:
        rows.append(["Logistic regression (baseline)", f"{baseline['auc']:.3f}", baseline["accuracy"], baseline["sensitivity"], baseline["specificity"]])
    rows += [["ResNet18 · τ 0.50", f"{metrics['auc']:.3f}", a5["accuracy"], a5["sensitivity_recall"], a5["specificity"]],
             [f"ResNet18 · τ {av['threshold']:.2f} (screening)", f"{metrics['auc']:.3f}", av["accuracy"], av["sensitivity_recall"], av["specificity"]]]
    html = ("<div class='card'><h4>Model comparison</h4><table style='width:100%;border-collapse:collapse;font-size:14px'>"
            "<tr style='color:#6B7684;text-align:left'><th>Model</th><th>AUC</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th></tr>")
    for r in rows:
        html += ("<tr style='border-top:1px solid #E3E7ED'>" + f"<td style='padding:9px 0'><b>{r[0]}</b></td><td>{r[1]}</td>" +
                 "".join(f"<td>{v*100:.1f}%</td>" for v in r[2:]) + "</tr>")
    st.markdown(html + "</table></div>", unsafe_allow_html=True)

    if unc_metrics:
        u = unc_metrics["test"]
        st.markdown("#### Does the confidence flag catch the mistakes?")
        k = [(f"{u['errors_in_low_conf']} / {u['errors']}", "errors flagged as low confidence", "amber"),
             (f"{u['accuracy_high_conf']*100:.1f}%", "accuracy on high-confidence cases", "accent"),
             (f"{u['coverage_high_conf']*100:.0f}%", "of studies are high confidence", ""),
             (f"{u['error_rate_low_conf']*100:.0f}%", "error rate when flagged low", "red")]
        for col, (val, lab, cls) in zip(st.columns(4), k):
            col.markdown(f"<div class='kpi {cls}'><b>{val}</b><span>{lab}</span></div>", unsafe_allow_html=True)
        st.write("")
        if (RES / "uncertainty.png").exists():
            with st.container(border=True):
                st.image(str(RES / "uncertainty.png"), width="stretch")

    g1, g2 = st.columns(2)
    with g1, st.container(border=True):
        st.markdown("<div class='sec'>ROC curve</div>", unsafe_allow_html=True)
        st.image(str(RES / "roc.png"), width="stretch")
    with g2, st.container(border=True):
        st.markdown("<div class='sec'>Confusion matrix (screening threshold)</div>", unsafe_allow_html=True)
        st.image(str(RES / "confusion_matrix.png"), width="stretch")
    with st.container(border=True):
        st.markdown("<div class='sec'>Training curves</div>", unsafe_allow_html=True)
        st.image(str(RES / "training_curves.png"), width="stretch")
    with st.container(border=True):
        st.markdown("<div class='sec'>Grad-CAM examples (test set)</div>", unsafe_allow_html=True)
        st.image(str(RES / "gradcam_examples.png"), width="stretch")


def tab_dashboard():
    s = db.stats(con)
    st.markdown("#### System overview")
    k = [(s["n"], "studies analysed", ""), (s["patients"], "patients", ""), (s["pending"], "awaiting signature", "amber"),
         (s["urgent"], "urgent findings", "red"), (s["low_conf"], "low-confidence results", "amber")]
    for col, (val, lab, cls) in zip(st.columns(5), k):
        col.markdown(f"<div class='kpi {cls}'><b>{val}</b><span>{lab}</span></div>", unsafe_allow_html=True)
    st.write("")
    rows = con.execute("SELECT level, confidence, substr(created_at,1,10) d FROM studies").fetchall()
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.markdown("<div class='sec'>Triage distribution</div>", unsafe_allow_html=True)
            st.bar_chart(df["level"].map(LEVEL_TXT).value_counts(), color="#0B7A8A", height=240, horizontal=True)
        with c2, st.container(border=True):
            st.markdown("<div class='sec'>Studies per day</div>", unsafe_allow_html=True)
            st.bar_chart(df["d"].value_counts().sort_index(), color="#0F2A44", height=240, horizontal=True)
    st.caption("Administrators see aggregate numbers only. Images and reports are restricted to physicians (least privilege).")


def tab_audit():
    st.markdown("#### Audit trail")
    st.caption("Append-only record of every sign-in, data access, analysis, signature and export (who, what, when, which patient).")
    c1, c2, _ = st.columns([1, 1, 2])
    users = ["all"] + [r["username"] for r in db.list_users(con)] + ["system"]
    fu = c1.selectbox("User", users)
    actions = ["all", "LOGIN", "LOGIN_FAILED", "LOGOUT", "IMPORT_DICOM", "ANALYSE", "VIEW_STUDY", "VIEW_PATIENT",
               "SIGN_REPORT", "EXPORT_PDF", "EXPORT_AUDIT", "USER_CREATE", "USER_STATUS", "SEED_DEMO"]
    fa = c2.selectbox("Action", actions)
    rows = db.audit_entries(con, username=None if fu == "all" else fu, action=None if fa == "all" else fa)
    df = pd.DataFrame([{"Time": r["ts"], "User": r["username"], "Action": r["action"], "Patient": r["pid"] or "",
                        "Detail": r["detail"]} for r in rows])
    st.dataframe(df, hide_index=True, width="stretch", height=420)
    st.download_button("⬇️  Export audit log (CSV)", df.to_csv(index=False).encode(), "audit_log.csv", "text/csv",
                       on_click=lambda: db.log(con, U, "EXPORT_AUDIT", detail=f"{len(df)} rows"))


def tab_users():
    st.markdown("#### User accounts")
    us = db.list_users(con)
    st.dataframe(pd.DataFrame([{"Username": r["username"], "Name": r["full_name"], "Role": db.ROLES[r["role"]],
                                "Active": bool(r["active"]), "Created": r["created_at"]} for r in us]),
                 hide_index=True, width="stretch")
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(border=True):
        st.markdown("<div class='sec'>Add user</div>", unsafe_allow_html=True)
        with st.form("add_user", clear_on_submit=True, border=False):
            nu = st.text_input("Username"); nn = st.text_input("Full name")
            npw = st.text_input("Initial password", type="password")
            nr = st.selectbox("Role", list(db.ROLES), format_func=db.ROLES.get)
            if st.form_submit_button("Create account", type="primary"):
                if not nu or len(npw) < 6:
                    st.error("Username required and password must be at least 6 characters.")
                elif any(r["username"] == nu for r in us):
                    st.error("Username already exists.")
                else:
                    db.create_user(con, nu.strip(), npw, nr, nn or nu)
                    db.log(con, U, "USER_CREATE", detail=f"{nu} ({nr})")
                    st.rerun()
    with c2, st.container(border=True):
        st.markdown("<div class='sec'>Enable / disable account</div>", unsafe_allow_html=True)
        others = [r for r in us if r["username"] != U]
        if others:
            who = st.selectbox("Account", [r["username"] for r in others])
            cur = next(r for r in others if r["username"] == who)
            label = "Disable account" if cur["active"] else "Enable account"
            if st.button(label):
                db.set_active(con, who, not cur["active"])
                db.log(con, U, "USER_STATUS", detail=f"{who} active={not cur['active']}")
                st.rerun()
    st.markdown("""<div class='card'><h4>Access policy</h4><ul class='reco'>
      <li><b>Physician</b>: import and analyse studies, view patient history, sign and export reports</li>
      <li><b>Administrator</b>: manage accounts, view audit trail and aggregate statistics; no access to images or reports</li>
      <li>Passwords: salted PBKDF2-SHA256 (200,000 iterations); failed sign-ins are logged</li></ul></div>""",
                unsafe_allow_html=True)


def tab_about():
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        st.markdown("""<div class='card'><h4>Purpose</h4>
        <p>Pneumonia is the leading infectious cause of death in children under five worldwide (WHO).
        Chest X-ray is the first-line test, but reading it requires a radiologist, who is not always available,
        and studies can wait hours in the queue.</p>
        <p><b>PneumoScan AI</b> is a decision-support prototype that <b>screens chest X-rays for pneumonia,
        prioritises suspicious and uncertain studies on the worklist and explains each result with a heatmap</b>,
        so the physician can decide faster and with an extra safety net. It does not replace the physician.</p></div>""",
                    unsafe_allow_html=True)
        st.markdown("""<div class='card'><h4>Medical informatics features</h4><ul class='reco'>
          <li><b>Interoperability:</b> reads DICOM, the imaging standard used by every hospital PACS</li>
          <li><b>Privacy by design:</b> de-identification on import (DICOM PS3.15 profile subset), pseudonymised IDs,
              role-based access control, salted password hashes, complete audit trail (KVKK / GDPR, HIPAA principles)</li>
          <li><b>Electronic patient record:</b> SQLite database of patients and studies with longitudinal comparison</li>
          <li><b>Clinical decision support:</b> triage, uncertainty flag, physician sign-off before any report leaves the system</li></ul></div>""",
                    unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div class='sec'>System pipeline</div>", unsafe_allow_html=True)
            if (RES / "architecture.png").exists():
                st.image(str(RES / "architecture.png"), width="stretch")
    with b:
        st.markdown("""<div class='card'><h4>Intended use</h4><ul class='reco'>
          <li>Users: emergency physicians, paediatricians, radiologists</li>
          <li>Input: frontal paediatric chest X-ray (DICOM / PNG / JPG)</li>
          <li>Output: probability, triage, confidence, heatmap, signed report</li>
          <li>Role: triage and second opinion, never autonomous diagnosis</li></ul></div>
        <div class='card'><h4>Model & data</h4><ul class='reco'>
          <li>ResNet18 CNN trained from scratch (PyTorch)</li>
          <li>PneumoniaMNIST: 5,856 paediatric X-rays (Kermany et al., 2018)</li>
          <li>Explainability: Grad-CAM (Selvaraju et al., 2017)</li>
          <li>Uncertainty: MC dropout (Gal & Ghahramani, 2016) + test-time augmentation</li></ul></div>
        <div class='card'><h4>Limitations</h4><ul class='reco'>
          <li>Single-centre paediatric data; external validation needed</li>
          <li>Low resolution (128×128); normal vs pneumonia only</li>
          <li>Demo DICOM cases use synthetic patient data</li>
          <li>Not a certified medical device (EU MDR / SaMD)</li></ul></div>
        <div class='card'><h4>Project</h4><div class='kv'>
          <div class='k'>Course</div><div class='v'>COMP 463 · Intro to Medical Informatics</div>
          <div class='k'>Student</div><div class='v'>Elif Erkişi · 042401169</div>
          <div class='k'>Instructor</div><div class='v'>Burak Arslan</div>
          <div class='k'>University</div><div class='v'>MEF University</div></div></div>""", unsafe_allow_html=True)


if is_admin:
    t = st.tabs(["📈  Dashboard", "🛡️  Audit trail", "👥  Users", "📊  Model performance", "ℹ️  About"])
    for tab, fn in zip(t, [tab_dashboard, tab_audit, tab_users, tab_performance, tab_about]):
        with tab:
            fn()
else:
    t = st.tabs(["🩻  New study", "📋  Worklist", "🗂️  Patients", "📊  Model performance", "ℹ️  About"])
    for tab, fn in zip(t, [tab_new_study, tab_worklist, tab_patients, tab_performance, tab_about]):
        with tab:
            fn()

with stats_box.container():  # filled last so it includes the study analysed in this run
    s = db.stats(con)
    c1, c2 = st.columns(2)
    c1.metric("Pending", s["pending"]); c2.metric("Urgent", s["urgent"])
    c1.metric("Low confidence", s["low_conf"]); c2.metric("Signed", s["signed"])
