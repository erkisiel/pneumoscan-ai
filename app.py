"""PneumoScan AI: a radiology decision-support workstation (Streamlit).

Upload a chest X-ray -> pneumonia probability, triage level, Grad-CAM heatmap,
and a preliminary report the physician can edit, approve and download as PDF.

Run:  streamlit run app.py
"""
import io
import json
import random
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from data import build_transforms  # noqa: E402
from gradcam import GradCAM, overlay  # noqa: E402
from model import load_model  # noqa: E402

CKPT = ROOT / "checkpoints" / "best_model.pt"
RES = ROOT / "results"
SAMPLES = ROOT / "samples"
MODEL_VERSION = "ResNet18 · v1.0"

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
.topbar .right { font-size: 13px; opacity: .9; display:flex; gap: 18px; align-items:center; }
.pill { background: rgba(255,255,255,.14); padding: 4px 10px; border-radius: 20px; }

.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(16,24,40,.05); margin-bottom: 14px; color:#1F2933; }
.card h4 { margin: 0 0 10px 0; font-size: 13px; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); }
.kv { display:grid; grid-template-columns: 110px 1fr; row-gap: 6px; font-size: 14px; }
.kv .k { color: var(--muted); } .kv .v { color: #1F2933; font-weight: 600; }
.sec { font-size: 12.5px; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); font-weight: 700; margin: 4px 0 6px 0; }

.gauge { width: 170px; height: 170px; border-radius: 50%; margin: 6px auto 12px auto; position: relative;
         display:flex; align-items:center; justify-content:center; }
.gauge::before { content:""; position:absolute; inset: 16px; background:#fff; border-radius:50%; }
.gauge .val { position: relative; text-align:center; }
.gauge .val b { font-size: 34px; color: var(--navy); display:block; line-height: 1.1; }
.gauge .val small { font-size: 12px; color: var(--muted); }

.triage { border-radius: 10px; padding: 11px 12px; font-weight: 700; font-size: 14px; text-align:center; margin: 4px 0 14px 0; }
.triage.urgent { background: #FDECEA; color: var(--red); border: 1px solid #F5B7B1; }
.triage.review { background: #FEF5E7; color: var(--amber); border: 1px solid #F8C471; }
.triage.routine { background: #E9F7EF; color: var(--green); border: 1px solid #A9DFBF; }
.small { font-size: 12.5px; color: var(--muted); }
.reco { padding-left: 18px; margin: 0; } .reco li { margin-bottom: 5px; font-size: 14px; }

.viewer-caption { color: #6B7684; font-size: 12px; text-align:center; margin-top: -4px; }
[data-testid="stImage"] img { border-radius: 10px; }

.kpi { background: var(--card); border:1px solid var(--line); border-radius: 12px; padding: 16px; text-align:center; }
.kpi b { display:block; font-size: 30px; color: var(--navy); }
.kpi span { font-size: 13px; color: var(--muted); }
.kpi.accent b { color: var(--teal); }

.wl-item { background:#fff; border:1px solid var(--line); border-left: 4px solid var(--muted); border-radius: 8px;
           padding: 8px 10px; margin-bottom: 8px; font-size: 13px; color:#1F2933; }
.wl-item.urgent { border-left-color: var(--red); } .wl-item.review { border-left-color: var(--amber); }
.wl-item.routine { border-left-color: var(--green); }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #E3E8EE; border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #fff !important; color: var(--teal) !important; }
section[data-testid="stSidebar"] { background: #EAEFF4; top: 56px !important; height: calc(100vh - 56px) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="topbar">
  <div class="brand">🫁 Pneumo<span class="hl">Scan</span> AI <span class="sub">Chest X-ray decision support · Radiology workstation</span></div>
  <div class="right"><span class="pill">{MODEL_VERSION}</span><span>{datetime.now():%d %b %Y · %H:%M}</span><span>COMP 463 demo</span></div>
</div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model
@st.cache_resource
def get_model():
    model, img_size, thr = load_model(str(CKPT))
    return model, img_size, thr, GradCAM(model, model.layer4)


@st.cache_data
def load_metrics():
    m = json.loads((RES / "metrics.json").read_text()) if (RES / "metrics.json").exists() else None
    b = json.loads((RES / "baseline_metrics.json").read_text()) if (RES / "baseline_metrics.json").exists() else None
    return m, b


if not CKPT.exists():
    st.error(f"Model file not found: {CKPT}. Train it first with `python src/train.py`.")
    st.stop()

model, img_size, default_thr, cam = get_model()
metrics, baseline = load_metrics()

if "worklist" not in st.session_state:
    st.session_state.worklist = []
if "pid" not in st.session_state:
    st.session_state.pid = f"ANON-{random.randint(10000, 99999)}"

COLORS = {"urgent": "#C0392B", "review": "#D68910", "routine": "#1E8449"}


def triage(p, thr):
    if p >= max(0.8, thr):
        return "urgent", "🔴 HIGH SUSPICION · prioritise for review"
    if p >= thr:
        return "review", "🟠 POSSIBLE PNEUMONIA · review recommended"
    return "routine", "🟢 NO PNEUMONIA DETECTED · routine reading"


def recommendations(level):
    return {
        "urgent": ["Move study to the top of the reading worklist",
                   "Correlate with symptoms, SpO₂, CRP / WBC",
                   "Start treatment per guidelines once confirmed"],
        "review": ["Radiologist review within normal turnaround",
                   "Correlate with clinical findings and labs",
                   "Consider follow-up imaging if symptoms persist"],
        "routine": ["Routine reading, no AI flag raised",
                    "A negative AI result does not exclude pneumonia",
                    "Re-evaluate clinically if symptoms worsen"],
    }[level]


def make_report_text(info, p, thr, level):
    impression = {
        "urgent": "Findings highly suggestive of pneumonia (AI). Prompt radiologist confirmation advised.",
        "review": "Findings possibly consistent with pneumonia (AI). Radiologist confirmation advised.",
        "routine": "No radiographic evidence of pneumonia detected by the AI model.",
    }[level]
    return (
        f"EXAMINATION: Chest radiograph, frontal view\n"
        f"CLINICAL INDICATION: {info['indication']}\n"
        f"TECHNIQUE: Digital radiograph analysed by {MODEL_VERSION} (input 128x128, Grad-CAM explanation)\n\n"
        f"AI FINDINGS: Pneumonia probability {p*100:.1f}% (decision threshold {thr:.2f}). "
        f"Grad-CAM highlights the regions that contributed most to the score.\n\n"
        f"IMPRESSION: {impression}\n\n"
        f"PHYSICIAN COMMENT: "
    )


def make_pdf(info, gray, ov, report_text, p, level):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    ax = fig.add_axes([0, 0.93, 1, 0.07]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0F2A44", transform=ax.transAxes))
    ax.text(0.04, 0.5, "PneumoScan AI  ·  Preliminary Chest X-ray Report", color="white", fontsize=15, weight="bold", va="center")
    ax.text(0.96, 0.5, datetime.now().strftime("%d.%m.%Y %H:%M"), color="white", fontsize=10, va="center", ha="right")
    t = fig.add_axes([0.06, 0.80, 0.88, 0.11]); t.axis("off")
    rows = [("Patient ID", info["pid"]), ("Age / Sex", f"{info['age']} y / {info['sex']}"),
            ("Referring physician", info["ref"] or "-"), ("Indication", info["indication"])]
    for i, (k, v) in enumerate(rows):
        t.text(0, 1 - i * 0.26, k, fontsize=10, color="#6B7684", va="top")
        t.text(0.28, 1 - i * 0.26, v, fontsize=10.5, weight="bold", va="top")
    t.text(1, 1, f"{p*100:.1f}%", fontsize=28, weight="bold", color=COLORS[level], ha="right", va="top")
    t.text(1, 0.45, "pneumonia probability", fontsize=9, color="#6B7684", ha="right", va="top")
    a1 = fig.add_axes([0.06, 0.50, 0.42, 0.28]); a1.imshow(gray, cmap="gray"); a1.axis("off"); a1.set_title("Original", fontsize=10)
    a2 = fig.add_axes([0.52, 0.50, 0.42, 0.28]); a2.imshow(ov); a2.axis("off"); a2.set_title("Grad-CAM", fontsize=10)
    r = fig.add_axes([0.06, 0.10, 0.88, 0.37]); r.axis("off")
    wrapped = "\n".join(textwrap.fill(line, 95) if line else "" for line in report_text.split("\n"))
    r.text(0, 1, wrapped, fontsize=9.5, va="top", linespacing=1.5)
    f = fig.add_axes([0.06, 0.02, 0.88, 0.06]); f.axis("off")
    f.text(0, 0.5, "Generated by an academic prototype (COMP 463). Not a medical device.\n"
                   "The final diagnosis must be made by a qualified physician.", fontsize=8, color="#6B7684", va="center")
    buf = io.BytesIO(); fig.savefig(buf, format="pdf"); plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    thr = st.slider("Decision threshold (τ)", 0.05, 0.95, float(default_thr), 0.01,
                    help="Lower τ = higher sensitivity (fewer missed patients) but more false alarms. "
                         f"Default {default_thr:.2f} was selected on the validation set (Youden's J).")
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)
    st.markdown("---")
    st.markdown("### 📋 Session worklist")
    wl_box = st.container()
    st.markdown("---")
    st.caption("🔒 Images are processed locally in memory and are never stored or uploaded.")

tab_dx, tab_perf, tab_about = st.tabs(["🩻  New study", "📊  Model performance", "ℹ️  About the project"])

# ------------------------------------------------------------------ new study
with tab_dx:
    left, center, right = st.columns([1.05, 1.9, 1.15], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("<div class='sec'>Patient & study</div>", unsafe_allow_html=True)
            pid = st.text_input("Patient ID (anonymised)", st.session_state.pid)
            c_a, c_s = st.columns(2)
            age = c_a.number_input("Age (years)", 0, 120, 4)
            sex = c_s.selectbox("Sex", ["F", "M"])
            indication = st.selectbox("Clinical indication",
                                      ["Fever and cough", "Shortness of breath", "Suspected pneumonia",
                                       "Follow-up of known pneumonia", "Pre-operative screening", "Other"])
            ref = st.text_input("Referring physician", "")
        with st.container(border=True):
            st.markdown("<div class='sec'>Image</div>", unsafe_allow_html=True)
            up = st.file_uploader("Upload chest X-ray (PNG / JPG)", type=["png", "jpg", "jpeg"])
            samples = sorted(p.name for p in SAMPLES.glob("*.png")) if SAMPLES.exists() else []
            sample = st.selectbox("…or load a demo case", ["—"] + samples)

    src, fname = None, None
    if up is not None:
        src, fname = up, up.name
    elif sample != "—":
        src, fname = SAMPLES / sample, sample

    if src is None:
        with center:
            st.markdown("""<div class='card' style='text-align:center;padding:80px 30px'>
              <div style='font-size:56px'>🩻</div>
              <h3 style='color:#0F2A44;margin:8px 0'>No study loaded</h3>
              <p class='small' style='font-size:14px'>Upload a frontal chest X-ray or pick a demo case on the left.<br>
              The AI estimates the probability of pneumonia, assigns a triage level and shows
              where in the image the evidence comes from.</p></div>""", unsafe_allow_html=True)
        with right:
            st.markdown("""<div class='card'><h4>How it works</h4><ol class='reco'>
              <li>Image is resized to 128×128 and normalised</li><li>ResNet18 CNN outputs P(pneumonia)</li>
              <li>Probability is compared with threshold τ</li><li>Grad-CAM shows the supporting regions</li>
              <li>Physician reviews, edits and approves the report</li></ol></div>""", unsafe_allow_html=True)
    else:
        img = Image.open(src).convert("L")
        x = build_transforms(img_size, train=False)(img).unsqueeze(0).requires_grad_(True)
        heat, probs = cam(x, class_idx=1)
        p = float(probs[1])
        level, label = triage(p, thr)
        gray = np.array(img.resize((img_size, img_size)))
        gray_big = np.array(img.resize((512, 512)))
        ov_big = np.array(Image.fromarray(overlay(gray, heat, alpha)).resize((512, 512), Image.BICUBIC))

        wl = st.session_state.worklist
        if not wl or (wl[-1]["pid"], wl[-1]["file"]) != (pid, fname):
            wl.append({"pid": pid, "file": fname, "p": p, "level": level, "time": datetime.now().strftime("%H:%M")})
        else:
            wl[-1].update(p=p, level=level)

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
                st.caption(f"File: {fname} · red = regions that raised the pneumonia score, blue = little contribution")

        with right:
            auc = f"{metrics['auc']:.3f}" if metrics else "–"
            st.markdown(f"""<div class='card'><h4>AI findings</h4>
              <div class='gauge' style='background: conic-gradient({COLORS[level]} {p*360:.1f}deg, #E8ECF1 0deg)'>
                <div class='val'><b>{p*100:.1f}%</b><small>pneumonia probability</small></div></div>
              <div class='triage {level}'>{label}</div>
              <div class='kv'><div class='k'>Threshold τ</div><div class='v'>{thr:.2f}</div>
                <div class='k'>Model</div><div class='v'>{MODEL_VERSION}</div>
                <div class='k'>Test AUC</div><div class='v'>{auc}</div></div></div>""", unsafe_allow_html=True)
            st.markdown("<div class='card'><h4>Suggested next steps</h4><ul class='reco'>" +
                        "".join(f"<li>{r}</li>" for r in recommendations(level)) + "</ul></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<div class='sec'>Preliminary report · editable by the physician</div>", unsafe_allow_html=True)
            info = {"pid": pid, "age": age, "sex": sex, "indication": indication, "ref": ref}
            report = st.text_area("Report", make_report_text(info, p, thr, level), height=220,
                                  label_visibility="collapsed", key=f"report_{pid}_{fname}_{level}_{indication}")
            c1, c2, _ = st.columns([1.3, 1.3, 3])
            approved = c1.checkbox("I have reviewed this result")
            c2.download_button("⬇️  Download PDF report", data=make_pdf(info, gray_big, ov_big, report, p, level),
                               file_name=f"{pid}_chest_xray_report.pdf", mime="application/pdf",
                               disabled=not approved, type="primary")
            st.caption("⚠️ Decision support only. The AI result must be confirmed by a qualified physician.")

with wl_box:
    if not st.session_state.worklist:
        st.caption("Analysed studies will appear here.")
    for w in reversed(st.session_state.worklist[-8:]):
        st.markdown(f"<div class='wl-item {w['level']}'><b>{w['pid']}</b> · {w['time']}<br>"
                    f"{w['file']} · <b>{w['p']*100:.0f}%</b></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ performance
with tab_perf:
    if not metrics:
        st.info("No results yet. Run `python src/evaluate.py --data <data>` to create them.")
    else:
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

# ------------------------------------------------------------------ about
with tab_about:
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        st.markdown("""<div class='card'><h4>Purpose</h4>
        <p>Pneumonia is the leading infectious cause of death in children under five worldwide (WHO).
        Chest X-ray is the first-line test, but reading it requires a radiologist, who is not always available,
        and studies can wait hours in the queue.</p>
        <p><b>PneumoScan AI</b> is a decision-support prototype that <b>screens chest X-rays for pneumonia,
        prioritises suspicious studies on the worklist and explains each result with a heatmap</b>, so the
        physician can decide faster and with an extra safety net. It does not replace the physician.</p></div>""",
                    unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div class='sec'>System pipeline</div>", unsafe_allow_html=True)
            if (RES / "architecture.png").exists():
                st.image(str(RES / "architecture.png"), width="stretch")
    with b:
        st.markdown("""<div class='card'><h4>Intended use</h4><ul class='reco'>
          <li>Users: emergency physicians, paediatricians, radiologists</li>
          <li>Input: frontal paediatric chest X-ray</li>
          <li>Output: probability, triage level, heatmap, draft report</li>
          <li>Role: triage and second opinion, never autonomous diagnosis</li></ul></div>
        <div class='card'><h4>Model & data</h4><ul class='reco'>
          <li>ResNet18 CNN trained from scratch (PyTorch)</li>
          <li>PneumoniaMNIST: 5,856 paediatric X-rays (Kermany et al., 2018)</li>
          <li>Explainability: Grad-CAM (Selvaraju et al., 2017)</li></ul></div>
        <div class='card'><h4>Limitations</h4><ul class='reco'>
          <li>Single-centre paediatric data; external validation needed</li>
          <li>Low resolution (128×128); normal vs pneumonia only</li>
          <li>Not a certified medical device (EU MDR / SaMD)</li></ul></div>
        <div class='card'><h4>Project</h4><div class='kv'>
          <div class='k'>Course</div><div class='v'>COMP 463 · Intro to Medical Informatics</div>
          <div class='k'>Student</div><div class='v'>Elif Erkişi · 042401169</div>
          <div class='k'>Instructor</div><div class='v'>Burak Arslan</div>
          <div class='k'>University</div><div class='v'>MEF University</div></div></div>""", unsafe_allow_html=True)
