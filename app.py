"""
app.py — EE200 Audio Fingerprinting (Streamlit)

Two modes:
  • Identify (single clip): shows spectrogram, constellation of peaks,
    offset histogram, the pipeline timings, and the match.
  • Batch: accepts many clips, writes results.csv with columns
    exactly  filename,prediction  (prediction = matched song's
    filename without extension).

No librosa — decodes via soundfile + scipy so it builds on
Streamlit Cloud (Python 3.14) without compiling numba/llvmlite.
"""
import io
import os
import gc
import gzip
import time
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from fingerprint import (
    SR, MAX_QUERY_SECONDS, smart_load, get_peaks, hashes_from_peaks, identify, compute_spectrogram,
)

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="EE200 · Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
)

DB_PATH = "fingerprint_db.pkl.gz"
SAMPLES_DIR = "samples"

# palette (matches the dark / teal mock)
BG       = "#0a0e0f"
PANEL    = "#11171a"
TEAL     = "#5eead4"
TEAL_DIM = "#2dd4bf"
INK      = "#e7eceb"
MUTE     = "#7c8a87"
GRID     = "#1c2629"
RED      = "#f87171"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp {{ background: {BG}; }}
section.main > div {{ padding-top: 1.2rem; }}

h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; color: {INK}; letter-spacing: -0.01em; }}
.stApp, p, label, span, div {{ color: {INK}; }}

/* eyebrow / mono labels */
.eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.28em;
            text-transform:uppercase; color:{MUTE}; margin-bottom:.2rem; }}

.brandwrap {{ display:flex; align-items:center; gap:.85rem; }}
.brandmark {{ width:46px; height:46px; border:1px solid {GRID}; border-radius:12px;
              display:flex; align-items:center; justify-content:center; background:{PANEL}; }}
.brandmark span {{ color:{TEAL}; font-size:1.5rem; }}
.title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.4rem; line-height:1;
          margin:0; color:{INK}; }}
.title b {{ color:{TEAL}; }}
.subtitle {{ color:{MUTE}; margin:.35rem 0 0; font-size:1.02rem; }}

/* metric pills (pipeline timings) */
.pipe {{ display:flex; gap:.6rem; flex-wrap:wrap; margin-top:.4rem; }}
.pill {{ flex:1; min-width:120px; background:{PANEL}; border:1px solid {GRID};
         border-radius:14px; padding:.8rem .9rem; }}
.pill .step {{ font-family:'JetBrains Mono',monospace; font-size:.62rem; letter-spacing:.18em;
               color:{MUTE}; text-transform:uppercase; }}
.pill .val  {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.45rem;
               color:{TEAL}; margin-top:.15rem; }}
.pill .sub  {{ font-family:'JetBrains Mono',monospace; font-size:.66rem; color:{MUTE}; margin-top:.1rem; }}

/* match banner */
.match {{ background:linear-gradient(135deg,#0f1d1b,{PANEL}); border:1px solid {TEAL_DIM}33;
          border-radius:18px; padding:1.4rem 1.6rem; margin-top:.6rem; }}
.match .lab {{ font-family:'JetBrains Mono',monospace; letter-spacing:.22em; font-size:.7rem;
               color:{TEAL}; text-transform:uppercase; }}
.match .song {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2rem;
                color:{INK}; margin:.2rem 0 0; }}
.nomatch {{ border-color:{RED}33; }}
.nomatch .lab {{ color:{RED}; }}

/* buttons */
.stButton>button, .stDownloadButton>button {{
    background:{TEAL}; color:#062420; border:0; border-radius:12px;
    font-family:'Space Grotesk',sans-serif; font-weight:700; padding:.55rem 1.4rem;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:{TEAL_DIM}; color:#041815; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:1.6rem; border-bottom:1px solid {GRID}; }}
.stTabs [data-baseweb="tab"] {{ font-family:'JetBrains Mono',monospace; letter-spacing:.14em;
    text-transform:uppercase; font-size:.78rem; color:{MUTE}; background:transparent; }}
.stTabs [aria-selected="true"] {{ color:{TEAL}; }}

hr {{ border-color:{GRID}; }}
[data-testid="stFileUploaderDropzone"] {{ background:{PANEL}; border:1px dashed {GRID}; border-radius:14px; }}
.smallcap {{ font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.22em;
             text-transform:uppercase; color:{MUTE}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# teal-on-dark colormap for spectrograms
TEAL_CMAP = LinearSegmentedColormap.from_list(
    "tealdark", ["#05080a", "#0b2b2a", "#10645c", "#2dd4bf", "#bff7ec"]
)


# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_db(path=DB_PATH):
    # find the database: prefer the configured path, fall back to the
    # other extension so a .pkl or .pkl.gz both just work.
    candidates = [path, "fingerprint_db.pkl.gz", "fingerprint_db.pkl"]
    real = next((p for p in candidates if os.path.exists(p)), None)
    if real is None:
        raise FileNotFoundError(path)
    opener = gzip.open if real.endswith(".gz") else open
    with opener(real, "rb") as fp:
        payload = pickle.load(fp)
    # support both the new payload dict and a bare dict (old notebook .pkl)
    if isinstance(payload, dict) and "db" in payload and "labels" in payload:
        return payload["db"], payload["labels"]
    db = payload
    labels = sorted({lab for v in db.values() for _, lab in v})
    return db, labels


def list_samples():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    return sorted(
        os.path.join(SAMPLES_DIR, f)
        for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith(exts)
    )


# ----------------------------------------------------------------------
# Plot helpers (styled to match the dark theme)
# ----------------------------------------------------------------------
def _style_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.xaxis.label.set_color(MUTE)
    ax.yaxis.label.set_color(MUTE)
    ax.title.set_color(INK)


def plot_spectrogram(f, t, Sdb):
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=120)
    fig.patch.set_facecolor(BG)
    m = ax.pcolormesh(t, f, Sdb, shading="gouraud", cmap=TEAL_CMAP, vmin=-70, vmax=-10)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram", fontsize=11, loc="left")
    cb = fig.colorbar(m, ax=ax, pad=0.01)
    cb.ax.tick_params(colors=MUTE, labelsize=7); cb.outline.set_edgecolor(GRID)
    cb.set_label("magnitude (dB)", color=MUTE, fontsize=8)
    _style_ax(ax); fig.tight_layout()
    return fig


def plot_constellation(f, t, Sdb, peaks):
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.pcolormesh(t, f, Sdb, shading="gouraud", cmap=TEAL_CMAP, vmin=-70, vmax=-10, alpha=0.55)
    if peaks:
        ti = [p[0] for p in peaks]; fi = [p[1] for p in peaks]
        ax.scatter(t[ti], f[fi], s=16, facecolors="none",
                   edgecolors=TEAL, linewidths=0.9)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Constellation · {len(peaks)} peaks", fontsize=11, loc="left")
    _style_ax(ax); fig.tight_layout()
    return fig


def plot_offset_hist(offsets, top_label, runner_label=None):
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=120)
    fig.patch.set_facecolor(BG)
    top = offsets.get(top_label, [])
    pool = list(top)
    if runner_label:
        pool += offsets.get(runner_label, [])
    if pool:
        lo, hi = min(pool), max(pool)
        bins = np.linspace(lo, hi, 80) if hi > lo else 30
        if runner_label and offsets.get(runner_label):
            ax.hist(offsets[runner_label], bins=bins, color=RED, alpha=0.65,
                    label=f"{runner_label[:22]} (wrong)")
        ax.hist(top, bins=bins, color=TEAL, alpha=0.9,
                label=f"{top_label[:22]} (match)")
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.set_xlabel("time offset (bins)"); ax.set_ylabel("matching hashes")
    ax.set_title("Offset histogram — the vote that decides the match",
                 fontsize=11, loc="left")
    _style_ax(ax); fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="brandwrap">
      <div class="brandmark"><span>♪</span></div>
      <div>
        <p class="title">EE<b>200</b>: Audio Fingerprinting</p>
        <div class="eyebrow" style="margin-top:.45rem">Signals, Systems &amp; Networks · Project Demo</div>
      </div>
    </div>
    <p class="subtitle">Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</p>
    """,
    unsafe_allow_html=True,
)

# guard: database present?
if not (os.path.exists("fingerprint_db.pkl.gz") or os.path.exists("fingerprint_db.pkl")):
    st.error(
        "**Database not found.** Expected `fingerprint_db.pkl.gz` (or "
        "`fingerprint_db.pkl`) next to app.py. Build it once with "
        "`python build_db.py /path/to/songs`."
    )
    st.stop()

db, LABELS = load_db()
st.markdown(
    f"<div class='smallcap'>library indexed · {len(LABELS)} songs · "
    f"{len(db):,} distinct hashes</div>",
    unsafe_allow_html=True,
)
st.write("")

tab_single, tab_batch, tab_lib = st.tabs(["Identify", "Batch", "Library"])


# ======================================================================
# SINGLE-CLIP MODE
# ======================================================================
def run_single(y):
    """Run the full pipeline, time each stage, render everything."""
    timings = {}

    t0 = time.perf_counter()
    f, t, Sdb = compute_spectrogram(y)
    timings["spectrogram"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    f2, t2, Sdb2, peaks = get_peaks(y)
    timings["constellation"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    qhashes = hashes_from_peaks(peaks)
    timings["hashing"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ranked, offsets, best_off = identify(y, db)
    timings["lookup"] = (time.perf_counter() - t0) * 1000
    timings["scoring"] = 0.0  # folded into lookup; kept for the pill layout

    total = sum(timings.values())

    top = ranked[0] if ranked else None
    runner = ranked[1] if len(ranked) > 1 else None

    # confidence: top score vs runner-up
    decisive = (top[1] / runner[1]) if (top and runner and runner[1]) else float("inf")
    is_match = bool(top) and top[1] >= 5  # tiny floor against pure noise

    # ---- pipeline pills ----
    pills = [
        ("① Spectrogram", f"{timings['spectrogram']:.0f} ms", f"{Sdb.shape[0]}×{Sdb.shape[1]}"),
        ("② Constellation", f"{timings['constellation']:.0f} ms", f"{len(peaks)} peaks"),
        ("③ Hashing", f"{timings['hashing']:.0f} ms", f"{len(qhashes):,} hashes"),
        ("④ DB lookup", f"{timings['lookup']:.0f} ms", f"{len(LABELS)} tracks"),
        ("⑤ Scoring", f"{timings['scoring']:.0f} ms", f"offset {best_off if best_off is not None else '—'}"),
    ]
    html = "<div class='pipe'>"
    for step, val, sub in pills:
        html += (f"<div class='pill'><div class='step'>{step}</div>"
                 f"<div class='val'>{val}</div><div class='sub'>{sub}</div></div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(f"<div class='smallcap' style='text-align:right;margin-top:.4rem'>"
                f"total {total:.0f} ms</div>", unsafe_allow_html=True)

    # ---- match banner ----
    if is_match:
        st.markdown(
            f"<div class='match'><div class='lab'>Match found</div>"
            f"<div class='song'>{top[0]}</div>"
            f"<div class='smallcap' style='margin-top:.5rem'>"
            f"score {top[1]} · decisiveness {decisive:.1f}× over runner-up"
            f"{(' · ' + runner[0]) if runner else ''}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='match nomatch'><div class='lab'>No confident match</div>"
            "<div class='song'>—</div>"
            "<div class='smallcap' style='margin-top:.5rem'>"
            "too few aligned hashes; try a longer or cleaner clip</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")

    # ---- intermediate visuals (required) ----
    c1, c2 = st.columns(2)
    fig_spec = plot_spectrogram(f, t, Sdb)
    fig_const = plot_constellation(f2, t2, Sdb2, peaks)
    fig_hist = plot_offset_hist(
        offsets, top[0] if top else "", runner[0] if runner else None
    )
    with c1:
        st.pyplot(fig_spec, use_container_width=True)
    with c2:
        st.pyplot(fig_const, use_container_width=True)
    st.pyplot(fig_hist, use_container_width=True)

    # ---- ranked table ----
    if ranked:
        with st.expander("Full ranking"):
            df = pd.DataFrame(ranked[:10], columns=["song", "score"])
            df.index = np.arange(1, len(df) + 1)
            st.dataframe(df, use_container_width=True)

    # ---- free everything (figures + arrays) and flush RAM ----
    for _fig in (fig_spec, fig_const, fig_hist):
        plt.close(_fig)
    del Sdb, Sdb2, offsets
    gc.collect()


with tab_single:
    st.markdown("<div class='eyebrow'>Search</div>", unsafe_allow_html=True)
    st.markdown("### Identify a clip")

    up = st.file_uploader(
        f"Upload a query clip — WAV, MP3, FLAC, OGG, M4A · only the first "
        f"{MAX_QUERY_SECONDS} s are analysed",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="single_up",
    )

    samples = list_samples()
    chosen_sample = None
    if samples:
        st.markdown("<div class='eyebrow' style='margin-top:.8rem'>Or try a sample</div>",
                    unsafe_allow_html=True)
        cols = st.columns(min(len(samples), 5))
        for i, sp in enumerate(samples):
            with cols[i % len(cols)]:
                st.audio(sp)
                if st.button(f"Try {os.path.splitext(os.path.basename(sp))[0]}",
                             key=f"samp_{i}"):
                    chosen_sample = sp

    st.write("")
    go = st.button("Identify", type="primary", key="single_go")

    if chosen_sample or (go and up):
        try:
            with st.spinner("Fingerprinting and matching…"):
                if chosen_sample:
                    y = smart_load(chosen_sample, filename=chosen_sample)
                else:
                    y = smart_load(up, filename=up.name)
            if len(y) < SR // 2:
                st.warning("Clip is shorter than ~0.5 s — results may be unreliable.")
            run_single(y)
        except Exception as e:
            st.error(f"Could not process that file: {e}")
    elif go and not up:
        st.info("Upload a clip first, or pick a sample.")


# ======================================================================
# BATCH MODE
# ======================================================================
with tab_batch:
    st.markdown("<div class='eyebrow'>Evaluation</div>", unsafe_allow_html=True)
    st.markdown("### Batch identify → results.csv")
    st.caption(
        "Upload multiple query clips. Output is a CSV with exactly two columns — "
        "`filename,prediction` — where prediction is the matched song's filename "
        "without extension."
    )

    ups = st.file_uploader(
        "Upload query clips",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        key="batch_up",
    )

    if st.button("Run batch", type="primary", key="batch_go"):
        if not ups:
            st.info("Upload one or more clips to run a batch.")
        else:
            rows = []
            prog = st.progress(0.0, text="Processing…")
            for i, fobj in enumerate(ups, 1):
                fname = fobj.name
                y = None
                try:
                    y = smart_load(fobj, filename=fname)   # capped to 15 s
                    ranked, _, _ = identify(y, db)
                    pred = ranked[0][0] if ranked else ""
                except Exception:
                    pred = ""
                finally:
                    # flush RAM after every file so batches stay light
                    del y
                    gc.collect()
                rows.append({"filename": fname, "prediction": pred})
                prog.progress(i / len(ups), text=f"Processed {i}/{len(ups)}")
            prog.empty()

            df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(df, use_container_width=True)

            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)        # -> exactly: filename,prediction
            st.download_button(
                "Download results.csv",
                data=csv_buf.getvalue(),
                file_name="results.csv",
                mime="text/csv",
                type="primary",
            )
            st.success(f"Done — {len(df)} clips. CSV columns: filename, prediction.")


# ======================================================================
# LIBRARY
# ======================================================================
with tab_lib:
    st.markdown("<div class='eyebrow'>Index</div>", unsafe_allow_html=True)
    st.markdown(f"### {len(LABELS)} songs in the database")
    st.caption("These labels are exactly what the identifier outputs (filename without extension).")
    libdf = pd.DataFrame({"song": LABELS})
    libdf.index = np.arange(1, len(libdf) + 1)
    st.dataframe(libdf, use_container_width=True, height=460)
