"""
app.py — EE200: Sonic Signatures (Streamlit)

Two modes:
  • Identify (single clip): shows the spectrogram, the constellation of
    peaks, the offset histogram that decides the match, the pipeline
    timings, and the predicted song — with an inline player.
  • Batch: accepts many clips and writes results.csv with columns
    exactly  filename,prediction  (prediction = matched song's
    filename without extension).

No librosa — decodes via soundfile + scipy so it builds on
Streamlit Cloud without compiling numba/llvmlite.
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
    SR, get_peaks, hashes_from_peaks, identify, compute_spectrogram,
)
try:
    from fingerprint import smart_load
except Exception:
    from fingerprint import load_audio as smart_load  # older fingerprint.py
try:
    from fingerprint import MAX_QUERY_SECONDS
except Exception:
    MAX_QUERY_SECONDS = 30   # fallback if an older fingerprint.py lacks it


# ----------------------------------------------------------------------
# Confidence gate (kept here in app.py so it has no extra import deps)
# ----------------------------------------------------------------------
# A genuine match dominates: a large aligned-hash score AND a large lead over
# the runner-up. A wrong / out-of-library clip yields only scattered
# coincidental collisions — a tiny score that barely beats the next song.
# Measured separation on the 50-song library: true matches score in the
# thousands with ~1000x+ leads; false queries top out at score 3, ratio 1.5x.
MIN_SCORE = 15      # minimum absolute aligned-hash count
MIN_RATIO = 2.5     # minimum lead over the runner-up


def is_confident(ranked):
    """Return (is_match, top_label_or_None, score, ratio) for a ranked list."""
    if not ranked:
        return False, None, 0, 0.0
    top_label, top_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    ratio = (top_score / runner) if runner else float("inf")
    ok = top_score >= MIN_SCORE and ratio >= MIN_RATIO
    return ok, top_label, top_score, ratio

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sonic Signatures · EE200",
    page_icon="◑",
    layout="wide",
)

DB_PATH = "fingerprint_db.pkl.gz"
SAMPLES_DIR = "samples"

# ---- palette: warm amber signature on deep plum, cream plots ----
BG        = "#17121f"   # deep plum-charcoal
PANEL     = "#1f1830"   # raised card
PANEL2    = "#271e3b"   # hover / inner
AMBER     = "#f5a623"   # signature accent
AMBER_DK  = "#d98a12"
VIOLET    = "#a78bfa"   # secondary accent
INK       = "#f3eee6"   # warm off-white text
MUTE      = "#9a8fb0"   # muted lavender-grey
LINE      = "#352a4a"   # hairlines
CREAM     = "#f5efe4"   # plot background (LIGHT — readable)
PLOTINK   = "#2a2233"   # plot foreground ink
ROSE      = "#e0607e"   # "wrong song" colour

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp {{ background:
   radial-gradient(1200px 500px at 12% -8%, #241a35 0%, transparent 60%),
   radial-gradient(900px 600px at 110% 0%, #2a1d2e 0%, transparent 55%),
   {BG}; }}
section.main > div {{ padding-top: 1rem; }}

/* fonts */
html, body, [class*="css"] {{ font-family:'IBM Plex Sans', sans-serif; }}
h1,h2,h3 {{ font-family:'Fraunces', serif !important; color:{INK}; }}
p, label, span, div, li {{ color:{INK}; }}

.mono {{ font-family:'IBM Plex Mono', monospace; }}
.kicker {{ font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.34em;
           text-transform:uppercase; color:{AMBER}; }}
.dim {{ color:{MUTE}; }}

/* ---------- header ---------- */
.masthead {{ display:flex; align-items:center; gap:1.2rem; padding:.5rem 0 .1rem; }}
.dial {{ width:64px; height:64px; border-radius:50%; flex:none;
         background:
           conic-gradient(from 220deg, {AMBER} 0deg, {VIOLET} 130deg, {AMBER} 300deg);
         padding:2px; box-shadow:0 0 30px #f5a62333; }}
.dial > div {{ width:100%; height:100%; border-radius:50%; background:{BG};
               display:flex; align-items:center; justify-content:center;
               font-size:1.6rem; color:{AMBER}; }}
.h-title {{ font-family:'Fraunces',serif; font-weight:700; font-size:3.1rem; line-height:.92;
            margin:.1rem 0 0; color:{INK}; letter-spacing:-.015em; }}
.h-title em {{ font-style:italic; color:{AMBER}; }}
.h-lede {{ margin:1rem 0 0; color:{INK}; font-size:1.18rem; line-height:1.5;
           max-width:64ch; font-weight:400; }}
.h-lede b {{ color:{AMBER}; font-weight:600; }}

/* how-it-works step chips */
.howto {{ display:flex; gap:.7rem; flex-wrap:wrap; margin:1.1rem 0 0; }}
.chip {{ flex:1; min-width:180px; background:{PANEL}; border:1px solid {LINE};
         border-radius:13px; padding:.7rem .9rem; display:flex; gap:.7rem; align-items:flex-start; }}
.chip .num {{ font-family:'Fraunces',serif; font-style:italic; font-weight:600;
              font-size:1.3rem; color:{AMBER}; line-height:1; flex:none; }}
.chip .txt b {{ display:block; font-family:'IBM Plex Sans',sans-serif; font-weight:600;
                font-size:.86rem; color:{INK}; }}
.chip .txt span {{ font-size:.78rem; color:{MUTE}; line-height:1.35; }}

.statline {{ display:flex; gap:1.6rem; margin:1.1rem 0 0; padding:.6rem 0 0;
             border-top:1px solid {LINE}; flex-wrap:wrap; align-items:baseline; }}
.stat b {{ font-family:'Fraunces',serif; font-weight:700; color:{AMBER}; font-size:1.25rem; }}
.stat span {{ font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.2em;
              text-transform:uppercase; color:{MUTE}; margin-left:.4rem; }}

/* ---------- generic card ---------- */
.card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:18px;
         padding:1.15rem 1.3rem; }}

/* ---------- pipeline strip ---------- */
.flow {{ display:flex; gap:.55rem; flex-wrap:wrap; }}
.node {{ flex:1; min-width:118px; background:{PANEL}; border:1px solid {LINE};
         border-radius:14px; padding:.75rem .85rem; position:relative; }}
.node .n {{ font-family:'IBM Plex Mono',monospace; font-size:.6rem; letter-spacing:.16em;
            text-transform:uppercase; color:{VIOLET}; }}
.node .v {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.5rem; color:{INK};
            margin-top:.1rem; line-height:1; }}
.node .s {{ font-family:'IBM Plex Mono',monospace; font-size:.64rem; color:{MUTE}; margin-top:.25rem; }}

/* ---------- verdict ---------- */
.verdict {{ background:linear-gradient(100deg,#221a17, {PANEL} 60%);
            border:1px solid {AMBER}33; border-left:4px solid {AMBER};
            border-radius:16px; padding:1.3rem 1.5rem; }}
.verdict .tag {{ font-family:'IBM Plex Mono',monospace; letter-spacing:.26em; font-size:.66rem;
                 text-transform:uppercase; color:{AMBER}; }}
.verdict .name {{ font-family:'Fraunces',serif; font-weight:700; font-size:2.1rem;
                  color:{INK}; margin:.25rem 0 0; line-height:1.05; }}
.verdict .meta {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:{MUTE}; margin-top:.5rem; }}
.verdict.miss {{ border-color:{ROSE}33; border-left-color:{ROSE}; background:linear-gradient(100deg,#241519,{PANEL} 60%); }}
.verdict.miss .tag {{ color:{ROSE}; }}

/* ---------- step panel (narrative block before each plot) ---------- */
.step {{ border-left:3px solid {VIOLET}; padding:.1rem 0 .1rem 1rem; margin:.2rem 0 .9rem; }}
.step.amber {{ border-left-color:{AMBER}; }}
.step .ey {{ font-family:'IBM Plex Mono',monospace; font-size:.66rem; letter-spacing:.26em;
             text-transform:uppercase; color:{VIOLET}; }}
.step.amber .ey {{ color:{AMBER}; }}
.step h4 {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.5rem; color:{INK};
            margin:.2rem 0 .35rem; }}
.step p {{ color:{MUTE}; font-size:.97rem; line-height:1.5; margin:0; max-width:78ch; }}
.step p b {{ color:{INK}; font-weight:600; }}
.step.amber p b {{ color:{AMBER}; }}

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button {{
    background:{AMBER}; color:#2a1c05; border:0; border-radius:11px;
    font-family:'IBM Plex Sans',sans-serif; font-weight:600; padding:.55rem 1.5rem;
    transition:transform .05s ease, background .15s ease; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:{AMBER_DK}; color:#2a1c05; }}
.stButton>button:active {{ transform:translateY(1px); }}

/* secondary (sample 'Try') buttons get an outline look */
div[data-testid="column"] .stButton>button {{
    background:transparent; color:{INK}; border:1px solid {LINE}; font-weight:500; }}
div[data-testid="column"] .stButton>button:hover {{ border-color:{AMBER}; color:{AMBER}; background:transparent; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family:'IBM Plex Mono',monospace; letter-spacing:.12em;
    text-transform:uppercase; font-size:.74rem; color:{MUTE}; padding:.5rem .9rem; }}
.stTabs [aria-selected="true"] {{ color:{AMBER}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{AMBER}; }}

/* uploader + misc */
[data-testid="stFileUploaderDropzone"] {{ background:{PANEL}; border:1px dashed {LINE}; border-radius:14px; }}
hr {{ border-color:{LINE}; }}
[data-testid="stExpander"] {{ border:1px solid {LINE}; border-radius:12px; background:{PANEL}; }}
.section {{ font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.3em;
            text-transform:uppercase; color:{VIOLET}; margin:.2rem 0 .1rem; }}
audio {{ width:100%; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# warm, readable spectrogram colormap on cream
WARM_CMAP = LinearSegmentedColormap.from_list(
    "warm", ["#f5efe4", "#e9d8a6", "#e8a33d", "#c84b2f", "#5e1f3a", "#241327"]
)


# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_db(path=DB_PATH):
    candidates = [path, "fingerprint_db.pkl.gz", "fingerprint_db.pkl"]
    real = next((p for p in candidates if os.path.exists(p)), None)
    if real is None:
        raise FileNotFoundError(path)
    opener = gzip.open if real.endswith(".gz") else open
    with opener(real, "rb") as fp:
        payload = pickle.load(fp)

    if isinstance(payload, dict) and "db" in payload and "labels" in payload:
        return payload["db"], list(payload["labels"])
    if isinstance(payload, dict) and "db" in payload and "meta" in payload:
        meta = payload["meta"]
        if isinstance(meta, dict) and meta:
            return payload["db"], sorted(map(str, meta.keys()))
        return payload["db"], _labels_from_db(payload["db"])
    if isinstance(payload, dict) and "db" in payload and not isinstance(payload.get("db"), dict):
        return payload, _labels_from_db(payload)
    return payload, _labels_from_db(payload)


def _labels_from_db(db):
    labels = set()
    for v in db.values():
        for entry in v:
            if isinstance(entry, (tuple, list)):
                if len(entry) == 2:
                    a, b = entry
                    labels.add(b if isinstance(b, str) else (a if isinstance(a, str) else b))
                elif len(entry) == 1:
                    labels.add(entry[0])
                else:
                    strs = [x for x in entry if isinstance(x, str)]
                    labels.add(strs[0] if strs else entry[-1])
            else:
                labels.add(entry)
    return sorted(map(str, labels))


def list_samples():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    return sorted(os.path.join(SAMPLES_DIR, f)
                  for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(exts))


# ----------------------------------------------------------------------
# Plots — LIGHT background so they are clearly visible
# ----------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(CREAM)
    for s in ax.spines.values():
        s.set_color("#cdbfa6")
    ax.tick_params(colors="#6b5d49", labelsize=8)
    ax.xaxis.label.set_color("#6b5d49")
    ax.yaxis.label.set_color("#6b5d49")
    ax.title.set_color(PLOTINK)


def plot_spectrogram(f, t, Sdb):
    fig, ax = plt.subplots(figsize=(7, 3.3), dpi=130)
    fig.patch.set_facecolor(CREAM)
    m = ax.pcolormesh(t, f, Sdb, shading="gouraud", cmap=WARM_CMAP, vmin=-70, vmax=-10)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram", fontsize=12, loc="left", fontweight="bold")
    cb = fig.colorbar(m, ax=ax, pad=0.01)
    cb.ax.tick_params(colors="#6b5d49", labelsize=7); cb.outline.set_edgecolor("#cdbfa6")
    cb.set_label("magnitude (dB)", color="#6b5d49", fontsize=8)
    _style(ax); fig.tight_layout()
    return fig


def plot_constellation(f, t, Sdb, peaks):
    fig, ax = plt.subplots(figsize=(7, 3.3), dpi=130)
    fig.patch.set_facecolor(CREAM)
    # faint warm spectrogram underneath, then crisp dark rings on top
    ax.pcolormesh(t, f, Sdb, shading="gouraud", cmap=WARM_CMAP,
                  vmin=-70, vmax=-10, alpha=0.30)
    if peaks:
        ti = [p[0] for p in peaks]; fi = [p[1] for p in peaks]
        ax.scatter(t[ti], f[fi], s=22, facecolors="none",
                   edgecolors=PLOTINK, linewidths=1.1)
        ax.scatter(t[ti], f[fi], s=3, color="#c84b2f")
    ax.set_ylim(0, 4000)
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Constellation · {len(peaks)} peaks", fontsize=12, loc="left", fontweight="bold")
    _style(ax); fig.tight_layout()
    return fig


def plot_offset_hist(offsets, top_label, runner_label=None):
    fig, ax = plt.subplots(figsize=(14.4, 3.4), dpi=130)
    fig.patch.set_facecolor(CREAM)
    top = offsets.get(top_label, [])
    pool = list(top)
    if runner_label:
        pool += offsets.get(runner_label, [])
    if pool:
        lo, hi = min(pool), max(pool)
        bins = np.linspace(lo, hi, 90) if hi > lo else 30
        if runner_label and offsets.get(runner_label):
            ax.hist(offsets[runner_label], bins=bins, color=ROSE, alpha=0.75,
                    label=f"{runner_label[:28]}  (other song)")
        ax.hist(top, bins=bins, color="#c84b2f",
                label=f"{top_label[:28]}  (match)")
        leg = ax.legend(fontsize=8, facecolor=CREAM, edgecolor="#cdbfa6", labelcolor=PLOTINK)
    ax.set_xlabel("time offset (bins)"); ax.set_ylabel("aligned hashes")
    ax.set_title("Offset histogram — genuine matches pile at one offset",
                 fontsize=12, loc="left", fontweight="bold")
    _style(ax); fig.tight_layout()
    return fig


def song_anchor_points(db, label, max_points=6000):
    """Reconstruct a song's stored fingerprint from the database: every hash
    anchor (time_frame, freq_of_first_peak) tagged with `label`. Returns two
    arrays (times, freqs). Sampled down to keep the scatter light."""
    times, freqs = [], []
    for h, entries in db.items():
        f1 = h[0] if isinstance(h, (tuple, list)) else None
        for entry in entries:
            # entry is (anchor_time, label)
            if len(entry) >= 2 and entry[-1] == label:
                times.append(entry[0])
                freqs.append(f1 if f1 is not None else 0)
    times = np.asarray(times); freqs = np.asarray(freqs)
    if times.size > max_points:                       # downsample for speed
        idx = np.random.default_rng(0).choice(times.size, max_points, replace=False)
        times, freqs = times[idx], freqs[idx]
    return times, freqs


def plot_song_map(db, label, best_offset, query_len_frames):
    """STEP 2 — the full stored fingerprint of the matched song, with the
    window where the query aligns highlighted."""
    fig, ax = plt.subplots(figsize=(14.4, 3.6), dpi=130)
    fig.patch.set_facecolor(CREAM)
    times, freqs = song_anchor_points(db, label)
    if times.size:
        ax.scatter(times, freqs, s=4, color="#3a2b4a", alpha=0.45, linewidths=0)
        # highlight the query window [best_offset, best_offset + query_len]
        if best_offset is not None:
            x0 = best_offset; x1 = best_offset + query_len_frames
            ax.axvspan(x0, x1, color="#c84b2f", alpha=0.16)
            ax.axvline(x0, color="#c84b2f", lw=1.2)
            ax.axvline(x1, color="#c84b2f", lw=1.2)
            ymax = freqs.max() if freqs.size else 1
            ax.text(x0, ymax * 1.02, " query clip sits here",
                    color="#c84b2f", fontsize=9, fontweight="bold", va="bottom")
    ax.set_xlabel("time (frames through the whole song)")
    ax.set_ylabel("freq bin")
    ax.set_title(f"Where in “{label}” the clip sits", fontsize=12, loc="left", fontweight="bold")
    _style(ax); fig.tight_layout()
    return fig


def plot_alignment_spike(offsets, top_label, top_score):
    """STEP 3 — the alignment spike: all matched-hash votes for the winning
    song across every offset. A real match converges into one tall bar above a
    flat noise floor."""
    fig, ax = plt.subplots(figsize=(14.4, 3.6), dpi=130)
    fig.patch.set_facecolor(CREAM)
    votes = offsets.get(top_label, [])
    if votes:
        lo, hi = min(votes), max(votes)
        span = max(hi - lo, 1)
        bins = np.linspace(lo - span * 0.05, hi + span * 0.05, 200)
        ax.hist(votes, bins=bins, color="#e8a33d")
        # annotate the dominant offset
        peak_off = Counter(votes).most_common(1)[0][0]
        ax.annotate(f"{top_score} hashes\nagree on one offset",
                    xy=(peak_off, top_score), xytext=(peak_off + span * 0.18, top_score * 0.7),
                    color="#9a5b00", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#c77a0a", lw=1.3))
        ax.text(0.99, 0.12, "chance matches (noise floor)", transform=ax.transAxes,
                ha="right", color="#7c8a87", fontsize=8)
    ax.set_xlabel("time offset  (database frame − query frame)")
    ax.set_ylabel("aligned hashes")
    ax.set_title("The alignment spike",
                 fontsize=12, loc="left", fontweight="bold")
    _style(ax); fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
      <div class="dial"><div>◑</div></div>
      <div>
        <div class="kicker">EE200 · Signals, Systems &amp; Networks</div>
        <p class="h-title">Sonic <em>Signatures</em></p>
      </div>
    </div>
<p class="h-sub">
  Identify any track from a fraction of its audio. By extracting a sparse 
  <b>constellation of time-frequency landmarks</b>, the engine uses combinatorial 
  hashing and precise offset alignment to lock onto a perfect match.
</p>
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
    """
    <div class="howto">
      <div class="chip"><div class="num">1</div><div class="txt">
        <b>Listen</b><span>A clip becomes a spectrogram — frequencies over time.</span></div></div>
      <div class="chip"><div class="num">2</div><div class="txt">
        <b>Distil</b><span>Only the strongest peaks are kept, then paired into hashes.</span></div></div>
      <div class="chip"><div class="num">3</div><div class="txt">
        <b>Match</b><span>The song whose hashes share one offset wins the vote.</span></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="statline">
      <div class="stat"><b>{len(LABELS)}</b><span>songs indexed</span></div>
      <div class="stat"><b>{len(db):,}</b><span>distinct hashes</span></div>
      <div class="stat"><b>{sum(len(v) for v in db.values()):,}</b><span>landmark entries</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

tab_single, tab_batch, tab_lib = st.tabs(["Identify", "Batch", "Library"])


# ======================================================================
# SINGLE-CLIP MODE
# ======================================================================
def run_single(y, audio_bytes=None, audio_mime=None):
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
    timings["match"] = (time.perf_counter() - t0) * 1000

    total = sum(timings.values())
    top = ranked[0] if ranked else None
    runner = ranked[1] if len(ranked) > 1 else None
    decisive = (top[1] / runner[1]) if (top and runner and runner[1]) else float("inf")

    # A real match dominates: a large absolute score AND a large lead over the
    # runner-up. A wrong song produces only scattered coincidental collisions —
    # a low score that barely beats the next song. We require both to clear the
    # bar, so an unrelated clip is reported as "not matched" rather than guessed.
    is_match, _, _, _ = is_confident(ranked)

    # ---- verdict first (the answer people want) ----
    if is_match:
        ratio_txt = "∞" if decisive == float("inf") else f"{decisive:.0f}×"
        st.markdown(
            f"<div class='verdict'><div class='tag'>Identified</div>"
            f"<div class='name'>{top[0]}</div>"
            f"<div class='meta'>score {top[1]}  ·  {ratio_txt} clearer than the runner-up"
            f"{('  ·  ' + runner[0]) if runner else ''}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        # explain WHY it didn't match, using the numbers
        if top is None:
            why = "no landmarks from this clip appear in the library"
        else:
            why = (f"best candidate “{top[0]}” scored only {top[1]} "
                   f"({decisive:.1f}× the runner-up) — that is noise-level, "
                   f"not a real alignment")
        st.markdown(
            f"<div class='verdict miss'><div class='tag'>Query not matched</div>"
            f"<div class='name'>Not in the library</div>"
            f"<div class='meta'>{why}</div></div>",
            unsafe_allow_html=True,
        )

    # ---- inline player for the analysed clip ----
    if audio_bytes is not None:
        st.markdown("<div class='section'>The clip</div>", unsafe_allow_html=True)
        st.audio(audio_bytes, format=audio_mime or "audio/wav")

    st.write("")

    # ---- pipeline strip ----
    st.markdown("<div class='section'>Pipeline</div>", unsafe_allow_html=True)
    nodes = [
        ("Spectrogram", f"{timings['spectrogram']:.0f}ms", f"{Sdb.shape[0]}×{Sdb.shape[1]} bins"),
        ("Constellation", f"{timings['constellation']:.0f}ms", f"{len(peaks)} peaks"),
        ("Hashing", f"{timings['hashing']:.0f}ms", f"{len(qhashes):,} hashes"),
        ("Match", f"{timings['match']:.0f}ms", f"offset {best_off if best_off is not None else '—'}"),
        ("Total", f"{total:.0f}ms", f"{len(LABELS)} tracks searched"),
    ]
    html = "<div class='flow'>"
    for n, v, s in nodes:
        html += f"<div class='node'><div class='n'>{n}</div><div class='v'>{v}</div><div class='s'>{s}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.write("")

    # ---- STEP 1: feature extraction (spectrogram -> constellation) ----
    st.markdown(
        f"""<div class="step"><div class="ey">Step 1 · Feature Extraction</div>
        <h4>Spectrogram to Constellation Map</h4>
        <p>The raw audio is first transformed into a spectrogram, mapping frequency intensity over time. To ensure robustness against noise and equalization variations, we discard the bulk of the data. Only the <b>{len(peaks)} strongest local maxima</b>—the constellation peaks—are preserved, distilling the track down to an unbreakable acoustic core.</p></div>""",
        unsafe_allow_html=True,
    )
    fig_spec = plot_spectrogram(f, t, Sdb)
    fig_const = plot_constellation(f2, t2, Sdb2, peaks)
    c1, c2 = st.columns(2)
    with c1: st.pyplot(fig_spec, use_container_width=True)
    with c2: st.pyplot(fig_const, use_container_width=True)

    query_len_frames = len(t2)   # number of time frames in the query

    # ---- STEP 2: database search (where in the song) ----
    if is_match and top:
        st.markdown(
            f"""<div class="step"><div class="ey">Step 2 · Database Search</div>
        <h4>Combinatorial Hash Matching and Temporal Alignment</h4>
        <p>The <b>{len(qhashes):,} combinatorial hashes</b> generated from the query clip are cross-referenced against the entire indexed database. The visualization below displays the complete stored fingerprint of the predicted match, <b>{top[0] if top else 'the top candidate'}</b>. The highlighted region demonstrates the precise temporal offset where the query's acoustic signature mathematically aligns with the original track.</p></div>""",
            unsafe_allow_html=True,
        )
        fig_map = plot_song_map(db, top[0], best_off, query_len_frames)
        st.pyplot(fig_map, use_container_width=True)
    else:
        fig_map = None

    # ---- STEP 3: the proof (alignment spike) ----
    st.markdown(
        f"""<div class="step"><div class="ey">Step 3 · The Proof</div>
        <h4>Temporal Convergence and The Alignment Spike</h4>
        <p>Each matched hash casts a vote for a relative time offset (database frame minus query frame). While random coincidences scatter uniformly across a flat noise floor, a true acoustic match forces these alignments to converge. Here, <b>{top[1] if top else 0} hashes mathematically agree on a single temporal offset</b>. A spike of this magnitude provides definitive proof of identification.</p></div>""",
        unsafe_allow_html=True,
    )
    fig_hist = plot_offset_hist(offsets, top[0] if top else "", runner[0] if runner else None)
    fig_spike = plot_alignment_spike(offsets, top[0] if top else "", top[1] if top else 0)
    st.pyplot(fig_spike, use_container_width=True)
    with st.expander("Compare against the runner-up song"):
        st.pyplot(fig_hist, use_container_width=True)

    if ranked:
        with st.expander("Full ranking (top 10)"):
            dfr = pd.DataFrame(ranked[:10], columns=["song", "score"])
            dfr.index = np.arange(1, len(dfr) + 1)
            st.dataframe(dfr, use_container_width=True)

    figs = [fig_spec, fig_const, fig_hist, fig_spike]
    if fig_map is not None:
        figs.append(fig_map)
    for _fig in figs:
        plt.close(_fig)
    del Sdb, Sdb2, offsets
    gc.collect()


with tab_single:
    st.markdown("<div class='section'>Search</div>", unsafe_allow_html=True)
    st.markdown("### Identify a clip")

    up = st.file_uploader(
        "Drop a query clip — WAV, MP3, FLAC, OGG or M4A",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="single_up",
    )

    # preview player for the uploaded file
    if up is not None:
        st.audio(up)

    samples = list_samples()
    chosen_sample = None
    if samples:
        st.markdown("<div class='section' style='margin-top:.7rem'>Or try a sample</div>",
                    unsafe_allow_html=True)
        cols = st.columns(min(len(samples), 4))
        for i, sp in enumerate(samples):
            with cols[i % len(cols)]:
                st.caption(os.path.splitext(os.path.basename(sp))[0])
                st.audio(sp)
                if st.button("Identify this", key=f"samp_{i}"):
                    chosen_sample = sp

    st.write("")
    go = st.button("Identify", type="primary", key="single_go")

    if chosen_sample or (go and up):
        try:
            with st.spinner("Listening…"):
                if chosen_sample:
                    y = smart_load(chosen_sample, filename=chosen_sample)
                    with open(chosen_sample, "rb") as fh:
                        ab = fh.read()
                    mime = "audio/" + os.path.splitext(chosen_sample)[1].lstrip(".")
                else:
                    ab = up.getvalue()
                    y = smart_load(io.BytesIO(ab), filename=up.name)
                    mime = up.type or "audio/wav"
            if y is None or len(y) == 0:
                st.error("That file didn't contain readable audio. "
                         "Try a WAV, MP3, FLAC, OGG or M4A clip.")
            else:
                if len(y) < SR // 2:
                    st.warning("That clip is very short — results may be unreliable.")
                run_single(y, audio_bytes=ab, audio_mime=mime)
        except Exception:
            st.error("Couldn't read that file — it may be corrupted or an "
                     "unsupported format. Try a WAV, MP3, FLAC, OGG or M4A clip.")
    elif go and not up:
        st.info("Drop a clip first, or pick a sample below.")


# ======================================================================
# BATCH MODE
# ======================================================================
with tab_batch:
    st.markdown("<div class='section'>Evaluation</div>", unsafe_allow_html=True)
    st.markdown("### Batch → results.csv")
    st.caption(
        "Upload a set of query clips. The output is a CSV with exactly two columns — "
        "filename, prediction — where prediction is the matched song's filename without extension."
    )

    ups = st.file_uploader(
        "Upload query clips",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
        key="batch_up",
    )

    gate = st.checkbox(
        "Leave prediction blank when no confident match",
        value=False,
        help="Off (default): always write the best-guess song for every clip — "
             "use this for automated evaluation where each clip is a library song. "
             "On: out-of-library clips get an empty prediction.",
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
                    y = smart_load(fobj, filename=fname)
                    ranked, _, _ = identify(y, db)
                    if gate:
                        ok, lab, _, _ = is_confident(ranked)
                        pred = lab if ok else ""
                    else:
                        pred = ranked[0][0] if ranked else ""
                except Exception:
                    pred = ""
                finally:
                    del y
                    gc.collect()
                rows.append({"filename": fname, "prediction": pred})
                prog.progress(i / len(ups), text=f"Processed {i}/{len(ups)}")
            prog.empty()

            df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(df, use_container_width=True)

            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)      # exactly: filename,prediction
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
    st.markdown("<div class='section'>Index</div>", unsafe_allow_html=True)
    st.markdown(f"### {len(LABELS)} songs in the library")
    st.caption("These labels are exactly what the identifier outputs.")
    q = st.text_input("Filter", placeholder="type to filter songs…", label_visibility="collapsed")
    shown = [l for l in LABELS if q.lower() in l.lower()] if q else LABELS
    libdf = pd.DataFrame({"song": shown})
    libdf.index = np.arange(1, len(libdf) + 1)
    st.dataframe(libdf, use_container_width=True, height=460)
