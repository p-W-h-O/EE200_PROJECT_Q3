"""
fingerprint.py — Audio fingerprinting core (Shazam-style).

NO librosa / numba / llvmlite.  Decodes with soundfile (libsndfile),
resamples with scipy.signal.resample_poly.  This is what makes the app
build on Streamlit Cloud's Python 3.14 (no source compilation).

Verified: a database built with librosa matches queries decoded here
hash-for-hash, because both ultimately read the same PCM via libsndfile
and the 11025 Hz resample target divides cleanly.
"""
from collections import defaultdict, Counter
import io
import os
import gc
import shutil
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from scipy import signal as sps
from scipy.ndimage import maximum_filter

# ---- configuration (MUST match the indexing notebook exactly) ----
SR        = 11025     # resample rate (Hz)
NPERSEG   = 1024      # STFT window length (samples)
HOP       = 512       # hop between windows
NEIGH     = 20        # neighborhood size for local-maxima detection
THRESH_DB = -45       # ignore peaks quieter than this (dB)
FAN_VALUE = 15        # pair each anchor with up to this many later peaks
MIN_DT    = 1         # min time-bin gap in a pair
MAX_DT    = 100       # max time-bin gap (target-zone width)

# ---- analysis window ----
# A fingerprint query needs only a short slice of audio for a confident match.
# We analyse the first 30 seconds of any clip, which keeps memory and compute
# low while giving plenty of evidence. Set to None to read the whole file
# (used when INDEXING the library, where we want every second of a song).
MAX_QUERY_SECONDS = 30

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ----------------------------------------------------------------------
# Audio loading  (soundfile + scipy — no librosa)
# ----------------------------------------------------------------------
def _resample(y, file_sr, sr):
    if file_sr == sr:
        return y.astype(np.float32)
    g = np.gcd(int(file_sr), int(sr))
    out = sps.resample_poly(y, sr // g, file_sr // g)
    return out.astype(np.float32)


def _decode_ffmpeg(path, sr=SR, max_seconds=None):
    """
    Decode with ffmpeg straight to mono float32 PCM at `sr`, reading only
    the first `max_seconds` (so RAM stays tiny). Returns a 1-D float32 array.
    Raises if ffmpeg isn't available or fails.
    """
    if not _HAS_FFMPEG:
        raise RuntimeError("ffmpeg not available")
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]      # decode only first N seconds
    cmd += [
        "-i", path,
        "-ac", "1",                          # mono
        "-ar", str(sr),                      # resample inside ffmpeg
        "-f", "f32le",                       # raw 32-bit float little-endian
        "-",                                 # write to stdout
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            "ffmpeg failed: " + proc.stderr.decode("utf-8", "ignore")[:200]
        )
    y = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    return y


def _decode_soundfile(path_or_file, sr=SR, max_seconds=None):
    """
    Decode with libsndfile, reading only the first `max_seconds` worth of
    frames so we never allocate the whole file. Returns mono float32 at `sr`.
    """
    with sf.SoundFile(path_or_file) as snd:
        file_sr = snd.samplerate
        frames = -1 if max_seconds is None else int(max_seconds * file_sr)
        y = snd.read(frames=frames, dtype="float32", always_2d=False)
    if y.ndim > 1:                           # stereo -> mono
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    return _resample(y, file_sr, sr)


def load_audio(path_or_file, sr=SR, max_seconds=None):
    """
    Decode an audio file to mono float32 at `sr` Hz, optionally capped to
    the first `max_seconds`. Accepts a path or a file-like object.

    Strategy: libsndfile first (fast, frame-capped). If that can't read the
    format (e.g. some M4A/AAC), fall back to ffmpeg with a hard time cap.
    """
    # file-like: rewind, then let soundfile try to stream it
    if hasattr(path_or_file, "seek"):
        try: path_or_file.seek(0)
        except Exception: pass

    try:
        return _decode_soundfile(path_or_file, sr=sr, max_seconds=max_seconds)
    except Exception:
        # ffmpeg needs a real path — spill a file-like upload to a temp file
        tmp = None
        try:
            if hasattr(path_or_file, "read"):
                try: path_or_file.seek(0)
                except Exception: pass
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(path_or_file.read()); tmp.flush(); tmp.close()
                path = tmp.name
            else:
                path = path_or_file
            return _decode_ffmpeg(path, sr=sr, max_seconds=max_seconds)
        finally:
            if tmp is not None:
                try: os.unlink(tmp.name)
                except Exception: pass


def smart_load(path_or_file, filename="", sr=SR, max_seconds=MAX_QUERY_SECONDS):
    """
    Load a QUERY clip for identification, analysing its leading window
    (`MAX_QUERY_SECONDS`). Pass max_seconds=None to read the whole file.
    """
    return load_audio(path_or_file, sr=sr, max_seconds=max_seconds)


def load_full(path_or_file, sr=SR):
    """Load an ENTIRE file (no time cap) — used when indexing the library."""
    return load_audio(path_or_file, sr=sr, max_seconds=None)


# ----------------------------------------------------------------------
# Spectrogram + constellation of peaks
# ----------------------------------------------------------------------
def compute_spectrogram(y):
    """Return (freqs, times, Sdb) — the dB spectrogram."""
    f, t, S = sps.spectrogram(
        y, fs=SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, window="hann"
    )
    Sdb = 10 * np.log10(S + 1e-10)
    return f, t, Sdb


def get_peaks(y):
    """
    Return (f, t, Sdb, peaks).
    peaks = list of (time_bin, freq_bin) local maxima above threshold.
    """
    f, t, Sdb = compute_spectrogram(y)
    mask = (maximum_filter(Sdb, size=NEIGH) == Sdb) & (Sdb > THRESH_DB)
    fi, ti = np.where(mask)
    return f, t, Sdb, list(zip(ti.tolist(), fi.tolist()))


# ----------------------------------------------------------------------
# Hashing (peak pairing)
# ----------------------------------------------------------------------
def hashes_from_peaks(peaks):
    """Pair each peak with nearby later peaks. hash=(f1,f2,dt); yields (hash, t1)."""
    peaks = sorted(peaks)
    out, n = [], len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        for j in range(1, FAN_VALUE + 1):
            if i + j < n:
                t2, f2 = peaks[i + j]
                dt = t2 - t1
                if MIN_DT <= dt <= MAX_DT:
                    out.append(((f1, f2, dt), t1))
    return out


# ----------------------------------------------------------------------
# Matching via the offset histogram
# ----------------------------------------------------------------------
def identify(query_y, db):
    """
    Match a query against the paired-hash database.

    Returns (ranked, offsets, best_offset_for_top):
      ranked  = [(label, score), ...] sorted by score desc
      offsets = {label: [offset, ...]}  raw offset votes per song
      best_offset_for_top = the winning offset bin of the top song (int) or None
    """
    _, _, _, peaks = get_peaks(query_y)
    offsets = defaultdict(list)
    for h, qt in hashes_from_peaks(peaks):
        if h in db:
            for db_time, label in db[h]:
                offsets[label].append(db_time - qt)
    scores = {lab: Counter(o).most_common(1)[0][1] for lab, o in offsets.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_offset = None
    if ranked:
        top = ranked[0][0]
        best_offset = Counter(offsets[top]).most_common(1)[0][0]
    return ranked, offsets, best_offset


def fingerprint_query(query_y):
    """Convenience: everything the single-clip view needs in one pass."""
    f, t, Sdb, peaks = get_peaks(query_y)
    return {"freqs": f, "times": t, "Sdb": Sdb, "peaks": peaks}


# ----------------------------------------------------------------------
# Confidence gate
# ----------------------------------------------------------------------
# A genuine match dominates: a large aligned-hash score AND a large lead over
# the runner-up. A wrong / out-of-library clip yields only scattered
# coincidental collisions — a tiny score that barely beats the next song.
# Measured separation on the 50-song library: true matches score in the
# thousands with ~1000x+ leads; false queries top out at score 3, ratio 1.5x.
# These thresholds sit comfortably in the gap with margin for weak/noisy clips.
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
