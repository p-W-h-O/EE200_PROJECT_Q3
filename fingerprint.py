
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


SR        = 11025     # resample rate (Hz)
NPERSEG   = 1024      # STFT window length (samples)
HOP       = 512       # hop between windows
NEIGH     = 20        # neighborhood size for local-maxima detection
THRESH_DB = -45       # ignore peaks quieter than this (dB)
FAN_VALUE = 15        # pair each anchor with up to this many later peaks
MIN_DT    = 1         # min time-bin gap in a pair
MAX_DT    = 100       # max time-bin gap (target-zone width)


MAX_QUERY_SECONDS = 30

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


# Audio loading 
def _resample(y, file_sr, sr):
    if file_sr == sr:
        return y.astype(np.float32)
    g = np.gcd(int(file_sr), int(sr))
    out = sps.resample_poly(y, sr // g, file_sr // g)
    return out.astype(np.float32)


def _decode_ffmpeg(path, sr=SR, max_seconds=None):

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

    with sf.SoundFile(path_or_file) as snd:
        file_sr = snd.samplerate
        frames = -1 if max_seconds is None else int(max_seconds * file_sr)
        y = snd.read(frames=frames, dtype="float32", always_2d=False)
    if y.ndim > 1:                           
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    return _resample(y, file_sr, sr)


def load_audio(path_or_file, sr=SR, max_seconds=None):

    if hasattr(path_or_file, "seek"):
        try: path_or_file.seek(0)
        except Exception: pass

    try:
        return _decode_soundfile(path_or_file, sr=sr, max_seconds=max_seconds)
    except Exception:
  
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

    return load_audio(path_or_file, sr=sr, max_seconds=max_seconds)


def load_full(path_or_file, sr=SR):

    return load_audio(path_or_file, sr=sr, max_seconds=None)


# Spectrogram and constellation of peaks

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



# Hashing (peak pairing)

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



# Matching via the offset histogram

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
