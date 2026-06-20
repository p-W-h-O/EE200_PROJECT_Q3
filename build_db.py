"""
build_db.py — Index the song library into fingerprint_db.pkl

Run this ONCE locally (or in Colab) after downloading the provided songs:

    python build_db.py /path/to/songs

It writes fingerprint_db.pkl in the current folder. Commit that .pkl
alongside the app so the deployed Streamlit app works immediately
without re-indexing (Streamlit Cloud has no song files and limited CPU).

Uses the SAME soundfile-based decoder as the app, so the database and
the queries are guaranteed consistent. (It also matches a librosa-built
database hash-for-hash — verified — so an existing .pkl from the Q3A
notebook is fine to reuse too.)
"""
import os, sys, glob, pickle, time
from collections import defaultdict
from fingerprint import load_full, get_peaks, hashes_from_peaks


def list_library(songs_dir):
    """(label, path) for every audio file; label = filename without extension."""
    exts = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(songs_dir, e))
    paths = sorted(p for p in paths
                   if "__MACOSX" not in p and not os.path.basename(p).startswith("._"))
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]


def build_database(library):
    db = defaultdict(list)
    for i, (label, path) in enumerate(library, 1):
        t0 = time.time()
        y = load_full(path)            # index the ENTIRE song (no time cap)
        _, _, _, peaks = get_peaks(y)
        for h, t1 in hashes_from_peaks(peaks):
            db[h].append((t1, label))
        print(f"  [{i:3d}/{len(library)}] {label:40s} "
              f"{len(peaks):5d} peaks  ({time.time()-t0:.1f}s)")
    return db


def main():
    import gzip
    songs_dir = sys.argv[1] if len(sys.argv) > 1 else "songs"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "fingerprint_db.pkl.gz"

    library = list_library(songs_dir)
    if not library:
        print(f"No audio files found in '{songs_dir}'.")
        sys.exit(1)
    print(f"Found {len(library)} songs in '{songs_dir}'. Indexing...\n")

    db = build_database(library)

    payload = {
        "db": dict(db),
        "labels": sorted({lab for lab, _ in library}),
        "config": {
            "SR": 11025, "NPERSEG": 1024, "HOP": 512,
            "NEIGH": 20, "THRESH_DB": -45,
            "FAN_VALUE": 15, "MIN_DT": 1, "MAX_DT": 100,
        },
    }
    opener = gzip.open if out_path.endswith(".gz") else open
    with opener(out_path, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nDistinct hashes : {len(db):,}")
    print(f"Total entries   : {sum(len(v) for v in db.values()):,}")
    print(f"Songs indexed   : {len(payload['labels'])}")
    print(f"Saved           : {out_path}")


if __name__ == "__main__":
    main()
