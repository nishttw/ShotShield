"""Download and index the open gunshot datasets used by ShotShield v4.

Every loader returns a DataFrame with the same columns:

    source    dataset name
    name      file name (for reports)
    path      local audio file
    group     recordings that must stay in the same split (same shot on several mics, same session, ...)
    label     1 = gunshot, 0 = anything else
    category  finer class, e.g. "c3gd:9x19mm" or "firecracker"
    kind      gunshot / firecracker / background / impulsive / other
    split     train / val / test (test = never trained on)
    rank      0 = the loudest 4 s window of the file, 1 = the second loudest (non-overlapping), ...
    license, credit

All datasets here are CC BY 4.0 unless noted (BGG: game audio, non-commercial research only).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UA = "Mozilla/5.0 (ShotShield research dataset download)"   # some hosts reject Python's default agent
COLUMNS = ["source", "name", "path", "group", "label", "category", "kind", "split", "rank", "license", "credit"]

C3GD_ZIP = "https://zenodo.org/api/records/22286299/files/C3GD.zip/content"
MULTI_FIREARM_ZIP = "https://zenodo.org/api/records/7004819/files/edge-collected-gunshot-audio.zip/content"
SESA_ZIP = "https://zenodo.org/api/records/3519845/files/SESA.zip/content"
M16_GUNSHOT_ZIP = ("https://data.mendeley.com/public-files/datasets/kbgw323248/files/"
                   "008e6bcb-b892-4c56-94d5-880e14239967/file_downloaded")
M16_FIRECRACKER_ZIP = ("https://data.mendeley.com/public-files/datasets/kbgw323248/files/"
                       "5f4cd8fc-0f78-4f70-9e0e-3e7e09b776da/file_downloaded")
BELIZE_API = "https://data.mendeley.com/public-api/datasets/x48cwz364j/files?folder_id={}&version=3"
BELIZE_FOLDERS = {"train_gunshot": "d21ad8ca-ca07-4a75-a852-29c2a505e62b",
                  "test_gunshot": "83dd9428-6b72-4db7-90e8-70f84be9fa7d",
                  "test_background": "86d76b5d-a89c-41c7-9bab-76b0c9de9e63",
                  "train_background_zip": "7c501143-6f46-412f-b784-b84fad4ccdb0"}
BGG_DRIVE_ID = "1TIEgt1KEJtcK5zDhnuHvtK1nisv-C1fj"

CREDITS = {
    "c3gd": "C3GD, S. Gurny and R. Quinn (Certus Innovations), CC BY 4.0, doi:10.48550/arXiv.2606.18135",
    "multi_firearm": "Multi-firearm, multi-orientation gunshot dataset, R. Kabealo et al. (Florida Tech / AFRL), "
                     "CC BY 4.0, doi:10.1016/j.dib.2023.109091",
    "belize": "Tropical forest gunshot dataset, L. Katsis et al. (Belize), CC BY 4.0, doi:10.17632/x48cwz364j.3",
    "m16": "M16 rifle gunshots and firecrackers, Mendeley Data, CC BY 4.0, doi:10.17632/kbgw323248.1",
    "sesa": "SESA (Sound Events for Surveillance Applications), Freesound-derived, CC BY 4.0, doi:10.5281/zenodo.3519845",
    "bgg": "BGG in-game PUBG gun sounds, J. Park et al. 2022 — non-commercial research only",
}


# ---------- helpers ----------
def fetch(url: str, path: str | Path, tries: int = 6) -> Path:
    """Download url to path (skipped if present), with retries."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r, open(str(path) + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(str(path) + ".part", path)
            return path
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404) or i == tries - 1:
                raise
        except Exception:
            if i == tries - 1:
                raise
        time.sleep(min(60, 2 ** i))
    return path


def fetch_json(url: str):
    import json
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def extract(zip_path: Path, dest: Path, members=None, delete_zip: bool = True) -> Path:
    """Unzip (only `members`, if given) and optionally delete the archive to save disk space."""
    marker = dest / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest, members=members)
        marker.touch()
    if delete_zip and Path(zip_path).exists():
        os.remove(zip_path)
    return dest


def hash_split(keys, val_frac: float = 0.15, salt: str = "v4"):
    """Deterministic train/val assignment by group key (the same key always lands in the same split)."""
    def h(k):
        return int(hashlib.md5(f"{salt}:{k}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return np.array(["val" if h(k) < val_frac else "train" for k in keys])


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df:
            df[c] = 0 if c == "rank" else ""
    df["rank"] = df["rank"].fillna(0).astype(int)
    return df[COLUMNS + [c for c in df.columns if c not in COLUMNS]]


# ---------- datasets ----------
def load_c3gd(data_dir: str | Path, test_event: str = "ny_gravel_pit", download: bool = True) -> pd.DataFrame:
    """C3GD: 12,112 one-second outdoor gunshots, 48 guns, 15 mics (incl. phones), 4 recording events.

    One whole event is held out as the test set (unseen place, day and setup). The same shot is often
    recorded by several mics, so train/val are split by original recording (file_id), never by clip.
    """
    root = Path(data_dir) / "c3gd"
    if download:
        extract(fetch(C3GD_ZIP, root / "C3GD.zip"), root)
    base = next(p.parent for p in root.rglob("metadata.csv") if (p.parent / "data").is_dir())
    md = pd.read_csv(base / "metadata.csv")
    group = md["event_id"] + ":" + md["file_id"].astype(str)
    split = np.where(md["event_id"] == test_event, "test", hash_split(group))
    return frame([{"source": "c3gd", "name": f, "path": str(base / "data" / f), "group": g, "label": 1,
                   "category": f"c3gd:{c}", "kind": "gunshot", "split": s, "license": "CC BY 4.0",
                   "credit": CREDITS["c3gd"], "mic": m, "is_phone": int(ph), "event": e}
                  for f, g, c, s, m, ph, e in zip(md["filename"], group, md["class_name"], split, md["mic_id"],
                                                  md["is_phone"], md["event_id"])])


def load_multi_firearm(data_dir: str | Path, test_session: int = 2, channels_per_recording: int = 1,
                       seed: int = 0, download: bool = True) -> pd.DataFrame:
    """Kabealo et al.: 549 recordings of 4 guns on 27 mics (phones, USB mics, Pi mic arrays), bursts up to 23 shots.

    Multichannel arrays add near-identical copies, so each array recording keeps its channel mean plus
    `channels_per_recording` random single channels. Recording session 2 is the test set.
    """
    root = Path(data_dir) / "multi_firearm"
    if download:
        extract(fetch(MULTI_FIREARM_ZIP, root / "mf.zip"), root)
    meta_path = next(root.rglob("gunshot-audio-all-metadata.csv"))
    md = pd.read_csv(meta_path)
    md.columns = [c.strip() for c in md.columns]
    per_uuid = md.drop_duplicates("uuid").set_index("uuid")
    shots = md.assign(key=md["filename"].astype(str)).set_index("key")["num_gunshots"].to_dict()
    rng = np.random.default_rng(seed)
    files = sorted(p for p in root.rglob("*.wav"))
    by_clip: dict[tuple[str, str], list[Path]] = {}                  # (recording, clip) -> mono / mean / channels
    for p in files:
        v = re.search(r"_v(\d+)$", p.stem)
        by_clip.setdefault((p.stem.split("_")[0], v.group(1) if v else ""), []).append(p)
    rows = []
    for (uuid, _), paths in sorted(by_clip.items()):
        chans = [p for p in paths if "_chan" in p.stem]
        keep = [p for p in paths if "_chan" not in p.stem]           # mono recordings and array means
        if chans:
            keep += list(rng.choice(chans, size=min(channels_per_recording, len(chans)), replace=False))
        m = per_uuid.loc[uuid] if uuid in per_uuid.index else None
        session = int(m["recording_session"]) if m is not None else 1
        for p in keep:
            version = re.search(r"_v(\d+)$", p.stem)
            n_shots = shots.get(f"{uuid}_v{version.group(1)}" if version else uuid)
            rows.append({"source": "multi_firearm", "name": p.name, "path": str(p), "group": uuid, "label": 1,
                         "category": f"mf:{p.parent.name}", "kind": "gunshot",
                         "split": "test" if session == test_session else None, "license": "CC BY 4.0",
                         "credit": CREDITS["multi_firearm"], "mic": m["microphone"] if m is not None else "",
                         "device": m["device_name"] if m is not None else "", "num_gunshots": n_shots,
                         "is_phone": int(m is not None and "Galaxy" in str(m["microphone"]))})
    df = frame(rows)
    todo = df["split"].isna()
    df.loc[todo, "split"] = hash_split(df.loc[todo, "group"])
    return df


def load_belize(data_dir: str | Path, n_background_train: int = 3000, n_background_test: int = 600,
                seed: int = 0, download: bool = True) -> pd.DataFrame:
    """Katsis et al.: 749 shotgun shots 0-1000 m away in Belizean forest + forest background, 8 kHz.

    The dataset's own 'Validation data' period (later in time) is used as the test set. Background clips
    are the negatives, so both classes share the same 8 kHz bandwidth.
    """
    root = Path(data_dir) / "belize"
    rng = np.random.default_rng(seed)
    rows = []

    def listed(key, n=None):
        from concurrent.futures import ThreadPoolExecutor
        items = fetch_json(BELIZE_API.format(BELIZE_FOLDERS[key]))
        if n is not None and len(items) > n:
            items = [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]
        paths = [root / key / it["filename"] for it in items]
        if download:                                   # one small file each: download in parallel
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda a: fetch(a[0]["content_details"]["download_url"], a[1]), zip(items, paths)))
        return paths

    for key, label, kind in [("train_gunshot", 1, "gunshot"), ("test_gunshot", 1, "gunshot"),
                             ("test_background", 0, "background")]:
        paths = listed(key, n_background_test if kind == "background" else None)
        for p in paths:
            rows.append({"source": "belize", "name": p.name, "path": str(p), "group": p.name[:6], "label": label,
                         "category": f"belize:{kind}", "kind": kind,
                         "split": "test" if key.startswith("test") else None})
    # training backgrounds come as one 1.3 GB zip: unpack only a random sample
    if n_background_train:
        item = fetch_json(BELIZE_API.format(BELIZE_FOLDERS["train_background_zip"]))[0]
        zpath = root / "background.zip"
        dest = root / "train_background"
        if download and not (dest / ".extracted").exists():
            fetch(item["content_details"]["download_url"], zpath)
            with zipfile.ZipFile(zpath) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".wav")]
                pick = [names[i] for i in sorted(rng.choice(len(names), min(n_background_train, len(names)),
                                                            replace=False))]
            extract(zpath, dest, members=pick)
        for p in sorted(dest.rglob("*.WAV")) + sorted(dest.rglob("*.wav")):
            rows.append({"source": "belize", "name": p.name, "path": str(p), "group": p.name[:6], "label": 0,
                         "category": "belize:background", "kind": "background", "split": None})
    df = frame(rows).assign(license="CC BY 4.0", credit=CREDITS["belize"])
    todo = df["split"].isna() | (df["split"] == "")
    df.loc[todo, "split"] = hash_split(df.loc[todo, "group"])
    return df


def _angle(path: Path) -> int:
    m = re.match(r"(\d+)", path.parent.name if path.parent.name.isdigit() else path.stem)
    return int(m.group(1)) % 360 if m else -1


def load_m16(data_dir: str | Path, test_angles=range(180, 360), firecracker_events: int = 4,
             download: bool = True) -> pd.DataFrame:
    """M16 rifle shots at 50-100 m and 0-345 degrees (4-mic array), plus firecrackers at the same positions.

    Firecrackers are hard negatives; each recording holds several bangs, so up to `firecracker_events`
    separate loud windows are taken per file. Positions at 180-345 degrees are the test set.
    """
    root = Path(data_dir) / "m16"
    if download:
        extract(fetch(M16_GUNSHOT_ZIP, root / "gunshot.zip"), root / "gunshot_x")
        extract(fetch(M16_FIRECRACKER_ZIP, root / "firecracker.zip"), root / "firecracker_x")
    rows = []
    for p in sorted((root / "gunshot_x").rglob("*.wav")):
        a = _angle(p)
        rows.append({"source": "m16", "name": f"{a}/{p.name}", "path": str(p), "group": f"gun:{a}:{p.stem}",
                     "label": 1, "category": "m16:rifle", "kind": "gunshot",
                     "split": "test" if a in test_angles else None, "angle": a})
    for p in sorted((root / "firecracker_x").rglob("*.wav")):
        a = _angle(p)
        for k in range(firecracker_events):
            rows.append({"source": "m16", "name": f"{p.name}#{k}", "path": str(p), "group": f"fc:{a}:{p.stem}",
                         "label": 0, "category": "firecracker", "kind": "firecracker", "rank": k,
                         "split": "test" if a in test_angles else None, "angle": a})
    df = frame(rows).assign(license="CC BY 4.0", credit=CREDITS["m16"])
    todo = df["split"].isna() | (df["split"] == "")
    df.loc[todo, "split"] = hash_split(df.loc[todo, "group"])
    return df


def load_sesa(data_dir: str | Path, download: bool = True) -> pd.DataFrame:
    """SESA: 585 Freesound clips (casual / gunshot / explosion / siren). Its own test folder is the test set.

    Taken from Freesound without IDs, so a few clips may also be in FSD50K or UrbanSound8K.
    """
    root = Path(data_dir) / "sesa"
    if download:
        extract(fetch(SESA_ZIP, root / "SESA.zip"), root)
    kinds = {"casual": "other", "gunshot": "gunshot", "explosion": "impulsive", "alarm": "other", "siren": "other"}
    rows = []
    for p in sorted(root.rglob("*.wav")):
        cls = p.stem.rsplit("_", 1)[0].lower()
        rows.append({"source": "sesa", "name": f"{p.parent.name}/{p.name}", "path": str(p), "group": p.stem,
                     "label": int(cls == "gunshot"), "category": f"sesa:{cls}", "kind": kinds.get(cls, "other"),
                     "split": "test" if p.parent.name == "test" else None})
    df = frame(rows).assign(license="CC BY 4.0", credit=CREDITS["sesa"])
    todo = df["split"].isna() | (df["split"] == "")
    df.loc[todo, "split"] = hash_split(df.loc[todo, "group"])
    return df


def load_bgg(data_dir: str | Path, test_frac: float = 0.2, download: bool = True) -> pd.DataFrame:
    """BGG: 2,195 PUBG in-game gun sounds (37 guns, 0-600 m). Non-commercial research use only.

    Used only in the 'with / without game audio' experiment. 20% is kept back as a game-sound test set.
    """
    root = Path(data_dir) / "bgg"
    if download and not any(root.rglob("*.mp3")):
        import gdown   # pip install gdown
        root.mkdir(parents=True, exist_ok=True)
        z = gdown.download(id=BGG_DRIVE_ID, output=str(root / "gun_sound_v2.zip"), quiet=True)
        extract(Path(z), root)
    rows = []
    for p in sorted(root.rglob("*.mp3")):
        parts = p.stem.split("_")
        gun = parts[0].lower()
        label = 0 if gun in ("nogun", "none", "no", "noise", "silence") else 1
        rows.append({"source": "bgg", "name": p.name, "path": str(p), "group": p.stem, "label": label,
                     "category": f"bgg:{gun}", "kind": "gunshot" if label else "other",
                     "split": "test" if hash_split([p.stem], test_frac, "bgg")[0] == "val" else "train"})
    return frame(rows).assign(license="non-commercial research only", credit=CREDITS["bgg"])


def load_home_clips(search_dir: str | Path) -> pd.DataFrame:
    """Clips saved with the live demo's "Save for training" (labels.csv + clips/*.wav). Test set only.

    Unreviewed clips are skipped; every folder with a demo labels.csv is used.
    """
    rows = []
    for csv in Path(search_dir).rglob("labels.csv"):
        md = pd.read_csv(csv)
        if not {"file", "label", "peak_prob"} <= set(md.columns):
            continue
        for _, r in md.iterrows():
            if r["label"] not in ("gunshot", "not_gunshot"):
                continue
            p = csv.parent / r["file"]
            if p.exists():
                note = r.get("note")
                rows.append({"source": "home", "name": p.name, "path": str(p), "group": csv.parent.name,
                             "label": int(r["label"] == "gunshot"), "category": f"home:{r['label']}",
                             "kind": "gunshot" if r["label"] == "gunshot" else "other", "split": "test",
                             "license": "own recording", "credit": "",
                             "note": "" if pd.isna(note) else str(note)})
    return frame(rows) if rows else frame([]).iloc[0:0]
