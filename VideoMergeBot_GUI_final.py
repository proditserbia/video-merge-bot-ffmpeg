#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Merge Bot — NVENC + Mezzanine + Auto‑Fallback (GPU-first) — FINAL
Author: prodit.rs

Additions vs previous FINAL:
• Mezzanine cache (uniform 2K/4K, 30 fps, yuv420p) → massive speedup on merges
• Auto-fallback encode paths: try NVDEC fast-path, then CPU decode + GPU upload
• Throughput tuning: constqp + p1, bilinear scaler, AQ/lookahead off
• Auto-concurrency: -1 picks sensible default based on GPU (A5000/4060 → 2)
• STOP is immediate (no new folders, ffmpeg terminate→kill), no “All done” on abort
• ffprobe launched with CREATE_NO_WINDOW on Windows (no flashing cmd)
• Help → About / Licenses preserved
• Tk Menu fix (tk.Menu, not ttk.Menu)

Base features:
• Numeric folder order
• Random no-repeat selection per folder, single target duration
• H.264 NVENC @ 30 fps, 2K/4K, bitrate 15–50 Mb/s (or constqp), AAC 320 kbps, Rec.709 tags
• No-stretch aspect ratio (scale+pad)
• Outputs per folder cap (respects existing), optional daily limit
• durations.json cache, pool.json, per-output logs
"""

from __future__ import annotations
import re
import json
import uuid
import random
import sqlite3
import threading
import subprocess
import hashlib
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List
import sys, webbrowser, time
from pathlib import Path
from tkinter import Tk, StringVar, IntVar, ttk, filedialog, Text, END, DISABLED, NORMAL, Toplevel

# -------------------------------------------------
# Small helpers
# -------------------------------------------------

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_gui(w: Optional[Text], msg: str):
    line = f"[{ts()}] {msg}\n"
    if w is None:
        print(line, end="")
        return
    w.configure(state=NORMAL)
    w.insert(END, line)
    w.see(END)
    w.configure(state=DISABLED)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        pass
    return default


def save_json(p: Path, data):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)

# -------------------------------------------------
# FFprobe duration cache (durations.json)
# -------------------------------------------------

def ffprobe_duration_seconds(ffprobe_bin: str, file_path: Path) -> float:
    try:
        cmd = [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        creationflags = 0
        try:
            creationflags = subprocess.CREATE_NO_WINDOW
        except Exception:
            pass
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=creationflags)
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


def get_duration_cached(ffprobe_bin: str, folder: Path, f: Path) -> float:
    cache_path = folder / "durations.json"
    cache = load_json(cache_path, {})
    key = f.name
    mtime = str(f.stat().st_mtime)
    entry = cache.get(key)
    if entry and entry.get("mtime") == mtime:
        return float(entry.get("duration", 0.0))
    dur = ffprobe_duration_seconds(ffprobe_bin, f)
    cache[key] = {"mtime": mtime, "duration": dur}
    save_json(cache_path, cache)
    return dur

# -------------------------------------------------
# No-repeat shuffle pool (pool.json)
# -------------------------------------------------

def load_pool(folder: Path) -> dict:
    return load_json(folder / "pool.json", {"cycle": 0, "remaining": []})


def save_pool(folder: Path, data: dict):
    save_json(folder / "pool.json", data)


def rebuild_pool(folder: Path) -> dict:
    clips = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.m4v", "*.avi"):
        clips.extend([x for x in folder.glob(ext) if x.is_file()])
    clips = [c.name for c in clips]
    random.shuffle(clips)
    data = load_pool(folder)
    data["cycle"] = int(data.get("cycle", 0)) + 1
    data["remaining"] = clips
    save_pool(folder, data)
    return data


def take_from_pool(folder: Path, count: int) -> List[str]:
    data = load_pool(folder)
    remaining = data.get("remaining", [])
    if len(remaining) < count:
        data = rebuild_pool(folder)
        remaining = data.get("remaining", [])
    take = remaining[:count]
    data["remaining"] = remaining[count:]
    save_pool(folder, data)
    return take

# -------------------------------------------------
# SQLite audit (outputs & daily limits)
# -------------------------------------------------
class AuditDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS outputs (
                id TEXT PRIMARY KEY,
                folder TEXT,
                created_at TEXT,
                target_minutes INTEGER,
                quality TEXT,
                bitrate_mbps INTEGER,
                result_path TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_count (
                folder TEXT,
                day TEXT,
                count INTEGER,
                PRIMARY KEY(folder, day)
            )
            """
        )
        conn.commit()
        conn.close()

    def increment_daily(self, folder: Path):
        day = date.today().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT count FROM daily_count WHERE folder=? AND day=?", (str(folder), day))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE daily_count SET count=? WHERE folder=? AND day=?", (row[0] + 1, str(folder), day))
        else:
            cur.execute("INSERT INTO daily_count(folder, day, count) VALUES (?,?,?)", (str(folder), day, 1))
        conn.commit()
        conn.close()

    def get_daily(self, folder: Path) -> int:
        day = date.today().isoformat()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT count FROM daily_count WHERE folder=? AND day=?", (str(folder), day))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else 0

    def add_output(self, folder: Path, target_minutes: int, quality: str, bitrate_mbps: int, result_path: Path):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO outputs(id, folder, created_at, target_minutes, quality, bitrate_mbps, result_path) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), str(folder), datetime.now().isoformat(timespec='seconds'), target_minutes, quality, bitrate_mbps, str(result_path))
        )
        conn.commit()
        conn.close()

# -------------------------------------------------
# Config & Engine
# -------------------------------------------------
@dataclass
class JobConfig:
    base_folder: Path
    target_minutes: int
    outputs_per_folder: int
    daily_limit: int  # per folder; 0 = unlimited
    quality: str      # '2K' or '4K'
    bitrate_mbps: int # 15–50
    ffmpeg_path: str
    ffprobe_path: str
    concurrency: int  # -1 = AUTO
    use_mezz: bool = True
    mezz_fps: int = 30
    mezz_qp: int = 23

QUALITY_MAP = {
    '2K': (2560, 1440),
    '4K': (3840, 2160),
}

# -------------------------------------------------
# Mezzanine helpers
# -------------------------------------------------

def sha1_of_path(p: Path) -> str:
    st = p.stat()
    h = hashlib.sha1(f"{st.st_size}-{int(st.st_mtime)}-{p.name}".encode("utf-8")).hexdigest()[:16]
    return h


def mezz_dir_for(folder: Path) -> Path:
    d = folder / "_mezz"
    ensure_dir(d)
    return d


def mezz_name_for(src: Path, w: int, h: int, fps: int, qp: int) -> str:
    tag = sha1_of_path(src)
    stem = src.stem[:40]
    return f"{stem}__{w}x{h}_{fps}fps_qp{qp}_{tag}.mp4"


def build_mezz_cmd(src: Path, dst: Path, w: int, h: int, fps: int, qp: int, cfg: JobConfig) -> List[str]:
    vf = (
        "format=nv12,"
        "hwupload_cuda,"
        f"scale_cuda=-2:{h}:interp_algo=bilinear,"
        f"pad_cuda={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    )
    return [
        cfg.ffmpeg_path, "-hide_banner", "-y", "-loglevel", "warning",
        "-i", str(src),
        "-r", str(fps),
        "-vf", vf,
        "-c:v", "h264_nvenc", "-preset", "p1", "-rc", "constqp", "-qp", str(qp),
        "-bf", "0", "-refs", "2", "-spatial_aq", "0", "-temporal_aq", "0", "-look_ahead", "0", "-multipass", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(dst)
    ]


def ensure_mezzanine_for_files(files: List[Path], folder: Path, cfg: JobConfig, w: int, h: int) -> List[Path]:
    target = mezz_dir_for(folder)
    mezz_paths: List[Path] = []
    for src in files:
        dst = target / mezz_name_for(src, w, h, cfg.mezz_fps, cfg.mezz_qp)
        if not dst.exists():
            cmd = build_mezz_cmd(src, dst, w, h, cfg.mezz_fps, cfg.mezz_qp, cfg)
            creationflags = 0
            try:
                creationflags = subprocess.CREATE_NO_WINDOW
            except Exception:
                pass
            ret = subprocess.call(cmd, creationflags=creationflags)
            if ret != 0 or not dst.exists():
                continue
        mezz_paths.append(dst)
    return mezz_paths

# -------------------------------------------------
# Encode command builders (fast-path + fallback)
# -------------------------------------------------

def _cmd_fast_hwaccel(list_file: Path, target_seconds: int, out_path: Path, cfg: JobConfig, w: int, h: int) -> List[str]:
    vf = (
        f"scale_cuda=-2:{h}:interp_algo=bilinear,"
        f"pad_cuda={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    )
    return [
        cfg.ffmpeg_path, "-hide_banner", "-y", "-loglevel", "info",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-fflags", "+genpts",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-t", str(target_seconds), "-r", "30",
        "-vf", vf,
        "-c:v", "h264_nvenc", "-preset", "p1",
        "-rc", "constqp", "-qp", "23",
        "-bf", "0", "-refs", "2", "-spatial_aq", "0", "-temporal_aq", "0", "-look_ahead", "0", "-multipass", "0",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-af", "aresample=async=1:first_pts=0",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_path)
    ]


def _cmd_fallback_cpu_upload(list_file: Path, target_seconds: int, out_path: Path, cfg: JobConfig, w: int, h: int) -> List[str]:
    vf = (
        "format=nv12,"
        "hwupload_cuda,"
        f"scale_cuda=-2:{h}:interp_algo=bilinear,"
        f"pad_cuda={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    )
    return [
        cfg.ffmpeg_path, "-hide_banner", "-y", "-loglevel", "info",
        "-fflags", "+genpts",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-t", str(target_seconds), "-r", "30",
        "-vf", vf,
        "-c:v", "h264_nvenc", "-preset", "p1",
        "-rc", "constqp", "-qp", "23",
        "-bf", "0", "-refs", "2", "-spatial_aq", "0", "-temporal_aq", "0", "-look_ahead", "0", "-multipass", "0",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-af", "aresample=async=1:first_pts=0",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_path)
    ]


def build_ffmpeg_cmds(list_file: Path, target_seconds: int, out_path: Path, cfg: JobConfig) -> List[List[str]]:
    w, h = QUALITY_MAP.get(cfg.quality, (2560, 1440))
    cmds: List[List[str]] = []
    if cfg.use_mezz:
        cmds.append(_cmd_fast_hwaccel(list_file, target_seconds, out_path, cfg, w, h))
    cmds.append(_cmd_fallback_cpu_upload(list_file, target_seconds, out_path, cfg, w, h))
    return cmds

# -------------------------------------------------
# Clip selection
# -------------------------------------------------

def write_concat_list(files: List[Path], list_path: Path):
    lines = []
    for f in files:
        p = str(f.resolve()).replace("'", "''")
        lines.append(f"file '{p}'")
    list_path.write_text("\n".join(lines), encoding='utf-8')


def select_clips_for_target(ffprobe_bin: str, folder: Path, target_seconds: int) -> tuple[List[Path], float]:
    chosen: List[Path] = []
    total = 0.0
    while total < target_seconds + 30:  # slight overfill
        batch = take_from_pool(folder, 10)
        if not batch:
            rebuild_pool(folder)
            batch = take_from_pool(folder, 10)
            if not batch:
                break
        for rel in batch:
            f = folder / rel
            if not f.exists():
                continue
            dur = get_duration_cached(ffprobe_bin, folder, f)
            if dur <= 0:
                continue
            chosen.append(f)
            total += dur
            if total >= target_seconds + 30:
                break
    return chosen, total

# -------------------------------------------------
# Engine
# -------------------------------------------------
class Engine:
    def __init__(self, cfg: JobConfig, log_widget: Optional[Text]):
        self.cfg = cfg
        self.log_widget = log_widget
        self.stop_event = threading.Event()
        self.db = AuditDB(cfg.base_folder / "_mergebot_audit.sqlite3")

    def stop(self):
        self.stop_event.set()

    def run(self):
        base = self.cfg.base_folder
        def _numeric_key(p: Path):
            m = re.search(r"\d+", p.name)
            return (int(m.group()) if m else 10**9, p.name.lower())
        folders = [p for p in base.iterdir() if p.is_dir() and not p.name.lower().endswith('_output')]
        folders.sort(key=_numeric_key)
        if not folders:
            log_gui(self.log_widget, "No subfolders found under base folder.")
            return

        # Auto-concurrency
        sem_count = self.cfg.concurrency
        if sem_count == -1:
            sem_count = pick_auto_concurrency()
        sem = threading.Semaphore(max(1, sem_count))

        threads: List[threading.Thread] = []
        for folder in folders:
            if self.stop_event.is_set():
                break
            t = threading.Thread(target=self._process_folder_thread, args=(folder, sem), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        if self.stop_event.is_set():
            log_gui(self.log_widget, "Aborted by user.")
        else:
            log_gui(self.log_widget, "All folders processed (or reached limits).")

    def _process_folder_thread(self, folder: Path, sem: threading.Semaphore):
        if self.stop_event.is_set():
            return
        with sem:
            try:
                if self.stop_event.is_set():
                    return
                self._process_folder_impl(folder)
            except Exception as e:
                log_gui(self.log_widget, f"[ERROR] {folder.name}: {e}")

    def _process_folder_impl(self, folder: Path):
        out_dir = folder.parent / f"{folder.name}_Output"
        ensure_dir(out_dir)
        reports_dir = out_dir / "reports"
        ensure_dir(reports_dir)

        per_day = self.cfg.daily_limit
        made_today = self.db.get_daily(folder) if per_day > 0 else 0
        remaining_today = max(0, per_day - made_today) if per_day > 0 else self.cfg.outputs_per_folder
        existing_count = len(list(out_dir.glob(f"{folder.name}_Video*.mp4")))
        cap_total = max(0, self.cfg.outputs_per_folder - existing_count)
        budget = min(cap_total, remaining_today) if per_day > 0 else cap_total
        if budget <= 0:
            log_gui(self.log_widget, f"{folder.name}: limit reached (existing={existing_count}, today={made_today}). Skipping.")
            return

        target_min = int(self.cfg.target_minutes)
        target_seconds = target_min * 60
        log_gui(self.log_widget, f"{folder.name}: planning {budget} outputs @ {target_min} min…")

        for _ in range(budget):
            if self.stop_event.is_set():
                return
            files, total = select_clips_for_target(self.cfg.ffprobe_path, folder, target_seconds)
            if not files:
                log_gui(self.log_widget, f"{folder.name}: no eligible clips found.")
                break

            # Mezzanine pass — huge speedup on merges
            w, h = QUALITY_MAP.get(self.cfg.quality, (2560, 1440))
            if self.cfg.use_mezz:
                mezz_files = ensure_mezzanine_for_files(files, folder, self.cfg, w, h)
                if len(mezz_files) >= 2:
                    files = mezz_files

            idx = len(list(out_dir.glob(f"{folder.name}_Video*.mp4"))) + 1
            base_name = f"{folder.name}_Video{idx:02d}"
            list_path = out_dir / f"{base_name}.list.txt"
            out_path  = out_dir / f"{base_name}.mp4"
            ff_log    = out_dir / f"{base_name}.ffmpeg.log"
            used_json = reports_dir / f"{base_name}_used.json"

            write_concat_list(files, list_path)

            cmds = build_ffmpeg_cmds(list_path, target_seconds, out_path, self.cfg)
            log_gui(self.log_widget, f"{folder.name}: encoding {out_path.name}…")

            creationflags = 0
            try:
                creationflags = subprocess.CREATE_NO_WINDOW
            except Exception:
                creationflags = 0

            chosen = False
            for ci, cmd in enumerate(cmds, start=1):
                with ff_log.open('a', encoding='utf-8') as flog:
                    flog.write(f"\n[Path {ci}/{len(cmds)}]\n")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=creationflags)
                with ff_log.open('a', encoding='utf-8') as flog:
                    while True:
                        if self.stop_event.is_set():
                            try:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=3)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                            except Exception:
                                pass
                            log_gui(self.log_widget, f"{folder.name}: aborted by user (STOP).")
                            return
                        line = proc.stdout.readline()
                        if not line:
                            break
                        flog.write(line)
                        fw = getattr(self.log_widget, 'ffmpeg_txt', None)
                        if fw is not None:
                            try:
                                fw.configure(state=NORMAL); fw.insert(END, line); fw.see(END); fw.configure(state=DISABLED)
                            except Exception:
                                pass
                ret = proc.wait()
                if ret == 0 and out_path.exists():
                    chosen = True
                    break

            if not chosen:
                log_gui(self.log_widget, f"{folder.name}: FAILED (all encode paths).")
                continue

            used = {"folder": str(folder), "output": str(out_path), "target_minutes": target_min, "files": [str(x) for x in files]}
            save_json(used_json, used)
            self.db.add_output(folder, target_min, self.cfg.quality, self.cfg.bitrate_mbps, out_path)
            if per_day > 0:
                self.db.increment_daily(folder)
            log_gui(self.log_widget, f"{folder.name}: DONE → {out_path.name}")

# -------------------------------------------------
# GPU detection (auto-concurrency)
# -------------------------------------------------

def detect_gpu_name() -> Optional[str]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        line = out.strip().splitlines()[0].strip()
        return line
    except Exception:
        return None


def pick_auto_concurrency() -> int:
    name = (detect_gpu_name() or "").lower()
    if any(k in name for k in ["a5000", "a6000", "4090", "4080", "rtx 6000", "rtx 5000", "4060"]):
        return 2
    if any(k in name for k in ["p2000", "t1000", "p1000"]):
        return 1
    return 1

# -------------------------------------------------
# GUI
# -------------------------------------------------
# Help → About / Licenses
LICENSE_ITEMS=[
    ("VideoMergeBot","© 2025 Prodit.rs, All Rights Reserved",None),
    ("FFmpeg","LGPL/GPL (depends on build)","https://www.ffmpeg.org/legal.html"),
    ("Python / Tkinter","PSF / Tk License","https://docs.python.org/3/license.html"),
    ("sqlite3","Public Domain","https://docs.python.org/3/library/sqlite3.html"),
]
ABOUT_TEXT=(
    "VideoMergeBot — Windows NVENC Automation\nVersion 1.0\n\n"
    "Merges clips into 1h/2h/3h outputs with NVENC, no-repeat shuffle,\n"
    "durations cache and daily limits.\n\nBuilt with Python + FFmpeg + Tkinter."
)


def _open_url(url: str):
    if url:
        webbrowser.open(url)


def show_about_dialog(parent):
    import tkinter as tk
    win = tk.Toplevel(parent)
    win.title("About / Licenses")
    win.transient(parent)
    win.grab_set()
    win.geometry("700x560")
    win.minsize(700, 560)

    hdr = tk.Label(win, text="About / Licenses", font=("Segoe UI", 12, "bold"))
    hdr.pack(pady=(10, 5))

    txt = tk.Text(win, height=8, wrap=tk.WORD, font=("Courier New", 10))
    txt.insert("1.0", ABOUT_TEXT)
    txt.config(state="disabled")
    txt.pack(fill=tk.X, padx=12, pady=(0, 8))

    list_frame = tk.Frame(win)
    list_frame.pack(fill=tk.BOTH, expand=True, padx=12)

    tk.Label(list_frame, text="Component", anchor="w", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
    tk.Label(list_frame, text="License", anchor="w", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w")
    tk.Label(list_frame, text="Link", anchor="w", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="w")

    for r, (name, lic, url) in enumerate(LICENSE_ITEMS, start=1):
        tk.Label(list_frame, text=name, anchor="w").grid(row=r, column=0, sticky="w", padx=(0,10), pady=2)
        tk.Label(list_frame, text=lic, anchor="w").grid(row=r, column=1, sticky="w", padx=(0,10), pady=2)
        if url:
            link = tk.Label(list_frame, text="Open", fg="blue", cursor="hand2", anchor="w")
            link.bind("<Button-1>", lambda _e, u=url: _open_url(u))
        else:
            link = tk.Label(list_frame, text="—", anchor="w")
        link.grid(row=r, column=2, sticky="w", pady=2)

    btn_frame = tk.Frame(win)
    btn_frame.pack(fill=tk.X, padx=12, pady=(12, 12))
    tk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)


class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Video Merge Bot — FFmpeg (NVENC)")
        root.geometry("920x640")

        # Variables
        self.base_folder = StringVar()
        self.target_minutes = IntVar(value=60)  # 60/120/180
        self.outputs_per_folder = IntVar(value=5)
        self.daily_limit = IntVar(value=0)  # 0 = unlimited
        self.quality = StringVar(value="2K")  # 2K/4K
        self.bitrate_mbps = IntVar(value=20)   # 15–50 (unused in constqp fast paths; kept for DB)
        self.concurrency = IntVar(value=-1)    # -1 = AUTO
        self.ffmpeg_path = StringVar(value="ffmpeg")
        self.ffprobe_path = StringVar(value="ffprobe")
        self.use_mezz = IntVar(value=1)

        # FFmpeg live log window refs (on-demand)
        self.ffmpeg_win = None
        self.ffmpeg_txt = None

        self.engine_thread: Optional[threading.Thread] = None
        self.engine: Optional[Engine] = None

        self._build_ui()

    def _build_ui(self):
        import tkinter as tk
        # Menu bar (Help → About / Licenses)
        menubar = tk.Menu(self.root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About / Licenses…", command=lambda: show_about_dialog(self.root))
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill='both', expand=True)

        ttk.Label(frm, text="Base Folder:").grid(row=0, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.base_folder, width=70).grid(row=0, column=1, sticky='we', padx=6)
        ttk.Button(frm, text="Browse…", command=self._choose_base).grid(row=0, column=2, sticky='w')

        ttk.Label(frm, text="Duration (min):").grid(row=1, column=0, sticky='w')
        ttk.Combobox(frm, textvariable=self.target_minutes, values=[60, 120, 180], width=10, state='readonly').grid(row=1, column=1, sticky='w')

        ttk.Label(frm, text="Outputs/Folder:").grid(row=1, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.outputs_per_folder, width=8).grid(row=1, column=2, sticky='w')

        ttk.Label(frm, text="Daily Limit/Folder (0=∞):").grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.daily_limit, width=10).grid(row=2, column=1, sticky='w')

        ttk.Label(frm, text="Quality:").grid(row=3, column=0, sticky='w')
        ttk.Combobox(frm, textvariable=self.quality, values=["2K", "4K"], width=10, state='readonly').grid(row=3, column=1, sticky='w')

        ttk.Label(frm, text="Video Bitrate (Mb/s):").grid(row=3, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.bitrate_mbps, width=8).grid(row=3, column=2, sticky='w')

        ttk.Label(frm, text="Concurrency (-1=AUTO):").grid(row=4, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.concurrency, width=8).grid(row=4, column=2, sticky='w')

        ttk.Checkbutton(frm, text="Use Mezzanine Cache", variable=self.use_mezz).grid(row=5, column=0, sticky='w')

        ttk.Label(frm, text="ffmpeg path:").grid(row=6, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ffmpeg_path, width=40).grid(row=6, column=1, sticky='w')
        ttk.Label(frm, text="ffprobe path:").grid(row=6, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.ffprobe_path, width=40).grid(row=6, column=2, sticky='w')

        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=3, pady=(10,5), sticky='w')
        ttk.Button(btns, text="START", command=self.start).pack(side='left', padx=(0,8))
        ttk.Button(btns, text="STOP", command=self.stop).pack(side='left')
        ttk.Button(btns, text="FFmpeg Log", command=self._open_ffmpeg_log).pack(side='left', padx=(8,0))
        ttk.Button(btns, text="📊 Show Stats", command=self._show_stats).pack(side='left', padx=(8,0))

        self.log = Text(frm, height=18, state=DISABLED)
        self.log.grid(row=8, column=0, columnspan=3, sticky='nsew', pady=(10,0))

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(8, weight=1)

    def _choose_base(self):
        d = filedialog.askdirectory(title="Select Base Folder")
        if d:
            self.base_folder.set(d)

    def start(self):
        if self.engine_thread and self.engine_thread.is_alive():
            log_gui(self.log, "Already running…")
            return
        base = Path(self.base_folder.get().strip())
        if not base.exists():
            log_gui(self.log, "Please select a valid Base Folder.")
            return
        conc = int(self.concurrency.get())
        if conc == -1:
            conc = pick_auto_concurrency()
        cfg = JobConfig(
            base_folder=base,
            target_minutes=int(self.target_minutes.get()),
            outputs_per_folder=max(1, int(self.outputs_per_folder.get())),
            daily_limit=max(0, int(self.daily_limit.get())),
            quality=self.quality.get(),
            bitrate_mbps=max(15, min(50, int(self.bitrate_mbps.get()))),
            ffmpeg_path=self.ffmpeg_path.get(),
            ffprobe_path=self.ffprobe_path.get(),
            concurrency=max(1, conc),
            use_mezz=bool(self.use_mezz.get()),
            mezz_fps=30,
            mezz_qp=23
        )
        self.engine = Engine(cfg, self.log)
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=True)
        self.engine_thread.start()
        log_gui(self.log, "Started.")

    def stop(self):
        if self.engine:
            self.engine.stop()
            log_gui(self.log, "Stop requested. Waiting for workers to finish…")

    def _open_ffmpeg_log(self, clear: bool = False):
        if self.ffmpeg_win is None or not self.ffmpeg_win.winfo_exists():
            self.ffmpeg_win = Toplevel(self.root)
            self.ffmpeg_win.title("FFmpeg Output")
            self.ffmpeg_win.geometry("900x400")
            self.ffmpeg_txt = Text(self.ffmpeg_win, wrap='none', state=DISABLED)
            self.ffmpeg_txt.pack(fill='both', expand=True)
        if clear and self.ffmpeg_txt is not None:
            self.ffmpeg_txt.configure(state=NORMAL)
            self.ffmpeg_txt.delete('1.0', END)
            self.ffmpeg_txt.configure(state=DISABLED)
        try:
            self.log.ffmpeg_txt = self.ffmpeg_txt
        except Exception:
            pass

    def _show_stats(self):
        try:
            base = Path(self.base_folder.get().strip())
        except Exception:
            base = None
        if not base or not base.exists():
            log_gui(self.log, "Select a valid Base Folder first.")
            return
        db_path = base / "_mergebot_audit.sqlite3"
        if not db_path.exists():
            log_gui(self.log, "No stats yet — database not found.")
            return

        win = Toplevel(self.root)
        win.title("Merge Bot — Stats")
        win.geometry("1000x520")
        nb = ttk.Notebook(win)
        nb.pack(fill='both', expand=True)

        frm_out = ttk.Frame(nb)
        nb.add(frm_out, text="Outputs")
        tv_out = ttk.Treeview(
            frm_out,
            columns=("created_at","folder","target","quality","bitrate","result"),
            show='headings'
        )
        for c, w in ("created_at",180), ("folder",220), ("target",80), ("quality",70), ("bitrate",80), ("result",420):
            tv_out.heading(c, text=c)
            tv_out.column(c, width=w, anchor='w')
        tv_out.pack(fill='both', expand=True)

        frm_day = ttk.Frame(nb)
        nb.add(frm_day, text="Daily count")
        tv_day = ttk.Treeview(
            frm_day,
            columns=("day","folder","count"),
            show='headings'
        )
        for c, w in ("day",120), ("folder",420), ("count",80):
            tv_day.heading(c, text=c)
            tv_day.column(c, width=w, anchor='w')
        tv_day.pack(fill='both', expand=True)

        import sqlite3 as _sql
        try:
            conn = _sql.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT created_at, folder, target_minutes, quality, bitrate_mbps, result_path
                FROM outputs
                ORDER BY created_at DESC
                LIMIT 1000
            """)
            for row in cur.fetchall():
                created_at, folder, target, quality, bitrate, result = row
                tv_out.insert('', 'end', values=(created_at, folder, target, quality, bitrate, result))
            cur.execute("""
                SELECT day, folder, count
                FROM daily_count
                ORDER BY day DESC, folder ASC
                LIMIT 1000
            """)
            for row in cur.fetchall():
                tv_day.insert('', 'end', values=row)
        except Exception as e:
            log_gui(self.log, f"Stats error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == '__main__':
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = Tk()
    app = App(root)
    root.mainloop()
