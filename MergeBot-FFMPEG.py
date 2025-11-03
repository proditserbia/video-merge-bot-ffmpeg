#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Merge Bot — FilterGraph Edition (v4, global job queue)
Author: prodit.rs

What’s new (v4):
• Global JOB QUEUE: concurrency = broj paralelnih izlaza (bez obzira na folder)
• STOP = hard kill svih aktivnih ffmpeg procesa, odmah
• Ignoriše _BadInputs folder
• Duration (min) je ručni unos (default 90)
• Kompatibilno sa starijim Python-om (bez PEP604 | hintova)
"""

import re
import json
import uuid
import random
import sqlite3
import threading
import subprocess
import queue
from dataclasses import dataclass
from datetime import datetime, date
import sys, webbrowser, os, signal
from pathlib import Path
from typing import Optional, List, Dict, Tuple

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
    try:
        w.configure(state=NORMAL)
        w.insert(END, line)
        w.see(END)
        w.configure(state=DISABLED)
    except Exception:
        print(line, end="")

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

def _nowindow_kwargs():
    try:
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    except Exception:
        return {}

# -------------------------------------------------
# FFprobe duration cache (durations.json)
# -------------------------------------------------

def ffprobe_duration_seconds(ffprobe_bin: str, file_path: Path) -> float:
    try:
        cmd = [ffprobe_bin, "-v", "error",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1",
               str(file_path)]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **_nowindow_kwargs())
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0

def get_duration_cached(ffprobe_bin: str, folder: Path, f: Path) -> float:
    cache_path = folder / "durations.json"
    cache = load_json(cache_path, {})
    key = f.name
    try:
        mtime = str(f.stat().st_mtime)
    except FileNotFoundError:
        return 0.0
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
    clips: List[str] = []
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
# Input validator: multi-spot sniff + optional remux; quarantine on failure
# -------------------------------------------------

QUARANTINE_DIRNAME = "_BadInputs"

def _ff_run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **_nowindow_kwargs())
    return p.returncode, (p.stdout or "")

def _dur_seconds(ffprobe_bin: str, f: Path) -> float:
    try:
        return ffprobe_duration_seconds(ffprobe_bin, f)
    except Exception:
        return 0.0

def multi_spot_decode_ok(ffmpeg_bin: str, ffprobe_bin: Optional[str], f: Path, sample_sec: int = 2) -> bool:
    spots: List[List[str]] = []
    dur = 0.0
    if ffprobe_bin:
        dur = _dur_seconds(ffprobe_bin, f)
    # start
    spots.append([ffmpeg_bin, "-v", "error", "-xerror", "-err_detect", "explode",
                  "-ss", "0", "-t", str(sample_sec), "-i", str(f),
                  "-an", "-f", "null", "-"])
    # middle
    if dur > (sample_sec * 4):
        mid = max(0, int(dur/2) - sample_sec//2)
        spots.append([ffmpeg_bin, "-v", "error", "-xerror", "-err_detect", "explode",
                      "-ss", str(mid), "-t", str(sample_sec), "-i", str(f),
                      "-an", "-f", "null", "-"])
    # tail
    spots.append([ffmpeg_bin, "-v", "error", "-xerror", "-err_detect", "explode",
                  "-sseof", f"-{sample_sec}", "-t", str(sample_sec), "-i", str(f),
                  "-an", "-f", "null", "-"])
    for cmd in spots:
        rc, _out = _ff_run(cmd)
        if rc != 0:
            return False
    return True

def try_remux(ffmpeg_bin: str, src: Path, dst: Path) -> bool:
    cmd = [
        ffmpeg_bin, "-y",
        "-err_detect", "ignore_err", "-fflags", "+discardcorrupt",
        "-i", str(src), "-map", "0",
        "-c", "copy", "-movflags", "+faststart",
        str(dst)
    ]
    rc, _ = _ff_run(cmd)
    return rc == 0 and dst.exists() and dst.stat().st_size > 0

def quarantine_bad_clip(f: Path, folder: Path, reason: str, log_cb=None) -> Path:
    qdir = folder / QUARANTINE_DIRNAME
    ensure_dir(qdir)
    target = qdir / f.name
    try:
        f.replace(target)
    except Exception:
        try:
            target.write_bytes(f.read_bytes())
            f.unlink(missing_ok=True)
        except Exception:
            pass
    if log_cb:
        log_cb(f"[QUARANTINE] {f.name} → {target} ({reason})")
    return target

def sanitize_or_skip(ffmpeg_bin: str, ffprobe_bin: str, f: Path, folder: Path, log_cb=None) -> Optional[Path]:
    try:
        dur = ffprobe_duration_seconds(ffprobe_bin, f)
        if dur <= 0:
            quarantine_bad_clip(f, folder, "duration<=0", log_cb)
            return None
        if multi_spot_decode_ok(ffmpeg_bin, ffprobe_bin, f, sample_sec=2):
            return f
        fixed = f.with_suffix(f.suffix + ".remux.mp4") if f.suffix.lower() != ".mp4" else f.with_name(f.stem + ".remux.mp4")
        if try_remux(ffmpeg_bin, f, fixed) and multi_spot_decode_ok(ffmpeg_bin, ffprobe_bin, fixed, sample_sec=2):
            if log_cb:
                log_cb(f"[FIX] {f.name} → using remux: {fixed.name}")
            return fixed
        quarantine_bad_clip(f, folder, "decode-fail (after remux)", log_cb)
        try:
            if fixed.exists():
                fixed.unlink()
        except Exception:
            pass
        return None
    except Exception as e:
        quarantine_bad_clip(f, folder, f"exception: {e}", log_cb)
        return None

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
# Config & common
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
    concurrency: int

QUALITY_MAP = {
    '2K': (1440, 2560),
    '4K': (2160, 3840),
}

def _is_media_folder(p: Path) -> bool:
    exts = {".mp4", ".mov", ".mkv", ".m4v", ".avi"}
    try:
        return any(x.is_file() and x.suffix.lower() in exts for x in p.iterdir())
    except Exception:
        return False

IN_OPTS = [
    "-fflags", "+genpts",
    "-thread_queue_size", "4096",
    "-probesize", "200M",
    "-analyzeduration", "200M",
]

def write_concat_list(files: List[Path], list_path: Path):
    lines: List[str] = []
    for f in files:
        p = str(f.resolve()).replace("'", "''")
        lines.append(f"file '{p}'")
    list_path.write_text("\n".join(lines), encoding='utf-8')

def preflight_concat(ffmpeg_bin: str, list_file: Path) -> bool:
    cmd = [
        ffmpeg_bin, "-v", "error", "-xerror",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-c", "copy", "-f", "null", "-"
    ]
    rc, _out = _ff_run(cmd)
    return rc == 0

# -------------------------------------------------
# Filtergraph cmd builder (ROBUST CONCAT)
# -------------------------------------------------
# --- target za true vertical ---
VERT_W, VERT_H = 1440, 2560

def build_ffmpeg_cmd_filtergraph(files: List[Path], target_seconds: int, out_path: Path, cfg: JobConfig) -> List[str]:
    # IGNORIŠEMO QUALITY_MAP za ovaj mod i forsiramo 1440x1920
    w, h = VERT_W, VERT_H

    vb_m = max(15, min(50, int(cfg.bitrate_mbps)))
    vb = f"{vb_m}M"
    maxrate = vb
    bufsize = f"{vb_m * 2}M"

    cmd: List[str] = [cfg.ffmpeg_path, "-hide_banner", "-y", "-loglevel", "info", *IN_OPTS]
    for f in files:
        cmd += ["-i", str(f)]

    # COVER logika: skaliranje do "prekrivanja" + crop na tačno 1080x2560.
    # Ovo uklanja sve crne trake, ali kod izvora koji nisu tačno 9:16, biće blagi crop gore/dole ili levo/desno.
    vf_steps = (
        f"fps=30,"
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},"
        f"format=yuv420p,setsar=1:1,setpts=PTS-STARTPTS"
    )

    af_steps = (
        "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        "aresample=48000:async=1:min_comp=0.001:first_pts=0,asetpts=PTS-STARTPTS"
    )

    chains: List[str] = []
    for i in range(len(files)):
        chains.append(f"[{i}:v]{vf_steps}[v{i}]")
        chains.append(f"[{i}:a]{af_steps}[a{i}]")

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(files)))
    concat = f"{concat_inputs}concat=n={len(files)}:v=1:a=1[v][a]"
    filter_complex = ";".join(chains + [concat])

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-t", str(target_seconds),
        "-vsync", "1",
        "-max_muxing_queue_size", "4096",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-c:v", "h264_nvenc", "-preset", "p1",
        "-rc", "cbr_hq", "-b:v", vb, "-maxrate", maxrate, "-bufsize", bufsize,
        "-g", "60",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart",
        "-dn",
        str(out_path)
    ]
    return cmd

# -------------------------------------------------
# Selection (thread-safe po folderu)
# -------------------------------------------------
_folder_locks: Dict[str, threading.Lock] = {}

def _get_folder_lock(folder: Path) -> threading.Lock:
    key = str(folder.resolve())
    if key not in _folder_locks:
        _folder_locks[key] = threading.Lock()
    return _folder_locks[key]

def select_clips_for_target(ffmpeg_bin: str, ffprobe_bin: str, folder: Path, target_seconds: int, log_cb=None) -> Tuple[List[Path], float]:
    """Gradi listu sanitizovanih klipova do targeta (+30s overfill). Zaključava pool.json po folderu."""
    lock = _get_folder_lock(folder)
    chosen: List[Path] = []
    total = 0.0
    with lock:
        while total < target_seconds + 30:
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
                valid_path = sanitize_or_skip(ffmpeg_bin, ffprobe_bin, f, folder, log_cb=log_cb)
                if valid_path is None:
                    continue
                dur = get_duration_cached(ffprobe_bin, folder, valid_path)
                if dur <= 0:
                    quarantine_bad_clip(valid_path, folder, "duration<=0 post-remux", log_cb)
                    continue
                chosen.append(valid_path)
                total += dur
                if total >= target_seconds + 30:
                    break
    return chosen, total

# -------------------------------------------------
# Engine with GLOBAL JOB QUEUE
# -------------------------------------------------
@dataclass
class Job:
    folder: Path
    out_index: int
    target_seconds: int

class Engine:
    def __init__(self, cfg: JobConfig, log_widget: Optional[Text]):
        self.cfg = cfg
        self.log_widget = log_widget
        self.stop_event = threading.Event()
        self.db = AuditDB(cfg.base_folder / "_mergebot_audit.sqlite3")
        self.job_queue: "queue.Queue[Job]" = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.active_procs: List[subprocess.Popen] = []
        self.active_lock = threading.Lock()

    def _register_proc(self, p: subprocess.Popen):
        with self.active_lock:
            self.active_procs.append(p)

    def _unregister_proc(self, p: subprocess.Popen):
        with self.active_lock:
            try:
                self.active_procs.remove(p)
            except ValueError:
                pass

    def _kill_all_procs(self):
        with self.active_lock:
            procs = list(self.active_procs)
        for p in procs:
            try:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()
        self._kill_all_procs()
        # isprazni queue
        try:
            while True:
                self.job_queue.get_nowait()
        except queue.Empty:
            pass

    def _list_folders(self) -> List[Path]:
        base = self.cfg.base_folder
        def _numeric_key(p: Path):
            m = re.search(r"\d+", p.name)
            return (int(m.group()) if m else 10**9, p.name.lower())
        # ignoriši _output i _BadInputs
        folders = [p for p in base.iterdir()
                   if p.is_dir()
                   and not p.name.lower().endswith('_output')
                   and p.name != QUARANTINE_DIRNAME]
        folders.sort(key=_numeric_key)
        if not folders and _is_media_folder(base):
            log_gui(self.log_widget, f"No subfolders detected — using Base folder as source: {base}")
            folders = [base]
        return folders

    def _plan_jobs(self, folders: List[Path]) -> int:
        total_jobs = 0
        for folder in folders:
            per_day = self.cfg.daily_limit
            made_today = self.db.get_daily(folder) if per_day > 0 else 0
            remaining_today = max(0, per_day - made_today) if per_day > 0 else self.cfg.outputs_per_folder

            out_dir = folder.parent / f"{folder.name}_Output"
            ensure_dir(out_dir)
            existing_count = len(list(out_dir.glob(f"{folder.name}_Video*.mp4")))
            cap_total = max(0, self.cfg.outputs_per_folder - existing_count)
            budget = min(cap_total, remaining_today) if per_day > 0 else cap_total
            if budget <= 0:
                log_gui(self.log_widget, f"{folder.name}: limit reached (existing={existing_count}, today={made_today}). Skipping.")
                continue

            target_min = int(self.cfg.target_minutes)
            log_gui(self.log_widget, f"{folder.name}: planning {budget} outputs @ {target_min} min (concurrency={self.cfg.concurrency})…")

            # ubaci poslove u red (po jedan output)
            for i in range(existing_count + 1, existing_count + 1 + budget):
                self.job_queue.put(Job(folder=folder, out_index=i, target_seconds=target_min * 60))
                total_jobs += 1
        return total_jobs

    def run(self):
        folders = self._list_folders()
        if not folders:
            log_gui(self.log_widget, "No subfolders (or media files) found under base folder.")
            return

        total = self._plan_jobs(folders)
        if total == 0:
            log_gui(self.log_widget, "All folders processed (or reached limits).")
            return

        # start workers
        for _ in range(self.cfg.concurrency):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)

        # čekaj dok se posao ne potroši
        while any(t.is_alive() for t in self.workers):
            if self.stop_event.is_set():
                break
            try:
                # ako je queue prazan i threadovi su se ugasili, izlazimo
                if self.job_queue.empty():
                    alive = any(t.is_alive() for t in self.workers)
                    if not alive:
                        break
                threading.Event().wait(0.2)
            except KeyboardInterrupt:
                self.stop()
                break

        if self.stop_event.is_set():
            log_gui(self.log_widget, "Aborted by user.")
        else:
            log_gui(self.log_widget, "All folders processed (or reached limits).")

    # ---------------- Worker ----------------

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                # nema poslova i verovatno ćemo uskoro završiti
                return
            try:
                if self.stop_event.is_set():
                    return
                self._do_job(job)
            finally:
                self.job_queue.task_done()

    def _do_job(self, job: Job):
        folder = job.folder
        out_dir = folder.parent / f"{folder.name}_Output"
        ensure_dir(out_dir)
        reports_dir = out_dir / "reports"
        ensure_dir(reports_dir)

        def _l(msg: str):
            log_gui(self.log_widget, f"{folder.name}: {msg}")

        # odaberi klipove
        files, total = select_clips_for_target(
            self.cfg.ffmpeg_path, self.cfg.ffprobe_path, folder, job.target_seconds, log_cb=_l
        )
        if not files:
            log_gui(self.log_widget, f"{folder.name}: no eligible clips found.")
            return

        # preflight
        tmp_list = out_dir / "_preflight.list.txt"
        write_concat_list(files, tmp_list)
        if not preflight_concat(self.cfg.ffmpeg_path, tmp_list):
            log_gui(self.log_widget, f"{folder.name}: preflight failed → isolating bad clip…")
            bad = self._bisect_bad_clip(files, tmp_list)
            if bad is not None:
                quarantine_bad_clip(Path(bad), folder, "preflight-fail",
                                    log_cb=lambda m: log_gui(self.log_widget, f"{folder.name}: {m}"))
                files = [p for p in files if str(p) != bad]
                if not files:
                    log_gui(self.log_widget, f"{folder.name}: no eligible clips after isolation.")
                    try: tmp_list.unlink(missing_ok=True)
                    except Exception: pass
                    return
            else:
                log_gui(self.log_widget, f"{folder.name}: could not isolate bad clip.")
        try:
            tmp_list.unlink(missing_ok=True)
        except Exception:
            pass

        base_name = f"{folder.name}_Video{job.out_index:02d}"
        out_path  = out_dir / f"{base_name}.mp4"
        ff_log    = out_dir / f"{base_name}.ffmpeg.log"
        used_json = reports_dir / f"{base_name}_used.json"

        cmd = build_ffmpeg_cmd_filtergraph(files, job.target_seconds, out_path, self.cfg)
        log_gui(self.log_widget, f"{folder.name}: encoding {out_path.name}…")

        with ff_log.open('w', encoding='utf-8') as flog:
            flog.write(" ")

        try:
            creationflags = subprocess.CREATE_NO_WINDOW
        except Exception:
            creationflags = 0

        aborted = False
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            creationflags=creationflags
        )
        self._register_proc(proc)

        try:
            with ff_log.open('a', encoding='utf-8') as flog:
                while True:
                    if self.stop_event.is_set():
                        aborted = True
                        try:
                            proc.terminate()
                            try:
                                proc.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                        except Exception:
                            pass
                        break
                    line = proc.stdout.readline()
                    if not line:
                        break
                    flog.write(line)
                    try:
                        fw = getattr(self.log_widget, 'ffmpeg_txt', None)
                        if fw is not None:
                            fw.configure(state=NORMAL)
                            fw.insert(END, line)
                            fw.see(END)
                            fw.configure(state=DISABLED)
                    except Exception:
                        pass
            ret = proc.wait()
        finally:
            self._unregister_proc(proc)

        if aborted or self.stop_event.is_set():
            log_gui(self.log_widget, f"{folder.name}: aborted by user (STOP).")
            return

        if (ret != 0 or not out_path.exists()) and not self.stop_event.is_set():
            log_gui(self.log_widget, f"{folder.name}: FFmpeg failed (code {ret}). Retrying with extra margin…")
            # dodaj još klipova za marginu i probaj ponovo
            files2, _ = select_clips_for_target(
                self.cfg.ffmpeg_path, self.cfg.ffprobe_path, folder, job.target_seconds + 120, log_cb=_l
            )
            files = files + files2
            cmd = build_ffmpeg_cmd_filtergraph(files, job.target_seconds, out_path, self.cfg)
            with ff_log.open('a', encoding='utf-8') as flog:
                flog.write("\n[Retry]\n")

            proc2 = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
                creationflags=creationflags
            )
            self._register_proc(proc2)
            try:
                with ff_log.open('a', encoding='utf-8') as flog:
                    for line in proc2.stdout:
                        flog.write(line)
                ret = proc2.wait()
            finally:
                self._unregister_proc(proc2)

            if self.stop_event.is_set():
                log_gui(self.log_widget, f"{folder.name}: aborted by user (STOP) after retry request.")
                return

        if ret == 0 and out_path.exists():
            used = {"folder": str(folder), "output": str(out_path), "target_minutes": int(self.cfg.target_minutes),
                    "files": [str(x) for x in files]}
            save_json(used_json, used)
            # increment DB
            per_day = self.cfg.daily_limit
            self.db.add_output(folder, int(self.cfg.target_minutes), self.cfg.quality, self.cfg.bitrate_mbps, out_path)
            if per_day > 0:
                self.db.increment_daily(folder)
            log_gui(self.log_widget, f"{folder.name}: DONE → {out_path.name}")
        else:
            log_gui(self.log_widget, f"{folder.name}: FAILED after retry: {out_path.name}")

    # --- helpers for preflight & bisect ---

    def _check_list_null(self, list_file: Path) -> bool:
        return preflight_concat(self.cfg.ffmpeg_path, list_file)

    def _bisect_bad_clip(self, files: List[Path], list_path: Path) -> Optional[str]:
        arr = [str(p) for p in files]
        write_concat_list([Path(p) for p in arr], list_path)
        if self._check_list_null(list_path):
            return None
        lo, hi = 0, len(arr) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            left = arr[lo:mid+1]
            right = arr[mid+1:hi+1]
            write_concat_list([Path(p) for p in left], list_path)
            left_ok = self._check_list_null(list_path)
            write_concat_list([Path(p) for p in right], list_path)
            right_ok = self._check_list_null(list_path)
            if not left_ok and len(left) == 1:
                return left[0]
            if not right_ok and len(right) == 1:
                return right[0]
            if not left_ok:
                hi = mid
            elif not right_ok:
                lo = mid + 1
            else:
                # oba OK → problem kombinacije; fallback linear
                for p in (left + right):
                    write_concat_list([Path(p) for p in [p]], list_path)
                    if not self._check_list_null(list_path):
                        return p
                return None
        return arr[lo]

# -------------------------------------------------
# GUI
# -------------------------------------------------

LICENSE_ITEMS = [
    ("VideoMergeBot","© 2025 Prodit.rs, All Rights Reserved",None),
    ("FFmpeg","LGPL/GPL (depends on build)","https://www.ffmpeg.org/legal.html"),
    ("Python / Tkinter","PSF / Tk License","https://docs.python.org/3/license.html"),
    ("sqlite3","Public Domain","https://docs.python.org/3/library/sqlite3.html"),
]

ABOUT_TEXT = (
    "VideoMergeBot — Windows NVENC Automation (FilterGraph Edition)\n"
    "Version 1.5 (global job queue + hard stop)\n\n"
    "Merges clips into long outputs with NVENC, per-input normalize chains,\n"
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
        root.title("Video Merge Bot — FFmpeg")
        root.geometry("900x620")

        # Variables
        self.base_folder = StringVar()
        self.target_minutes = IntVar(value=90)  # default 90
        self.outputs_per_folder = IntVar(value=5)
        self.daily_limit = IntVar(value=0)
        self.quality = StringVar(value="2K")
        self.bitrate_mbps = IntVar(value=20)
        self.concurrency = IntVar(value=2)
        self.ffmpeg_path = StringVar(value="ffmpeg")
        self.ffprobe_path = StringVar(value="ffprobe")

        self.ffmpeg_win = None
        self.ffmpeg_txt = None

        self.engine_thread: Optional[threading.Thread] = None
        self.engine: Optional[Engine] = None

        self._build_ui()

    def _build_ui(self):
        import tkinter as tk
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
        ttk.Entry(frm, textvariable=self.target_minutes, width=10).grid(row=1, column=1, sticky='w')

        ttk.Label(frm, text="Outputs/Folder:").grid(row=1, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.outputs_per_folder, width=8).grid(row=1, column=2, sticky='w')

        ttk.Label(frm, text="Daily Limit/Folder (0=∞):").grid(row=2, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.daily_limit, width=10).grid(row=2, column=1, sticky='w')

        ttk.Label(frm, text="Quality:").grid(row=3, column=0, sticky='w')
        ttk.Combobox(frm, textvariable=self.quality, values=["2K", "4K"], width=10, state='readonly').grid(row=3, column=1, sticky='w')

        ttk.Label(frm, text="Video Bitrate (Mb/s):").grid(row=3, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.bitrate_mbps, width=8).grid(row=3, column=2, sticky='w')

        ttk.Label(frm, text="Concurrency:").grid(row=4, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.concurrency, width=8).grid(row=4, column=2, sticky='w')

        ttk.Label(frm, text="ffmpeg path:").grid(row=5, column=0, sticky='w')
        ttk.Entry(frm, textvariable=self.ffmpeg_path, width=40).grid(row=5, column=1, sticky='w')
        ttk.Label(frm, text="ffprobe path:").grid(row=5, column=1, sticky='e')
        ttk.Entry(frm, textvariable=self.ffprobe_path, width=40).grid(row=5, column=2, sticky='w')

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=3, pady=(10,5), sticky='w')
        ttk.Button(btns, text="START", command=self.start).pack(side='left', padx=(0,8))
        ttk.Button(btns, text="STOP (Kill All)", command=self.stop).pack(side='left')
        ttk.Button(btns, text="FFmpeg Log", command=self._open_ffmpeg_log).pack(side='left', padx=(8,0))
        ttk.Button(btns, text="📊 Show Stats", command=self._show_stats).pack(side='left', padx=(8,0))

        self.log = Text(frm, height=18, state=DISABLED)
        self.log.grid(row=7, column=0, columnspan=3, sticky='nsew', pady=(10,0))

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(7, weight=1)

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
        cfg = JobConfig(
            base_folder=base,
            target_minutes=max(1, int(self.target_minutes.get())),
            outputs_per_folder=max(1, int(self.outputs_per_folder.get())),
            daily_limit=max(0, int(self.daily_limit.get())),
            quality=self.quality.get(),
            bitrate_mbps=max(15, min(50, int(self.bitrate_mbps.get()))),
            ffmpeg_path=self.ffmpeg_path.get(),
            ffprobe_path=self.ffprobe_path.get(),
            concurrency=max(1, int(self.concurrency.get()))
        )
        self.engine = Engine(cfg, self.log)
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=True)
        self.engine_thread.start()
        log_gui(self.log, "Started.")

    def stop(self):
        if self.engine:
            self.engine.stop()
            log_gui(self.log, "STOP issued — all ffmpeg processes terminated.")

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
