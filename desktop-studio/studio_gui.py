#!/usr/bin/env python3
"""
SmartScout AI 棒球測速工作站 (Windows Desktop GUI)
--------------------------------------------------
雙擊桌面捷徑即可啟動，支援：
1. 選擇單支/多支影片或整個牛棚資料夾，批次產出 SmartScout 3D 軌跡＋8項數據卡片 MP4 與 CSV 報表
2. 右上角「檢查並更新」一鍵從 GitHub 同步最新程式與最新訓練好的 AI 棒球模型
3. 一鍵開啟成品資料夾或網頁版測速
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

STUDIO_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDIO_DIR.parent
ENGINE_SCRIPT = STUDIO_DIR / "smartscout_studio.py"
OUTPUT_DIR = STUDIO_DIR / "output"
ICON_PATH = STUDIO_DIR / "app_icon.ico"
WEB_URL = "https://s87ngsr5f5-dotcom.github.io/pitch-speed-web/"

BG_DARK = "#0b0f17"
CARD_BG = "#141b29"
BORDER = "#243047"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_RED = "#ef4444"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#10b981"


def get_python_exe() -> str:
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def get_git_short_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip()
        return out or "local"
    except Exception:
        return "local"


class SmartScoutStudioApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SmartScout AI 棒球測速工作站")
        self.geometry("860x660")
        self.minsize(780, 580)
        self.configure(bg=BG_DARK)
        if ICON_PATH.exists():
            try:
                self.iconbitmap(str(ICON_PATH))
            except Exception:
                pass

        self.selected_paths: list[str] = []
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker_proc: subprocess.Popen | None = None
        self.is_running = False

        self.distance_var = tk.StringVar(value="18.44")
        self.angle_var = tk.StringVar(value="behind_pitcher")
        self.slowmo_var = tk.StringVar(value="1.0")
        self.prefer640_var = tk.BooleanVar(value=True)
        self.version_var = tk.StringVar(value=f"版本：{get_git_short_rev()}")

        self._build_ui()
        self.after(120, self._poll_logs)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.TCombobox", fieldbackground="#1e293b", background="#1e293b", foreground=TEXT_MAIN)

        # Header
        header = tk.Frame(self, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        header.pack(fill="x", padx=16, pady=(14, 10))

        title_box = tk.Frame(header, bg=CARD_BG)
        title_box.pack(side="left", padx=16, pady=12)
        tk.Label(
            title_box,
            text="⚾ SmartScout AI 棒球測速工作站",
            font=("Microsoft JhengHei UI", 16, "bold"),
            fg=TEXT_MAIN,
            bg=CARD_BG,
        ).pack(anchor="w")
        tk.Label(
            title_box,
            textvariable=self.version_var,
            font=("Microsoft JhengHei UI", 9),
            fg=TEXT_MUTED,
            bg=CARD_BG,
        ).pack(anchor="w")

        btn_box = tk.Frame(header, bg=CARD_BG)
        btn_box.pack(side="right", padx=14, pady=12)

        self.update_btn = tk.Button(
            btn_box,
            text="🔄 檢查並更新",
            font=("Microsoft JhengHei UI", 10, "bold"),
            bg="#1d4ed8",
            fg=TEXT_MAIN,
            activebackground="#2563eb",
            activeforeground=TEXT_MAIN,
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            command=self._start_update,
        )
        self.update_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            btn_box,
            text="🌐 網頁版",
            font=("Microsoft JhengHei UI", 10),
            bg="#1e293b",
            fg=TEXT_MAIN,
            activebackground="#334155",
            activeforeground=TEXT_MAIN,
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
            command=lambda: webbrowser.open(WEB_URL),
        ).pack(side="left")

        # Input selection card
        in_card = tk.Frame(self, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        in_card.pack(fill="x", padx=16, pady=(0, 10))

        top_row = tk.Frame(in_card, bg=CARD_BG)
        top_row.pack(fill="x", padx=14, pady=(12, 8))
        tk.Label(
            top_row,
            text="1. 選擇要測速與產片的影片 / 資料夾",
            font=("Microsoft JhengHei UI", 11, "bold"),
            fg=TEXT_MAIN,
            bg=CARD_BG,
        ).pack(side="left")

        tk.Button(
            top_row,
            text="🧹 清空",
            font=("Microsoft JhengHei UI", 9),
            bg="#334155",
            fg=TEXT_MAIN,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._clear_files,
        ).pack(side="right")

        tk.Button(
            top_row,
            text="📁 選取整個資料夾",
            font=("Microsoft JhengHei UI", 9, "bold"),
            bg="#0f766e",
            fg=TEXT_MAIN,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._add_folder,
        ).pack(side="right", padx=(0, 8))

        tk.Button(
            top_row,
            text="🎬 選取影片檔 (可多選)",
            font=("Microsoft JhengHei UI", 9, "bold"),
            bg="#0284c7",
            fg=TEXT_MAIN,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._add_files,
        ).pack(side="right", padx=(0, 8))

        self.file_listbox = tk.Listbox(
            in_card,
            height=5,
            bg="#0f172a",
            fg=TEXT_MAIN,
            selectbackground="#1d4ed8",
            relief="flat",
            font=("Consolas", 9),
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.file_listbox.pack(fill="x", padx=14, pady=(0, 12))

        # Settings card
        cfg_card = tk.Frame(self, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        cfg_card.pack(fill="x", padx=16, pady=(0, 10))

        cfg_inner = tk.Frame(cfg_card, bg=CARD_BG)
        cfg_inner.pack(fill="x", padx=14, pady=10)

        tk.Label(cfg_inner, text="投捕距離 (m):", font=("Microsoft JhengHei UI", 9), fg=TEXT_MUTED, bg=CARD_BG).pack(side="left")
        tk.Entry(
            cfg_inner,
            textvariable=self.distance_var,
            width=7,
            bg="#0f172a",
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            relief="flat",
            font=("Consolas", 10),
        ).pack(side="left", padx=(6, 16))

        tk.Label(cfg_inner, text="拍攝視角:", font=("Microsoft JhengHei UI", 9), fg=TEXT_MUTED, bg=CARD_BG).pack(side="left")
        angle_cb = ttk.Combobox(
            cfg_inner,
            textvariable=self.angle_var,
            values=["behind_pitcher", "behind_catcher"],
            state="readonly",
            width=15,
            style="Dark.TCombobox",
        )
        angle_cb.pack(side="left", padx=(6, 16))

        tk.Label(cfg_inner, text="輸出速率:", font=("Microsoft JhengHei UI", 9), fg=TEXT_MUTED, bg=CARD_BG).pack(side="left")
        slow_cb = ttk.Combobox(
            cfg_inner,
            textvariable=self.slowmo_var,
            values=["1.0", "0.5", "0.25"],
            state="readonly",
            width=6,
            style="Dark.TCombobox",
        )
        slow_cb.pack(side="left", padx=(6, 16))

        tk.Checkbutton(
            cfg_inner,
            text="優先使用 640 高精度",
            variable=self.prefer640_var,
            bg=CARD_BG,
            fg=TEXT_MAIN,
            selectcolor="#0f172a",
            activebackground=CARD_BG,
            activeforeground=TEXT_MAIN,
            font=("Microsoft JhengHei UI", 9),
        ).pack(side="left")

        # Action bar
        act_bar = tk.Frame(self, bg=BG_DARK)
        act_bar.pack(fill="x", padx=16, pady=(0, 10))

        self.run_btn = tk.Button(
            act_bar,
            text="▶ 開始高精度批次測速與產片",
            font=("Microsoft JhengHei UI", 11, "bold"),
            bg=ACCENT_RED,
            fg=TEXT_MAIN,
            activebackground="#dc2626",
            activeforeground=TEXT_MAIN,
            relief="flat",
            padx=18,
            pady=8,
            cursor="hand2",
            command=self._start_processing,
        )
        self.run_btn.pack(side="left")

        self.stop_btn = tk.Button(
            act_bar,
            text="⏹ 停止",
            font=("Microsoft JhengHei UI", 10),
            bg="#334155",
            fg=TEXT_MAIN,
            relief="flat",
            padx=12,
            pady=8,
            state="disabled",
            cursor="hand2",
            command=self._stop_processing,
        )
        self.stop_btn.pack(side="left", padx=(10, 0))

        tk.Button(
            act_bar,
            text="📂 開啟成品輸出資料夾",
            font=("Microsoft JhengHei UI", 10, "bold"),
            bg="#1e293b",
            fg=ACCENT_GREEN,
            activebackground="#334155",
            activeforeground=ACCENT_GREEN,
            relief="flat",
            padx=14,
            pady=8,
            cursor="hand2",
            command=self._open_output_dir,
        ).pack(side="right")

        # Console Log
        log_card = tk.Frame(self, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        log_card.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        tk.Label(
            log_card,
            text="即時處理與測速結果",
            font=("Microsoft JhengHei UI", 10, "bold"),
            fg=TEXT_MUTED,
            bg=CARD_BG,
        ).pack(anchor="w", padx=14, pady=(8, 4))

        self.log_text = scrolledtext.ScrolledText(
            log_card,
            bg="#080c14",
            fg="#e2e8f0",
            font=("Consolas", 9),
            relief="flat",
            insertbackground=TEXT_MAIN,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._append_log("歡迎使用 SmartScout AI 棒球測速工作站！\n")
        self._append_log("提示：點上方「選取影片檔」或「選取整個資料夾」，再按紅色「開始高精度批次測速與產片」。\n")
        self._append_log("以後若有新模型或新功能，直接點右上角「🔄 檢查並更新」即可一鍵同步最新程式！\n")

    def _append_log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_logs(self) -> None:
        while not self.log_queue.empty():
            self._append_log(self.log_queue.get_nowait())
        self.after(120, self._poll_logs)

    def _add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="選擇投球影片",
            filetypes=[("影片檔", "*.mp4 *.mov *.m4v *.mkv *.avi"), ("所有檔案", "*.*")],
        )
        for f in files:
            if f not in self.selected_paths:
                self.selected_paths.append(f)
                self.file_listbox.insert("end", f)

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="選擇包含多支投球影片的資料夾")
        if folder and folder not in self.selected_paths:
            self.selected_paths.append(folder)
            self.file_listbox.insert("end", f"[資料夾] {folder}")

    def _clear_files(self) -> None:
        self.selected_paths.clear()
        self.file_listbox.delete(0, "end")

    def _open_output_dir(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(OUTPUT_DIR))

    def _start_processing(self) -> None:
        if self.is_running:
            return
        if not self.selected_paths:
            messagebox.showwarning("尚未選擇影片", "請先點選「選取影片檔」或「選取整個資料夾」。")
            return
        try:
            dist = float(self.distance_var.get().strip())
            if dist <= 3 or dist > 40:
                raise ValueError
        except ValueError:
            messagebox.showerror("距離格式錯誤", "請輸入合理的投捕距離（例如 18.44 或 14.0）。")
            return

        cmd = [
            get_python_exe(),
            str(ENGINE_SCRIPT),
            *self.selected_paths,
            "-o", str(OUTPUT_DIR),
            "--distance", str(dist),
            "--angle", self.angle_var.get(),
            "--slowmo", self.slowmo_var.get(),
        ]
        if self.prefer640_var.get():
            cmd.append("--prefer-640")

        self.is_running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._append_log("\n" + "=" * 60 + "\n")
        self._append_log(f"啟動批次測速引擎...\n")

        def run_thread() -> None:
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                self.worker_proc = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_queue.put(line)
                code = proc.wait()
                self.log_queue.put(f"\n[結束] 回傳碼: {code}\n")
            except Exception as e:
                self.log_queue.put(f"\n[錯誤] {e}\n")
            finally:
                self.worker_proc = None
                self.is_running = False
                self.after(0, lambda: self.run_btn.configure(state="normal"))
                self.after(0, lambda: self.stop_btn.configure(state="disabled"))

        threading.Thread(target=run_thread, daemon=True).start()

    def _stop_processing(self) -> None:
        if self.worker_proc is not None:
            try:
                self.worker_proc.terminate()
                self._append_log("\n[使用者手動中止作業]\n")
            except Exception:
                pass

    def _start_update(self) -> None:
        if self.is_running:
            messagebox.showinfo("正在測速中", "請等目前測速任務結束後再更新。")
            return
        self.update_btn.configure(state="disabled", text="更新中...")
        self._append_log("\n[自動更新] 正在連線 GitHub 檢查最新程式與 AI 模型...\n")

        def upd_thread() -> None:
            try:
                before = get_git_short_rev()
                res = subprocess.run(
                    ["git", "pull", "--ff-only", "origin", "main"],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                out_msg = (res.stdout or "") + (res.stderr or "")
                self.log_queue.put(out_msg + "\n")
                after = get_git_short_rev()
                self.after(0, lambda: self.version_var.set(f"版本：{after}"))
                if res.returncode != 0:
                    self.after(0, lambda: messagebox.showwarning("更新失敗", f"無法自動更新，請檢查網路狀態：\n{out_msg[:300]}"))
                elif before == after:
                    self.after(0, lambda: messagebox.showinfo("已是最新版本", f"目前已經是最新版 ({after})，無需更新！"))
                else:
                    def prompt_restart() -> None:
                        if messagebox.askyesno(
                            "更新完成！",
                            f"已從 {before} 更新到最新版本 {after}（含最新程式與 AI 模型）！\n是否立刻重新啟動工作站套用更新？",
                        ):
                            subprocess.Popen(
                                [sys.executable, str(Path(__file__).resolve())],
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            )
                            self.destroy()
                    self.after(0, prompt_restart)
            except Exception as e:
                self.log_queue.put(f"[更新錯誤] {e}\n")
            finally:
                self.after(0, lambda: self.update_btn.configure(state="normal", text="🔄 檢查並更新"))

        threading.Thread(target=upd_thread, daemon=True).start()


if __name__ == "__main__":
    app = SmartScoutStudioApp()
    app.mainloop()
