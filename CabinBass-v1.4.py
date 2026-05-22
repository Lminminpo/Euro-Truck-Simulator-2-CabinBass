import sys
import ctypes
import miniaudio
import numpy as np
import sounddevice as sd
from pynput import keyboard
import os
import random
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog, ttk
from scipy.signal import butter, lfilter, lfilter_zi
from collections import deque
import urllib.request
import zipfile
import tempfile
import shutil
import time

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg')
LIVE_CHANNELS = 2

VBCABLE_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"

DEFAULT_CAB_SETTINGS = {
    'closed_bass': 0.80,
    'open_bass': -0.50,
    'closed_freq': 9900,
    'open_freq': 10900,
    'closed_reflection': 0.75,
    'open_reflection': 0.19,
    'closed_saturation': 1.15,
    'open_saturation': 1.00,
    'closed_volume': 1.0,
    'open_volume': 0.46,
    'bass_low': 20,
    'bass_high': 500,
    'air_freq': 12000,
    'closed_refl_decay': 0.60,
    'open_refl_decay': 0.60,
    'closed_stereo_width': 1.2,
    'open_stereo_width': 1.2,
}


class Smooth:
    def __init__(self, v, s=0.015):
        self.c = v
        self.t = v
        self.s = s
    def set(self, v):
        self.t = v
    def get(self):
        self.c += (self.t - self.c) * self.s
        return self.c


class CabinChannel:
    def __init__(self, settings=None):
        self._settings = dict(DEFAULT_CAB_SETTINGS)
        if settings:
            self._settings.update(settings)
        self._window_open = False

        bl = self._settings['bass_low']
        bh = self._settings['bass_high']
        self.bass_b, self.bass_a = self._build_bass_filter(bl, bh)
        self.bass_zi = lfilter_zi(self.bass_b, self.bass_a) * 0

        af = self._settings['air_freq']
        self.air_b, self.air_a = self._build_air_filter(af)
        self.air_zi = lfilter_zi(self.air_b, self.air_a) * 0

        self.buf = np.zeros(SAMPLE_RATE)
        self.pos = 0

        self.closed_amount = Smooth(1.0, 0.015)
        self.bass_boost = Smooth(self._settings['closed_bass'], 0.015)
        self.reflection_mix = Smooth(self._settings['closed_reflection'], 0.015)
        self.refl_decay = Smooth(self._settings['closed_refl_decay'], 0.015)
        self.stereo_width = Smooth(self._settings['closed_stereo_width'], 0.015)

        self._build_filters()

    def _build_bass_filter(self, low, high):
        nyq = SAMPLE_RATE / 2
        lo = max(20, min(low, high - 10)) / nyq
        hi = min(nyq - 1, max(high, low + 10)) / nyq
        return butter(2, [lo, hi], btype='band')

    def _build_air_filter(self, freq):
        nyq = SAMPLE_RATE / 2
        f = max(1000, min(freq, nyq - 1)) / nyq
        return butter(1, f, btype='low')

    def _build_filters(self):
        nyq = SAMPLE_RATE / 2
        cf = min(max(self._settings['closed_freq'], 50), nyq - 1)
        of = min(max(self._settings['open_freq'], 50), nyq - 1)
        self.closed_b, self.closed_a = butter(2, cf / nyq, btype='low')
        self.closed_zi = lfilter_zi(self.closed_b, self.closed_a) * 0
        self.open_b, self.open_a = butter(2, of / nyq, btype='low')
        self.open_zi = lfilter_zi(self.open_b, self.open_a) * 0

    def update_settings(self, new_settings):
        bass_changed = (new_settings.get('bass_low') != self._settings.get('bass_low') or
                        new_settings.get('bass_high') != self._settings.get('bass_high'))
        air_changed = (new_settings.get('air_freq') != self._settings.get('air_freq'))
        freq_changed = (new_settings.get('closed_freq') != self._settings.get('closed_freq') or
                        new_settings.get('open_freq') != self._settings.get('open_freq'))
        self._settings.update(new_settings)
        if freq_changed:
            self._build_filters()
        if bass_changed:
            bl = self._settings['bass_low']
            bh = self._settings['bass_high']
            self.bass_b, self.bass_a = self._build_bass_filter(bl, bh)
            self.bass_zi = lfilter_zi(self.bass_b, self.bass_a) * 0
        if air_changed:
            af = self._settings['air_freq']
            self.air_b, self.air_a = self._build_air_filter(af)
            self.air_zi = lfilter_zi(self.air_b, self.air_a) * 0
        self.set_window(self._window_open)

    def set_window(self, opened):
        self._window_open = opened
        if opened:
            self.closed_amount.set(0.0)
            self.bass_boost.set(self._settings['open_bass'])
            self.reflection_mix.set(self._settings['open_reflection'])
            self.refl_decay.set(self._settings['open_refl_decay'])
            self.stereo_width.set(self._settings['open_stereo_width'])
        else:
            self.closed_amount.set(1.0)
            self.bass_boost.set(self._settings['closed_bass'])
            self.reflection_mix.set(self._settings['closed_reflection'])
            self.refl_decay.set(self._settings['closed_refl_decay'])
            self.stereo_width.set(self._settings['closed_stereo_width'])

    def get_stereo_width(self):
        return self.stereo_width.get()

    def process(self, data):
        n = len(data)
        out = data.copy()
        bass, self.bass_zi = lfilter(self.bass_b, self.bass_a, out, zi=self.bass_zi)
        out = out + bass * self.bass_boost.get()
        closed_out, self.closed_zi = lfilter(self.closed_b, self.closed_a, out, zi=self.closed_zi)
        open_out, self.open_zi = lfilter(self.open_b, self.open_a, out, zi=self.open_zi)
        ca = self.closed_amount.get()
        out = closed_out * ca + open_out * (1 - ca)
        blen = len(self.buf)
        widx = (np.arange(n) + self.pos) % blen
        self.buf[widx] = out
        delays = [int(SAMPLE_RATE * d) for d in [0.005, 0.008, 0.012, 0.016, 0.022, 0.030, 0.042, 0.058, 0.080, 0.110]]
        gains  = [0.22, 0.18, 0.14, 0.11, 0.09, 0.07, 0.05, 0.035, 0.02, 0.012]
        rd = self.refl_decay.get()
        early = np.zeros(n, dtype=np.float32)
        for ds, g in zip(delays, gains):
            decay = rd ** (ds / 4851.0)
            idx = (np.arange(n) + self.pos - ds) % blen
            early += self.buf[idx] * g * decay
        self.pos = (self.pos + n) % blen
        out, self.air_zi = lfilter(self.air_b, self.air_a, out, zi=self.air_zi)
        rm = self.reflection_mix.get()
        out = out * (1 - rm) + early * rm
        open_sat = self._settings['open_saturation']
        closed_sat = self._settings['closed_saturation']
        dist = open_sat + ca * (closed_sat - open_sat)
        out = np.tanh(out * dist) * 0.90
        open_vol = self._settings['open_volume']
        closed_vol = self._settings['closed_volume']
        volume = open_vol + ca * (closed_vol - open_vol)
        out = out * volume
        return np.clip(out, -0.99, 0.99)


def _key_display_name(key):
    if hasattr(key, 'char') and key.char:
        return key.char.upper()
    name_map = {
        keyboard.Key.space: "空格", keyboard.Key.enter: "回车",
        keyboard.Key.shift: "Shift", keyboard.Key.shift_r: "R-Shift",
        keyboard.Key.ctrl: "Ctrl", keyboard.Key.ctrl_r: "R-Ctrl",
        keyboard.Key.alt: "Alt", keyboard.Key.alt_r: "R-Alt",
        keyboard.Key.tab: "Tab", keyboard.Key.caps_lock: "CapsLock",
        keyboard.Key.backspace: "退格", keyboard.Key.esc: "Esc",
        keyboard.Key.up: "↑", keyboard.Key.down: "↓",
        keyboard.Key.left: "←", keyboard.Key.right: "→",
    }
    if key in name_map:
        return name_map[key]
    if hasattr(key, 'name'):
        kname = key.name
        if kname and kname.startswith('f') and kname[1:].isdigit():
            return kname.upper()
    return str(key).replace("Key.", "")


def _key_to_config(key):
    if hasattr(key, 'char') and key.char:
        return "char:" + key.char.lower()
    return "vk:" + str(key).replace("Key.", "")


def _config_to_key(s):
    if s.startswith("char:"):
        return s[5:]
    if s.startswith("vk:"):
        vk_name = s[3:]
        try:
            return getattr(keyboard.Key, vk_name)
        except AttributeError:
            return None
    return None


def _find_vbcable():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] >= 2:
            name = d['name'].lower()
            if 'cable' in name and 'vb' in name:
                return True
    return False


class SettingsDialog:
    GLOBAL_PARAMS = [
        ('bass_low',    '低音下限 (Hz)',       10,     50,     1),
        ('bass_high',   '低音上限 (Hz)',      100,    900,     1),
        ('air_freq',    '空气感 (Hz)',        4000,  20000,   100),
    ]

    PARAMS = [
        ('bass',        '低音增强',           -1.00,  1.50,  0.01),
        ('freq',        '高频截止 (Hz)',      2000,  20000,   100),
        ('reflection',  '混响强度',            0.00,  1.50,  0.01),
        ('refl_decay',  '反射尾长',            0.10,  1.10,  0.01),
        ('saturation',  '温暖度',              0.80,  1.50,  0.01),
        ('stereo_width','声场宽度',            0.00,  2.40,  0.01),
        ('volume',      '音量',                0.10,  1.80,  0.01),
    ]

    def __init__(self, parent, current_settings):
        self.result = None
        self.vars = {}

        self.win = tk.Toplevel(parent)
        self.win.title("CabinBass - 音效设置")
        self.win.geometry("500x680")
        self.win.minsize(500, 400)
        self.win.resizable(False, True)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._on_cancel)

        tk.Label(self.win, text="调整驾驶室声学参数", font=("Arial", 10), fg="gray").pack(pady=(10, 5))

        container = tk.Frame(self.win)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._canvas = tk.Canvas(container, highlightthickness=0)
        vscroll = tk.Scrollbar(container, orient=tk.VERTICAL, command=self._canvas.yview)
        self._scroll_frame = tk.Frame(self._canvas)
        self._canvas_window = self._canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        self._scroll_frame.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._canvas.bind("<Enter>", self._bind_wheel)
        self._canvas.bind("<Leave>", self._unbind_wheel)

        global_frame = tk.LabelFrame(self._scroll_frame, text="全局参数", font=("Arial", 10, "bold"))
        global_frame.pack(fill=tk.X, pady=(5, 5))
        for i, (key, label, lo, hi, res) in enumerate(self.GLOBAL_PARAMS):
            default = current_settings.get(key, DEFAULT_CAB_SETTINGS[key])
            self.vars[key] = tk.DoubleVar(value=default)
            tk.Label(global_frame, text=label, font=("Arial", 9), width=14, anchor='w').grid(row=i, column=0, padx=(10, 0), pady=2)
            tk.Scale(global_frame, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=self.vars[key], font=("Arial", 8), showvalue=True,
                     length=300).grid(row=i, column=1, padx=(0, 10), pady=2, sticky='ew')

        closed_frame = tk.LabelFrame(self._scroll_frame, text="关窗音效", font=("Arial", 10, "bold"))
        closed_frame.pack(fill=tk.X, pady=(5, 5))
        for i, (key, label, lo, hi, res) in enumerate(self.PARAMS):
            var_key = 'closed_' + key
            default = current_settings.get(var_key, DEFAULT_CAB_SETTINGS[var_key])
            self.vars[var_key] = tk.DoubleVar(value=default)
            tk.Label(closed_frame, text=label, font=("Arial", 9), width=14, anchor='w').grid(row=i, column=0, padx=(10, 0), pady=2)
            tk.Scale(closed_frame, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=self.vars[var_key], font=("Arial", 8), showvalue=True,
                     length=300).grid(row=i, column=1, padx=(0, 10), pady=2, sticky='ew')

        open_frame = tk.LabelFrame(self._scroll_frame, text="开窗音效", font=("Arial", 10, "bold"))
        open_frame.pack(fill=tk.X, pady=(5, 5))
        for i, (key, label, lo, hi, res) in enumerate(self.PARAMS):
            var_key = 'open_' + key
            default = current_settings.get(var_key, DEFAULT_CAB_SETTINGS[var_key])
            self.vars[var_key] = tk.DoubleVar(value=default)
            tk.Label(open_frame, text=label, font=("Arial", 9), width=14, anchor='w').grid(row=i, column=0, padx=(10, 0), pady=2)
            tk.Scale(open_frame, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=self.vars[var_key], font=("Arial", 8), showvalue=True,
                     length=300).grid(row=i, column=1, padx=(0, 10), pady=2, sticky='ew')

        btn_frame = tk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))
        tk.Button(btn_frame, text="恢复默认", command=self._on_reset, font=("Arial", 9), width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="确定", command=self._on_ok, font=("Arial", 10), width=10).pack(side=tk.RIGHT, padx=5)

    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)
    def _bind_wheel(self, event):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    def _unbind_wheel(self, event):
        self._canvas.unbind_all("<MouseWheel>")
    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    def _on_ok(self):
        self._canvas.unbind_all("<MouseWheel>")
        self.result = {}
        for key, _, _, _, _ in self.GLOBAL_PARAMS:
            self.result[key] = self.vars[key].get()
        for key, _, _, _, _ in self.PARAMS:
            self.result['closed_' + key] = self.vars['closed_' + key].get()
            self.result['open_' + key] = self.vars['open_' + key].get()
        self.win.destroy()
    def _on_reset(self):
        for key, _, _, _, _ in self.GLOBAL_PARAMS:
            self.vars[key].set(DEFAULT_CAB_SETTINGS[key])
        for key, _, _, _, _ in self.PARAMS:
            self.vars['closed_' + key].set(DEFAULT_CAB_SETTINGS['closed_' + key])
            self.vars['open_' + key].set(DEFAULT_CAB_SETTINGS['open_' + key])
    def _on_cancel(self):
        self._canvas.unbind_all("<MouseWheel>")
        self.win.destroy()


class KeyBindDialog:
    STEPS = [
        ("设置按键：打开左窗", "open_left"),
        ("设置按键：关闭左窗", "close_left"),
        ("设置按键：打开右窗", "open_right"),
        ("设置按键：关闭右窗", "close_right"),
    ]

    def __init__(self, parent, config_path):
        self.result = None
        self._config_path = config_path
        self._step = 0
        self._keys = {}
        self._done = False

        self.win = tk.Toplevel(parent)
        self.win.title("CabinBass - 按键设置")
        self.win.geometry("380x300")
        self.win.resizable(False, False)
        self.win.grab_set()
        self.win.focus_force()
        self.win.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.win.after(50, self._disable_ime)

        tk.Label(self.win, text="欢迎使用 CabinBass", font=("Arial", 16, "bold")).pack(pady=(25, 5))
        tk.Label(self.win, text="首次使用，请设置车窗控制按键", font=("Arial", 10), fg="gray").pack(pady=(0, 20))
        self.lbl_step = tk.Label(self.win, text="", font=("Arial", 13))
        self.lbl_step.pack(pady=5)
        self.lbl_key = tk.Label(self.win, text="等待按键...", font=("Arial", 20, "bold"), fg="#3a7")
        self.lbl_key.pack(pady=15)
        tk.Label(self.win, text="请在键盘上按下对应按键", font=("Arial", 9), fg="gray").pack()
        tk.Button(self.win, text="跳过（使用默认按键）", font=("Arial", 9),
                  command=self._on_skip, relief=tk.FLAT, fg="gray").pack(side=tk.BOTTOM, pady=(0, 12))
        self.win.bind('<KeyPress>', self._on_key)
        self._update_step_text()

    def _disable_ime(self):
        try:
            hwnd = int(self.win.frame(), 16)
            ctypes.windll.imm32.ImmAssociateContext(hwnd, 0)
        except: pass
    def _update_step_text(self):
        title, _ = self.STEPS[self._step]
        self.lbl_step.config(text=f"[{self._step + 1}/4] {title}")
    def _on_key(self, event):
        if self._done: return
        key_cfg, display = self._tk_event_to_config(event)
        if key_cfg is None: return
        _, field = self.STEPS[self._step]
        for k, v in self._keys.items():
            if k != field and v == key_cfg:
                self.lbl_key.config(text=f"{display}（已被占用，请重按）", fg="#c33")
                return
        self._keys[field] = key_cfg
        self.lbl_key.config(text=display, fg="#3a7")
        self.win.after(600, self._next_step)
    def _tk_event_to_config(self, event):
        char, keysym = event.char, event.keysym
        modifier_keysyms = {'Shift_L','Shift_R','Control_L','Control_R','Alt_L','Alt_R',
                            'Caps_Lock','Num_Lock','Scroll_Lock','Win_L','Win_R','Meta_L','Meta_R'}
        if char and keysym not in modifier_keysyms:
            if len(char) == 1 and char.isprintable():
                return "char:" + char.lower(), char.upper()
        keysym_map = {
            'space':('vk:space','空格'),'Return':('vk:enter','回车'),
            'Shift_L':('vk:shift','Shift'),'Shift_R':('vk:shift_r','R-Shift'),
            'Control_L':('vk:ctrl','Ctrl'),'Control_R':('vk:ctrl_r','R-Ctrl'),
            'Alt_L':('vk:alt','Alt'),'Alt_R':('vk:alt_r','R-Alt'),
            'Tab':('vk:tab','Tab'),'Caps_Lock':('vk:caps_lock','CapsLock'),
            'BackSpace':('vk:backspace','退格'),'Escape':('vk:esc','Esc'),
            'Up':('vk:up','↑'),'Down':('vk:down','↓'),
            'Left':('vk:left','←'),'Right':('vk:right','→'),
            'Insert':('vk:insert','Insert'),'Delete':('vk:delete','Delete'),
            'Home':('vk:home','Home'),'End':('vk:end','End'),
            'Prior':('vk:page_up','PageUp'),'Next':('vk:page_down','PageDown'),
        }
        if keysym in keysym_map: return keysym_map[keysym]
        if keysym.startswith('F') and keysym[1:].isdigit():
            fname = keysym.lower(); return "vk:" + fname, fname.upper()
        return "vk:" + keysym.lower(), keysym
    def _next_step(self):
        if self._done: return
        self._step += 1
        if self._step >= len(self.STEPS): self._finish(); return
        self._update_step_text(); self.lbl_key.config(text="等待按键...", fg="#3a7"); self.win.focus_force()
    def _finish(self):
        if self._done: return
        self._done = True
        result = {field: self._keys[field] for _, field in self.STEPS}
        self._save_keys(result); self.result = result
        try: self.win.destroy()
        except tk.TclError: pass
    def _on_skip(self):
        if self._done: return
        self._done = True
        d = {"open_left":"char:a","close_left":"char:s","open_right":"char:d","close_right":"char:w"}
        self._save_keys(d); self.result = d
        try: self.win.destroy()
        except tk.TclError: pass
    def _on_cancel(self): self._on_skip()
    def _save_keys(self, key_dict):
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                f.write("keybinds_v1\n")
                for field in ("open_left","close_left","open_right","close_right"):
                    f.write(field + "=" + key_dict[field] + "\n")
        except: pass


class App:
    def __init__(self):
        self.playlist = []
        self.current_index = -1
        self.audio_data = None
        self.current_pos = 0
        self.playing = False
        self.left_open = False
        self.right_open = False
        self.volume = 0.8
        self.play_mode = "loop"
        self._auto_next = False
        self.lock = threading.Lock()

        self._shuffle_order = []
        self._shuffle_pos = 0

        self.key_open_left = "char:a"
        self.key_close_left = "char:s"
        self.key_open_right = "char:d"
        self.key_close_right = "char:w"

        self._cab_settings = dict(DEFAULT_CAB_SETTINGS)
        self.cab_l = CabinChannel(self._cab_settings)
        self.cab_r = CabinChannel(self._cab_settings)

        self._live_active = False
        self._live_input = None
        self._live_buf = deque(maxlen=20)
        self._live_lock = threading.Lock()
        self._last_live_device = ""
        self._live_dev_sr = SAMPLE_RATE
        self._live_resample_tail = np.zeros((0, LIVE_CHANNELS), dtype=np.float32)

        self._build_gui()
        self._setup_audio()
        self._load_config()
        self._setup_keyboard()
        self._tick()

    def _config_path(self):
        if getattr(sys, 'frozen', False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "cabinbass.cfg")

    def _get_current_position_sec(self):
        if self.audio_data is not None:
            return self.current_pos / SAMPLE_RATE
        return 0

    def _save_config(self):
        try:
            folder = os.path.dirname(self.playlist[0]) if self.playlist else ""
            with open(self._config_path(), "w", encoding="utf-8") as f:
                f.write("keybinds_v1\n")
                f.write("open_left=" + self.key_open_left + "\n")
                f.write("close_left=" + self.key_close_left + "\n")
                f.write("open_right=" + self.key_open_right + "\n")
                f.write("close_right=" + self.key_close_right + "\n")
                f.write(folder + "\n")
                f.write(str(self.volume) + "\n")
                f.write(self.play_mode + "\n")
                f.write(str(self.current_index) + "\n")
                f.write(str(self._get_current_position_sec()) + "\n")
                f.write("settings_v1\n")
                for key in sorted(self._cab_settings.keys()):
                    f.write(f"{key}={self._cab_settings[key]}\n")
                f.write(f"live_device={self._last_live_device}\n")
        except: pass

    def _load_config(self):
        try:
            with open(self._config_path(), "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            if lines and lines[0].strip() == "keybinds_v1":
                key_fields = ("open_left","close_left","open_right","close_right")
                key_section = {}
                for i, field in enumerate(key_fields):
                    line_idx = 1 + i
                    if line_idx < len(lines):
                        parts = lines[line_idx].strip().split("=", 1)
                        if len(parts) == 2 and parts[0] == field:
                            key_section[field] = parts[1]
                self.key_open_left = key_section.get("open_left","char:a")
                self.key_close_left = key_section.get("close_left","char:s")
                self.key_open_right = key_section.get("open_right","char:d")
                self.key_close_right = key_section.get("close_right","char:w")
                rest = lines[5:]
                saved_folder = rest[0].strip() if len(rest) > 0 else ""
                saved_vol = float(rest[1].strip()) if len(rest) > 1 else 0.8
                saved_mode = rest[2].strip() if len(rest) > 2 else "loop"
                saved_index = int(rest[3].strip()) if len(rest) > 3 else -1
                saved_seek = float(rest[4].strip()) if len(rest) > 4 else 0
                self.volume = saved_vol
                self.vol_scale.set(saved_vol)
                self.play_mode = saved_mode if saved_mode in ("loop","shuffle") else "loop"
                self._update_mode_label()
                settings_start = 10
                if len(lines) > settings_start and lines[settings_start].strip() == "settings_v1":
                    int_keys = {'closed_freq','open_freq','bass_low','bass_high','air_freq'}
                    for i in range(settings_start + 1, len(lines)):
                        line = lines[i].strip()
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k == 'live_device':
                                self._last_live_device = v
                            elif k in self._cab_settings:
                                try:
                                    if k in int_keys:
                                        self._cab_settings[k] = int(float(v))
                                    else:
                                        self._cab_settings[k] = float(v)
                                except: pass
                self._apply_cab_settings()
                if saved_folder and os.path.isdir(saved_folder):
                    self.root.after(100, lambda: self._restore_session(saved_folder, saved_index, saved_seek))
            else:
                self._run_first_time_setup()
        except FileNotFoundError:
            self._run_first_time_setup()
        except: pass

    def _apply_cab_settings(self):
        with self.lock:
            self.cab_l.update_settings(self._cab_settings)
            self.cab_r.update_settings(self._cab_settings)

    def _open_settings(self):
        dialog = SettingsDialog(self.root, self._cab_settings)
        self.root.wait_window(dialog.win)
        if dialog.result:
            self._cab_settings = dialog.result
            self._apply_cab_settings()
            self._save_config()

    def _run_first_time_setup(self):
        self.root.withdraw()
        dialog = KeyBindDialog(self.root, self._config_path())
        self.root.wait_window(dialog.win)
        self.root.deiconify()
        if dialog.result:
            self.key_open_left = dialog.result["open_left"]
            self.key_close_left = dialog.result["close_left"]
            self.key_open_right = dialog.result["open_right"]
            self.key_close_right = dialog.result["close_right"]

    def _restore_session(self, folder, index, seek_sec):
        self._load_folder(folder)
        if index >= 0 and index < len(self.playlist):
            self._load_song(index, seek_sec=seek_sec, auto_play=False)

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("CabinBass v1.4")
        self.root.geometry("420x680")
        self.root.resizable(False, False)

        btn_row = tk.Frame(self.root)
        btn_row.pack(fill=tk.X, padx=10, pady=(10, 5))
        tk.Button(btn_row, text="选择音乐文件夹", command=self._pick_folder, font=("Arial", 11)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(btn_row, text="音效设置", command=self._open_settings, font=("Arial", 11)).pack(side=tk.RIGHT, padx=(3, 0))

        mode_row = tk.Frame(self.root)
        mode_row.pack(fill=tk.X, padx=10, pady=(5, 0))
        self._mode_var = tk.StringVar(value="file")
        tk.Radiobutton(mode_row, text="本地文件", variable=self._mode_var, value="file",
                       command=self._on_mode_change, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="实时监听", variable=self._mode_var, value="live",
                       command=self._on_mode_change, font=("Arial", 10)).pack(side=tk.LEFT, padx=(10, 0))

        self._vf_frame = tk.Frame(self.root)
        tk.Label(self._vf_frame, text="音量", font=("Arial", 9)).pack(side=tk.LEFT)
        self.vol_scale = tk.Scale(self._vf_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, command=self._on_vol, showvalue=False)
        self.vol_scale.set(0.8)
        self.vol_scale.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self._vf_frame.pack(fill=tk.X, padx=10, pady=2)

        sf = tk.LabelFrame(self.root, text="窗户状态", font=("Arial", 10))
        sf.pack(fill=tk.X, padx=10, pady=8)
        self.lbl_win = tk.Label(sf, text="左窗: 关 | 右窗: 关", font=("Arial", 12))
        self.lbl_win.pack(pady=6)
        self.lbl_keys = tk.Label(sf, text="", font=("Arial", 8), fg="gray")
        self.lbl_keys.pack(pady=(0, 6))
        self.lbl_f9 = tk.Label(sf, text="F9 = 播放/暂停", font=("Arial", 8), fg="gray")
        self.lbl_f9.pack(pady=(0, 6))

        self.lbl_status = tk.Label(self.root, text="就绪", font=("Arial", 9), fg="gray")
        self.lbl_status.pack(fill=tk.X, padx=10, pady=(0, 8))

        self._live_frame = tk.Frame(self.root)

        self._lbl_vb_title = tk.Label(self._live_frame, text="", font=("Arial", 13, "bold"))
        self._lbl_vb_title.pack(pady=(20, 5), anchor='w')

        self._lbl_vb_desc = tk.Label(self._live_frame, text="", font=("Arial", 9), fg="gray", justify=tk.LEFT)
        self._lbl_vb_desc.pack(anchor='w')

        self._btn_install = tk.Button(self._live_frame, text="一键安装 VB-Cable (安装需要重启)", font=("Arial", 12),
                                      command=self._do_install, width=22, height=2)
        self._btn_install.pack(pady=(15, 5))

        self._lbl_install_hint = tk.Label(self._live_frame, text="", font=("Arial", 9), fg="gray")
        self._lbl_install_hint.pack(anchor='w')

        sep = tk.Frame(self._live_frame, height=1, bg="#ccc")
        sep.pack(fill=tk.X, pady=15)

        guide_text = "① 把音乐软件的输出设备改为「CABLE Input」\n② 在下方选择「CABLE Output」\n③ 点击「开始监听」"
        tk.Label(self._live_frame, text=guide_text, font=("Arial", 9), fg="#555",
                 justify=tk.LEFT).pack(anchor='w', pady=(0, 10))

        dev_row = tk.Frame(self._live_frame)
        dev_row.pack(fill=tk.X, pady=(0, 5))
        self.combo_device = ttk.Combobox(dev_row, state="readonly", font=("Arial", 9), width=35)
        self.combo_device.pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(dev_row, text="刷新", command=self._refresh_devices, font=("Arial", 9), width=5).pack(side=tk.LEFT)

        self.btn_live = tk.Button(self._live_frame, text="开始监听", command=self._toggle_live,
                                  font=("Arial", 11), width=15)
        self.btn_live.pack(pady=8)

        self.playlist_frame = tk.Frame(self.root)

        frame = tk.Frame(self.playlist_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(frame, yscrollcommand=sb.set, font=("Consolas", 10), activestyle="none")
        self.listbox.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.listbox.yview)
        self.listbox.bind('<Double-1>', lambda e: self._on_select())

        self.lbl_song = tk.Label(self.playlist_frame, text="未选择歌曲", font=("Arial", 11), anchor="w")
        self.lbl_song.pack(fill=tk.X, padx=10, pady=2)
        self.lbl_time = tk.Label(self.playlist_frame, text="", font=("Arial", 9), anchor="w", fg="gray")
        self.lbl_time.pack(fill=tk.X, padx=10)

        ctrl = tk.Frame(self.playlist_frame)
        ctrl.pack(pady=8)
        tk.Button(ctrl, text="⏮", command=self._prev, width=5, font=("Arial", 14)).pack(side=tk.LEFT, padx=4)
        self.btn_play = tk.Button(ctrl, text="▶", command=self._toggle_play, width=5, font=("Arial", 14))
        self.btn_play.pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="⏭", command=self._next, width=5, font=("Arial", 14)).pack(side=tk.LEFT, padx=4)

        mode_frame = tk.Frame(self.playlist_frame)
        mode_frame.pack(pady=2)
        self.btn_mode = tk.Button(mode_frame, text="列表循环 🔂", command=self._cycle_mode, width=12, font=("Arial", 9))
        self.btn_mode.pack()

        self.playlist_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0, before=self._vf_frame)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_mode_change(self):
        mode = self._mode_var.get()
        if mode == "live":
            self.playlist_frame.pack_forget()
            self._live_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, before=self._vf_frame)
            self._refresh_devices()
        else:
            self._live_frame.pack_forget()
            self.playlist_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0, before=self._vf_frame)
            if self._live_active:
                self._stop_live()

    def _list_input_devices(self):
        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] >= 2:
                result.append((i, dev['name']))
        return result

    def _refresh_devices(self):
        devices = self._list_input_devices()
        self._device_map = {name: idx for idx, name in devices}
        names = list(self._device_map.keys())
        self.combo_device['values'] = names

        has_vb = _find_vbcable()
        if has_vb:
            self._lbl_vb_title.config(text="VB-Cable 已就绪")
            self._lbl_vb_desc.config(text="")
            self._btn_install.pack_forget()
            self._lbl_install_hint.config(text="")
        else:
            self._lbl_vb_title.config(text="需要安装 VB-Cable")
            self._lbl_vb_desc.config(text="免费虚拟声卡，用于桥接音乐软件声音到 CabinBass")
            if not self._btn_install.winfo_ismapped():
                self._btn_install.pack(pady=(15, 5))

        if self._last_live_device and self._last_live_device in self._device_map:
            self.combo_device.set(self._last_live_device)
        elif names:
            for n in names:
                nl = n.lower()
                if 'cable' in nl and 'output' in nl:
                    self.combo_device.set(n)
                    return
            self.combo_device.set(names[0])

    def _do_install(self):
        self._lbl_install_hint.config(text="正在下载 VB-Cable，请稍候...")
        self._btn_install.config(state=tk.DISABLED)
        self.root.update()

        def _worker():
            try:
                tmp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(tmp_dir, "vbcable.zip")

                urllib.request.urlretrieve(VBCABLE_URL, zip_path)

                self.root.after(0, lambda: self._lbl_install_hint.config(text="正在解压..."))

                extract_dir = os.path.join(tmp_dir, "vbcable")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(extract_dir)

                setup_exe = None
                for root, dirs, files in os.walk(extract_dir):
                    for f in files:
                        fl = f.lower()
                        if 'setup_x64' in fl and fl.endswith('.exe'):
                            setup_exe = os.path.join(root, f)
                            break
                        elif 'setup' in fl and fl.endswith('.exe'):
                            if setup_exe is None:
                                setup_exe = os.path.join(root, f)
                    if setup_exe and 'x64' in setup_exe.lower():
                        break

                if not setup_exe:
                    self.root.after(0, lambda: self._lbl_install_hint.config(text="下载失败：找不到安装程序"))
                    self.root.after(0, lambda: self._btn_install.config(state=tk.NORMAL))
                    try: shutil.rmtree(tmp_dir)
                    except: pass
                    return

                self.root.after(0, lambda: self._lbl_install_hint.config(text="正在安装（需要管理员权限）..."))

                ret = ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", setup_exe, "/S", str(tmp_dir), 1
                )
                if ret <= 32:
                    self.root.after(0, lambda: self._lbl_install_hint.config(text="安装被取消"))
                    self.root.after(0, lambda: self._btn_install.config(state=tk.NORMAL))
                    try: shutil.rmtree(tmp_dir)
                    except: pass
                    return

                for _ in range(60):
                    time.sleep(1)
                    if _find_vbcable():
                        break

                try: shutil.rmtree(tmp_dir)
                except: pass

                if _find_vbcable():
                    self.root.after(0, lambda: self._lbl_install_hint.config(
                        text="安装完成！请重启电脑后重新打开 CabinBass"))
                    self.root.after(0, lambda: self._btn_install.config(state=tk.NORMAL))
                    self.root.after(500, self._refresh_devices)
                else:
                    self.root.after(0, lambda: self._lbl_install_hint.config(
                        text="安装已完成，请重启电脑后重新打开 CabinBass"))
                    self.root.after(0, lambda: self._btn_install.config(state=tk.NORMAL))

            except Exception as e:
                self.root.after(0, lambda: self._lbl_install_hint.config(text=f"安装失败：{e}"))
                self.root.after(0, lambda: self._btn_install.config(state=tk.NORMAL))

        threading.Thread(target=_worker, daemon=True).start()

    def _toggle_live(self):
        if self._live_active:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self):
        sel = self.combo_device.get()
        if not sel or sel not in self._device_map:
            self.lbl_status.config(text="请先选择输入设备")
            return
        self.playing = False
        self.btn_play.config(text="▶")
        input_idx = self._device_map[sel]
        dev_info = sd.query_devices(input_idx)
        self._live_dev_sr = int(dev_info['default_samplerate'])
        self._live_resample_tail = np.zeros((0, LIVE_CHANNELS), dtype=np.float32)
        try:
            self._live_buf.clear()
            self._live_input = sd.InputStream(
                samplerate=self._live_dev_sr,
                blocksize=BLOCK_SIZE,
                channels=LIVE_CHANNELS,
                dtype='float32',
                device=input_idx,
                callback=self._input_cb
            )
            self._live_input.start()
            self._live_active = True
            self._last_live_device = sel
            self.btn_live.config(text="停止监听")
            self.lbl_status.config(text=f"监听中: {sel}")
        except Exception as e:
            self.lbl_status.config(text=f"启动失败: {e}")

    def _stop_live(self):
        self._live_active = False
        if self._live_input:
            try:
                self._live_input.stop()
                self._live_input.close()
            except: pass
            self._live_input = None
        self._live_buf.clear()
        self.btn_live.config(text="开始监听")
        self.lbl_status.config(text="监听已停止")

    def _input_cb(self, indata, frames, time_info, status):
        if not self._live_active:
            return
        chunk = indata[:, :LIVE_CHANNELS].copy()
        if self._live_dev_sr != SAMPLE_RATE:
            merged = np.vstack([self._live_resample_tail, chunk])
            ratio = SAMPLE_RATE / self._live_dev_sr
            target_n = int(round(len(merged) * ratio))
            src_idx = np.linspace(0, len(merged) - 1, target_n)
            out = np.column_stack([
                np.interp(src_idx, np.arange(len(merged)), merged[:, ch])
                for ch in range(LIVE_CHANNELS)
            ]).astype(np.float32)
            self._live_resample_tail = merged[-1:, :]
            ready = out
        else:
            ready = chunk
        with self._live_lock:
            self._live_buf.append(ready)

    def _apply_stereo_width(self, left, right, width):
        if abs(width - 1.0) < 0.01:
            return left, right
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        side *= width
        if width > 1.0:
            mid *= max(0.0, 2.0 - width)
        return mid + side, mid - side

    def _update_key_label(self):
        def _disp(cfg):
            k = _config_to_key(cfg)
            if k is None: return "?"
            if isinstance(k, str): return k.upper()
            return _key_display_name(k)
        ol = _disp(self.key_open_left)
        cl = _disp(self.key_close_left)
        orr = _disp(self.key_open_right)
        cr = _disp(self.key_close_right)
        self.lbl_keys.config(text=f"{ol}=左窗开  {cl}=左窗关  {orr}=右窗开  {cr}=右窗关")

    def _build_shuffle_order(self):
        if not self.playlist:
            self._shuffle_order = []; self._shuffle_pos = 0; return
        indices = list(range(len(self.playlist)))
        random.shuffle(indices)
        if len(indices) > 1 and self.current_index >= 0 and indices[0] == self.current_index:
            swap = random.randint(1, len(indices) - 1)
            indices[0], indices[swap] = indices[swap], indices[0]
        self._shuffle_order = indices; self._shuffle_pos = 0

    def _cycle_mode(self):
        if self.play_mode == "loop":
            self.play_mode = "shuffle"; self._build_shuffle_order()
        else:
            self.play_mode = "loop"
        self._update_mode_label(); self._save_config()

    def _update_mode_label(self):
        labels = {"loop": "列表循环 🔂", "shuffle": "随机播放 🔀"}
        self.btn_mode.config(text=labels.get(self.play_mode, "列表循环 🔂"))

    def _match_key(self, pressed_key, cfg_str):
        target = _config_to_key(cfg_str)
        if target is None: return False
        if isinstance(target, str):
            if hasattr(pressed_key, 'char') and pressed_key.char:
                return pressed_key.char.lower() == target
            return False
        else:
            return pressed_key == target

    def _setup_keyboard(self):
        self._update_key_label()
        def on_press(key):
            if key == keyboard.Key.f9:
                self.root.after(0, self._toggle_play); return
            changed = False
            if self._match_key(key, self.key_open_left):
                self.left_open = True; changed = True
            elif self._match_key(key, self.key_close_left):
                self.left_open = False; changed = True
            elif self._match_key(key, self.key_open_right):
                self.right_open = True; changed = True
            elif self._match_key(key, self.key_close_right):
                self.right_open = False; changed = True
            if changed:
                with self.lock:
                    self.cab_l.set_window(self.left_open)
                    self.cab_r.set_window(self.right_open)
        self.kl = keyboard.Listener(on_press=on_press)
        self.kl.daemon = True; self.kl.start()

    def _setup_audio(self):
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, callback=self._audio_cb, blocksize=BLOCK_SIZE, dtype='float32')
        self.stream.start()

    def _audio_cb(self, outdata, frames, time_info, status):
        if self._live_active:
            with self._live_lock:
                chunk = self._live_buf.popleft() if self._live_buf else None
            if chunk is None:
                outdata[:] = 0; return
            if len(chunk) < frames:
                chunk = np.vstack([chunk, np.zeros((frames - len(chunk), 2))])
            elif len(chunk) > frames:
                chunk = chunk[:frames]
            with self.lock:
                left = self.cab_l.process(chunk[:, 0])
                right = self.cab_r.process(chunk[:, 1])
                sw = (self.cab_l.get_stereo_width() + self.cab_r.get_stereo_width()) * 0.5
            left, right = self._apply_stereo_width(left, right, sw)
            vol = self.volume
            outdata[:, 0] = left * vol
            outdata[:, 1] = right * vol
            return
        if self.audio_data is None or not self.playing:
            outdata[:] = 0; return
        with self.lock:
            remaining = len(self.audio_data) - self.current_pos
            if remaining <= 0:
                outdata[:] = 0; self.playing = False; self._auto_next = True; return
            n = min(frames, remaining)
            chunk = self.audio_data[self.current_pos:self.current_pos + n].copy()
            if n < frames:
                chunk = np.vstack([chunk, np.zeros((frames - n, 2))])
            left = self.cab_l.process(chunk[:, 0])
            right = self.cab_r.process(chunk[:, 1])
            sw = (self.cab_l.get_stereo_width() + self.cab_r.get_stereo_width()) * 0.5
            left, right = self._apply_stereo_width(left, right, sw)
            vol = self.volume
            outdata[:, 0] = left * vol
            outdata[:, 1] = right * vol
            self.current_pos += n
            if self.current_pos >= len(self.audio_data):
                self.playing = False; self._auto_next = True

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="选择音乐文件夹")
        if not folder: return
        self._load_folder(folder); self._save_config()

    def _load_folder(self, folder):
        self.playlist = []; self.listbox.delete(0, tk.END)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(EXTENSIONS):
                self.playlist.append(os.path.join(folder, f))
                self.listbox.insert(tk.END, os.path.splitext(f)[0])
        self.lbl_status.config(text=f"已加载 {len(self.playlist)} 首歌曲")
        if self.playlist:
            self.listbox.select_set(0); self.current_index = 0
        self._build_shuffle_order()

    def _next_index(self):
        if not self.playlist: return -1
        if self.play_mode == "shuffle":
            if len(self.playlist) <= 1: return 0
            if not self._shuffle_order or self._shuffle_pos >= len(self._shuffle_order):
                self._build_shuffle_order()
            idx = self._shuffle_order[self._shuffle_pos]; self._shuffle_pos += 1; return idx
        else:
            return (self.current_index + 1) % len(self.playlist)

    def _load_song(self, index, seek_sec=0, auto_play=True):
        if self._live_active:
            self._stop_live()
        if not self.playlist or index < 0 or index >= len(self.playlist): return
        self.playing = False; self.lbl_status.config(text="加载中...")
        def _do():
            try:
                path = self.playlist[index]
                decoded = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=2, sample_rate=SAMPLE_RATE)
                raw = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
                if decoded.nchannels == 1:
                    raw = np.column_stack([raw, raw]).flatten()
                stereo = raw.reshape(-1, 2)
                seek_pos = int(seek_sec * SAMPLE_RATE)
                if seek_pos >= len(stereo): seek_pos = 0
                with self.lock:
                    self.audio_data = stereo; self.current_pos = seek_pos
                    self.current_index = index; self.playing = auto_play
                name = os.path.splitext(os.path.basename(path))[0]
                self.root.after(0, lambda: self._update_play_ui(name, index, auto_play))
            except Exception as e:
                self.root.after(0, lambda: self.lbl_status.config(text=f"加载失败: {e}"))
        threading.Thread(target=_do, daemon=True).start()

    def _update_play_ui(self, name, index, auto_play):
        self.lbl_song.config(text=name)
        self.listbox.select_clear(0, tk.END)
        self.listbox.select_set(index); self.listbox.see(index)
        if auto_play:
            self.lbl_status.config(text="正在播放"); self.btn_play.config(text="⏸")
        else:
            self.lbl_status.config(text="已暂停"); self.btn_play.config(text="▶")

    def _toggle_play(self):
        if self._live_active:
            self.lbl_status.config(text="请先停止实时监听"); return
        if self.audio_data is None:
            if self.playlist: self._load_song(0)
            return
        self.playing = not self.playing
        if self.playing:
            self.btn_play.config(text="⏸"); self.lbl_status.config(text="正在播放")
        else:
            self.btn_play.config(text="▶"); self.lbl_status.config(text="已暂停")

    def _next(self):
        if not self.playlist: return
        self._load_song(self._next_index())

    def _prev(self):
        if not self.playlist: return
        self._load_song((self.current_index - 1) % len(self.playlist))

    def _on_select(self):
        sel = self.listbox.curselection()
        if sel: self._load_song(sel[0])

    def _on_vol(self, val):
        self.volume = float(val)

    def _tick(self):
        l = "开" if self.left_open else "关"
        r = "开" if self.right_open else "关"
        self.lbl_win.config(text=f"左窗: {l} | 右窗: {r}")
        if self._mode_var.get() == "live":
            self.lbl_f9.config(text="")
        else:
            self.lbl_f9.config(text="F9 = 播放/暂停")
        if self._live_active:
            self.lbl_time.config(text="实时监听中...")
        elif self.audio_data is not None:
            cur = self.current_pos / SAMPLE_RATE
            total = len(self.audio_data) / SAMPLE_RATE
            self.lbl_time.config(text=f"{int(cur)//60}:{int(cur)%60:02d} / {int(total)//60}:{int(total)%60:02d}")
        if self._auto_next:
            self._auto_next = False; self._next()
        self.root.after(200, self._tick)

    def _on_close(self):
        if self._live_active:
            self._stop_live()
        self._save_config()
        self.playing = False
        self.stream.stop(); self.stream.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
