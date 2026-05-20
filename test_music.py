import sys
import miniaudio
import numpy as np
import sounddevice as sd
from pynput import keyboard
import os
import random
import threading
import tkinter as tk
from tkinter import filedialog
from scipy.signal import butter, lfilter, lfilter_zi

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg')


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
    def __init__(self):
        self.bass_b, self.bass_a = butter(2, [80 / (SAMPLE_RATE / 2), 150 / (SAMPLE_RATE / 2)], btype='band')
        self.bass_zi = lfilter_zi(self.bass_b, self.bass_a) * 0
        self.closed_b, self.closed_a = butter(2, 7000 / (SAMPLE_RATE / 2), btype='low')
        self.closed_zi = lfilter_zi(self.closed_b, self.closed_a) * 0
        self.open_b, self.open_a = butter(2, 14000 / (SAMPLE_RATE / 2), btype='low')
        self.open_zi = lfilter_zi(self.open_b, self.open_a) * 0
        self.air_b, self.air_a = butter(1, 12000 / (SAMPLE_RATE / 2), btype='low')
        self.air_zi = lfilter_zi(self.air_b, self.air_a) * 0
        self.buf = np.zeros(SAMPLE_RATE)
        self.pos = 0
        self.closed_amount = Smooth(1.0, 0.015)
        self.bass_boost = Smooth(0.30, 0.015)
        self.reflection_mix = Smooth(0.45, 0.015)

    def set_window(self, opened):
        if opened:
            self.closed_amount.set(0.0)
            self.bass_boost.set(-0.15)
            self.reflection_mix.set(0.15)
        else:
            self.closed_amount.set(1.0)
            self.bass_boost.set(0.30)
            self.reflection_mix.set(0.45)

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
        early = np.zeros(n, dtype=np.float32)
        for ds, g in zip(delays, gains):
            idx = (np.arange(n) + self.pos - ds) % blen
            early += self.buf[idx] * g
        self.pos = (self.pos + n) % blen
        out, self.air_zi = lfilter(self.air_b, self.air_a, out, zi=self.air_zi)
        rm = self.reflection_mix.get()
        out = out * (1 - rm) + early * rm
        dist = 1.02 + ca * 0.13
        out = np.tanh(out * dist) * 0.90
        volume = 0.5 + ca * 0.5
        out = out * volume
        return np.clip(out, -0.99, 0.99)


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
        self.cab_l = CabinChannel()
        self.cab_r = CabinChannel()

        self._build_gui()
        self._setup_keyboard()
        self._setup_audio()
        self._load_config()
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
                f.write(folder + "\n")
                f.write(str(self.volume) + "\n")
                f.write(self.play_mode + "\n")
                f.write(str(self.current_index) + "\n")
                f.write(str(self._get_current_position_sec()) + "\n")
        except:
            pass

    def _load_config(self):
        try:
            with open(self._config_path(), "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                saved_folder = lines[0].strip() if len(lines) > 0 else ""
                saved_vol = float(lines[1].strip()) if len(lines) > 1 else 0.8
                saved_mode = lines[2].strip() if len(lines) > 2 else "loop"
                saved_index = int(lines[3].strip()) if len(lines) > 3 else -1
                saved_seek = float(lines[4].strip()) if len(lines) > 4 else 0
                self.volume = saved_vol
                self.vol_scale.set(saved_vol)
                self.play_mode = saved_mode if saved_mode in ("loop", "shuffle") else "loop"
                self._update_mode_label()
                if saved_folder and os.path.isdir(saved_folder):
                    self.root.after(100, lambda: self._restore_session(saved_folder, saved_index, saved_seek))
        except:
            pass

    def _restore_session(self, folder, index, seek_sec):
        self._load_folder(folder)
        if index >= 0 and index < len(self.playlist):
            self._load_song(index, seek_sec=seek_sec, auto_play=False)

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("CabinBass")
        self.root.geometry("420x600")
        self.root.resizable(False, False)

        tk.Button(self.root, text="选择音乐文件夹", command=self._pick_folder, font=("Arial", 11)).pack(fill=tk.X, padx=10, pady=(10, 5))

        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(frame, yscrollcommand=sb.set, font=("Consolas", 10), activestyle="none")
        self.listbox.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.listbox.yview)
        self.listbox.bind('<Double-1>', lambda e: self._on_select())

        self.lbl_song = tk.Label(self.root, text="未选择歌曲", font=("Arial", 11), anchor="w")
        self.lbl_song.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_time = tk.Label(self.root, text="", font=("Arial", 9), anchor="w", fg="gray")
        self.lbl_time.pack(fill=tk.X, padx=10)

        ctrl = tk.Frame(self.root)
        ctrl.pack(pady=8)
        tk.Button(ctrl, text="⏮", command=self._prev, width=5, font=("Arial", 14)).pack(side=tk.LEFT, padx=4)
        self.btn_play = tk.Button(ctrl, text="▶", command=self._toggle_play, width=5, font=("Arial", 14))
        self.btn_play.pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="⏭", command=self._next, width=5, font=("Arial", 14)).pack(side=tk.LEFT, padx=4)

        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=2)
        self.btn_mode = tk.Button(mode_frame, text="列表循环 🔂", command=self._cycle_mode, width=12, font=("Arial", 9))
        self.btn_mode.pack()

        vf = tk.Frame(self.root)
        vf.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(vf, text="音量", font=("Arial", 9)).pack(side=tk.LEFT)
        self.vol_scale = tk.Scale(vf, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, command=self._on_vol, showvalue=False)
        self.vol_scale.set(0.8)
        self.vol_scale.pack(fill=tk.X, expand=True, side=tk.LEFT)

        sf = tk.LabelFrame(self.root, text="窗户状态", font=("Arial", 10))
        sf.pack(fill=tk.X, padx=10, pady=8)
        self.lbl_win = tk.Label(sf, text="左窗: 关 | 右窗: 关", font=("Arial", 12))
        self.lbl_win.pack(pady=6)
        tk.Label(sf, text="A=左窗开  S=左窗关  D=右窗开  W=右窗关", font=("Arial", 8), fg="gray").pack(pady=(0, 6))
        tk.Label(sf, text="F9 = 播放/暂停", font=("Arial", 8), fg="gray").pack(pady=(0, 6))

        self.lbl_status = tk.Label(self.root, text="就绪", font=("Arial", 9), fg="gray")
        self.lbl_status.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _cycle_mode(self):
        if self.play_mode == "loop":
            self.play_mode = "shuffle"
        else:
            self.play_mode = "loop"
        self._update_mode_label()
        self._save_config()

    def _update_mode_label(self):
        labels = {"loop": "列表循环 🔂", "shuffle": "随机播放 🔀"}
        self.btn_mode.config(text=labels.get(self.play_mode, "列表循环 🔂"))

    def _setup_keyboard(self):
        def on_press(key):
            if key == keyboard.Key.f9:
                self.root.after(0, self._toggle_play)
                return
            try:
                k = key.char.lower() if hasattr(key, 'char') and key.char else None
            except:
                return
            if k == 'a':
                self.left_open = True
            elif k == 's':
                self.left_open = False
            elif k == 'd':
                self.right_open = True
            elif k == 'w':
                self.right_open = False
            else:
                return
            with self.lock:
                self.cab_l.set_window(self.left_open)
                self.cab_r.set_window(self.right_open)
        self.kl = keyboard.Listener(on_press=on_press)
        self.kl.daemon = True
        self.kl.start()

    def _setup_audio(self):
        self.stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, callback=self._audio_cb, blocksize=BLOCK_SIZE, dtype='float32')
        self.stream.start()

    def _audio_cb(self, outdata, frames, time_info, status):
        if self.audio_data is None or not self.playing:
            outdata[:, :] = 0
            return
        with self.lock:
            remaining = len(self.audio_data) - self.current_pos
            if remaining <= 0:
                outdata[:, :] = 0
                self.playing = False
                self._auto_next = True
                return
            n = min(frames, remaining)
            chunk = self.audio_data[self.current_pos:self.current_pos + n].copy()
            if n < frames:
                chunk = np.vstack([chunk, np.zeros((frames - n, 2))])
            left = self.cab_l.process(chunk[:, 0])
            right = self.cab_r.process(chunk[:, 1])
            vol = self.volume
            outdata[:, 0] = left * vol
            outdata[:, 1] = right * vol
            self.current_pos += n
            if self.current_pos >= len(self.audio_data):
                self.playing = False
                self._auto_next = True

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="选择音乐文件夹")
        if not folder:
            return
        self._load_folder(folder)
        self._save_config()

    def _load_folder(self, folder):
        self.playlist = []
        self.listbox.delete(0, tk.END)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(EXTENSIONS):
                self.playlist.append(os.path.join(folder, f))
                self.listbox.insert(tk.END, os.path.splitext(f)[0])
        self.lbl_status.config(text=f"已加载 {len(self.playlist)} 首歌曲")
        if self.playlist:
            self.listbox.select_set(0)
            self.current_index = 0

    def _next_index(self):
        if not self.playlist:
            return -1
        if self.play_mode == "shuffle":
            if len(self.playlist) <= 1:
                return 0
            idx = random.randint(0, len(self.playlist) - 1)
            while idx == self.current_index:
                idx = random.randint(0, len(self.playlist) - 1)
            return idx
        else:
            return (self.current_index + 1) % len(self.playlist)

    def _load_song(self, index, seek_sec=0, auto_play=True):
        if not self.playlist or index < 0 or index >= len(self.playlist):
            return
        self.playing = False
        self.lbl_status.config(text="加载中...")

        def _do():
            try:
                path = self.playlist[index]
                decoded = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=2, sample_rate=SAMPLE_RATE)
                raw = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
                if decoded.nchannels == 1:
                    raw = np.column_stack([raw, raw]).flatten()
                stereo = raw.reshape(-1, 2)
                seek_pos = int(seek_sec * SAMPLE_RATE)
                if seek_pos >= len(stereo):
                    seek_pos = 0
                with self.lock:
                    self.audio_data = stereo
                    self.current_pos = seek_pos
                    self.current_index = index
                    self.playing = auto_play
                name = os.path.splitext(os.path.basename(path))[0]
                self.root.after(0, lambda: self._update_play_ui(name, index, auto_play))
            except Exception as e:
                self.root.after(0, lambda: self.lbl_status.config(text=f"加载失败: {e}"))
        threading.Thread(target=_do, daemon=True).start()

    def _update_play_ui(self, name, index, auto_play):
        self.lbl_song.config(text=name)
        self.listbox.select_clear(0, tk.END)
        self.listbox.select_set(index)
        self.listbox.see(index)
        if auto_play:
            self.lbl_status.config(text="正在播放")
            self.btn_play.config(text="⏸")
        else:
            self.lbl_status.config(text="已暂停")
            self.btn_play.config(text="▶")

    def _toggle_play(self):
        if self.audio_data is None:
            if self.playlist:
                self._load_song(0)
            return
        self.playing = not self.playing
        if self.playing:
            self.btn_play.config(text="⏸")
            self.lbl_status.config(text="正在播放")
        else:
            self.btn_play.config(text="▶")
            self.lbl_status.config(text="已暂停")

    def _next(self):
        if not self.playlist:
            return
        self._load_song(self._next_index())

    def _prev(self):
        if not self.playlist:
            return
        self._load_song((self.current_index - 1) % len(self.playlist))

    def _on_select(self):
        sel = self.listbox.curselection()
        if sel:
            self._load_song(sel[0])

    def _on_vol(self, val):
        self.volume = float(val)

    def _tick(self):
        l = "开" if self.left_open else "关"
        r = "开" if self.right_open else "关"
        self.lbl_win.config(text=f"左窗: {l} | 右窗: {r}")

        if self.audio_data is not None:
            cur = self.current_pos / SAMPLE_RATE
            total = len(self.audio_data) / SAMPLE_RATE
            self.lbl_time.config(text=f"{int(cur)//60}:{int(cur)%60:02d} / {int(total)//60}:{int(total)%60:02d}")

        if self._auto_next:
            self._auto_next = False
            self._next()

        self.root.after(200, self._tick)

    def _on_close(self):
        self._save_config()
        self.playing = False
        self.stream.stop()
        self.stream.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = App()
    app.run()
