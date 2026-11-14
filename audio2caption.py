#!/usr/bin/env python3
"""Live desktop-audio subtitle translator (offline, faster-whisper, CUDA).

Captures the default PulseAudio/PipeWire sink monitor with ffmpeg, segments
audio into utterances by energy, translates to English with Whisper, and shows
the result in a PyQt5 glass HUD and click-through subtitle overlay matching NeoLens.
"""
import ctypes
import glob
import os
import queue
import signal
import subprocess
import sys
import threading
import time

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


def _preload_cuda_libs():
    """Preload venv NVIDIA wheels and fall back to system CUDA 12 libs for CTranslate2."""
    patterns = [
        os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "*", "lib", "*.so*"),
        "/usr/local/lib/ollama/cuda_v12/libcudart.so.12",
        "/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12",
        "/usr/local/lib/ollama/cuda_v12/libcublas.so.12",
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cuda_libs()
from faster_whisper import WhisperModel

SR = 16000
BLOCK = SR // 20                 # 50 ms blocks (800 samples)
SILENCE_RMS = 0.008              # energy threshold for "speech"
END_SILENCE_BLOCKS = 6           # 0.30 s of quiet ends an utterance
MIN_BLOCKS, MAX_BLOCKS = 6, 56   # 0.30 s .. 2.8 s per chunk
LANGS = {"KO": "ko", "JA": "ja", "ZH": "zh", "RU": "ru"}
JUNK = ("thank you for watching", "subscribe", "thanks for watching", "subtitles by")

state = {"lang": "ko", "running": True, "paused": False}
chunks = queue.Queue()
texts = queue.Queue()


def _clear_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _warmup(model):
    """Warm up CUDA kernels and Silero VAD so the first real utterance has no cold start."""
    segs, _ = model.transcribe(
        np.zeros(SR // 2, dtype=np.float32),
        language=state["lang"],
        task="translate",
        beam_size=1,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    list(segs)
    return model


def load_model():
    try:
        m = WhisperModel("small", device="cuda", compute_type="float16", local_files_only=True)
        print("[NeoLens Audio] loaded cached model: small")
        return _warmup(m)
    except Exception:
        print("[NeoLens Audio] no cached model, downloading 'small'")
        return _warmup(WhisperModel("small", device="cuda", compute_type="float16"))


def start_ffmpeg():
    sink = subprocess.run(
        ["pactl", "get-default-sink"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"[NeoLens Audio] capturing {sink}.monitor", flush=True)
    return subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "warning", "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0",
            "-f", "pulse", "-fragment_size", str(BLOCK * 2), "-i", f"{sink}.monitor",
            "-ac", "1", "-ar", str(SR), "-f", "s16le", "-flush_packets", "1", "-",
        ],
        stdout=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        bufsize=BLOCK * 2,
    )


def capture(proc):
    """Read PCM, cut into utterances at pauses (or at the quietest point if too long)."""
    buf, rms_hist, quiet, prev = [], [], 0, None
    got_audio = False
    try:
        while state["running"]:
            raw = proc.stdout.read(BLOCK * 2)
            if not raw or len(raw) < BLOCK * 2:
                if state["running"]:
                    rc = proc.poll()
                    print(f"[NeoLens Audio] ffmpeg stream ended (rc={rc})", flush=True)
                break
            if state["paused"]:
                buf, rms_hist, quiet, prev = [], [], 0, None
                continue
            block = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(block ** 2)))
            loud = rms > SILENCE_RMS
            if loud and not got_audio:
                got_audio = True
                print(f"[NeoLens Audio] audio detected (rms={rms:.4f})", flush=True)
            if not buf and not loud:
                prev = block
                continue
            if not buf and loud and prev is not None:
                buf.append(prev)
                rms_hist.append(0.0)
                prev = None
            buf.append(block)
            rms_hist.append(rms)
            quiet = 0 if loud else quiet + 1
            if quiet >= END_SILENCE_BLOCKS or len(buf) >= MAX_BLOCKS:
                forced = len(buf) >= MAX_BLOCKS and quiet < END_SILENCE_BLOCKS
                if forced:
                    # No pause found: split at the quietest block of the last 0.8 s.
                    cut = len(buf) - 16 + int(np.argmin(rms_hist[-16:])) + 1
                    keep = max(0, cut - 2)
                else:
                    cut = len(buf)
                    keep = cut
                if cut - quiet >= MIN_BLOCKS:
                    while chunks.qsize() >= 1:  # Stay real-time: drop stale audio.
                        try:
                            chunks.get_nowait()
                        except queue.Empty:
                            break
                    chunks.put(np.concatenate(buf[:cut]))
                buf, rms_hist = buf[keep:], rms_hist[keep:]
                quiet = 0
    except Exception:
        import traceback
        traceback.print_exc()


def transcribe(model):
    try:
        while state["running"]:
            if state["paused"]:
                time.sleep(0.05)
                continue
            try:
                audio = chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.monotonic()
            segs, _ = model.transcribe(
                audio,
                language=state["lang"],
                task="translate",
                beam_size=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 100},
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            segs = list(segs)
            ms = int((time.monotonic() - t0) * 1000)
            out = " ".join(
                s.text.strip() for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0
            ).strip()
            if out and not any(j in out.lower() for j in JUNK):
                print(f"[NeoLens Audio] ({state['lang']}, {ms}ms) {out}", flush=True)
                texts.put((out, ms))
            elif segs:
                print(f"[NeoLens Audio] filtered {len(segs)} seg(s): {[s.text for s in segs]}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()


# ----------------------------------------------------------------------------- Overlay & Transparent Glass HUD
class SubtitleOverlay(QtWidgets.QWidget):
    """Click-through translucent bottom subtitle pill matching NeoLens Overlay."""

    def __init__(self, geom):
        super().__init__(
            None,
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.X11BypassWindowManagerHint
            | QtCore.Qt.WindowTransparentForInput,
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setGeometry(geom)
        self.text = "🎧 Listening…"
        self.peek_mode = False
        self.font_scale = 1.0
        self.font = QtGui.QFont("Sans Serif")
        self.font.setWeight(QtGui.QFont.DemiBold)

    def set_text(self, text):
        if self.text != text:
            self.text = text
            self.update()

    def set_peek(self, peek):
        if self.peek_mode != peek:
            self.peek_mode = peek
            self.update()

    def adjust_font_scale(self, delta):
        new_scale = round(max(0.70, min(1.60, self.font_scale + delta)), 2)
        if new_scale != self.font_scale:
            self.font_scale = new_scale
            self.update()
        return self.font_scale

    def paintEvent(self, _):
        if not self.text:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)

        fsize = max(12, int(round(24 * self.font_scale)))
        self.font.setPixelSize(fsize)
        p.setFont(self.font)

        max_w = max(320, int(self.width() * 0.76))
        flags = QtCore.Qt.TextWordWrap | QtCore.Qt.AlignCenter
        br = QtGui.QFontMetrics(self.font).boundingRect(0, 0, max_w - 36, 2000, flags, self.text)

        pill_w = min(max_w, max(200, br.width() + 36))
        pill_h = max(44, br.height() + 20)
        rx = (self.width() - pill_w) / 2.0
        ry = max(20.0, self.height() - pill_h - 60.0)
        rect = QtCore.QRectF(rx, ry, pill_w, pill_h)

        bg_alpha = 36 if self.peek_mode else 238
        fg_alpha = 40 if self.peek_mode else 255
        border_alpha = 20 if self.peek_mode else 95

        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 8.0, 8.0)
        p.fillPath(path, QtGui.QColor(15, 18, 26, bg_alpha))
        p.setPen(QtGui.QPen(QtGui.QColor(135, 140, 155, border_alpha), 1.0))
        p.drawPath(path)

        p.setPen(QtGui.QColor(255, 255, 255, fg_alpha))
        p.drawText(rect.adjusted(14, 6, -14, -6), flags, self.text)
        p.end()


class GlassHUD(QtWidgets.QWidget):
    """Floating dark-glass control bar matching NeoLens GlassHUD."""

    def __init__(self, ctl):
        super().__init__(
            None,
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.X11BypassWindowManagerHint,
        )
        self.ctl = ctl
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self._drag = None
        self.setWindowOpacity(0.95)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.frame = QtWidgets.QFrame()
        self.frame.setStyleSheet("""
            QFrame {
                background-color: rgba(15, 18, 26, 210);
                border: 1px solid rgba(255, 255, 255, 48);
                border-radius: 12px;
            }
            QLabel {
                background: transparent;
                border: none;
                color: rgba(235, 240, 255, 235);
                font: 600 9pt "Sans Serif";
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 22);
                color: rgba(230, 235, 250, 225);
                border: 1px solid rgba(255, 255, 255, 28);
                border-radius: 7px;
                padding: 4px 8px;
                font: 600 8.5pt "Sans Serif";
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 48);
            }
            QPushButton:checked {
                background-color: rgba(16, 185, 129, 215);
                color: #ffffff;
                border: 1px solid rgba(52, 211, 153, 255);
            }
            QPushButton#closeBtn {
                background-color: rgba(239, 68, 68, 190);
                color: white;
                padding: 4px 8px;
            }
            QPushButton#closeBtn:hover {
                background-color: rgba(248, 113, 113, 245);
            }
        """)
        outer.addWidget(self.frame)

        lay = QtWidgets.QHBoxLayout(self.frame)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(5)

        title = QtWidgets.QLabel("✦ NeoLens Audio")
        title.setStyleSheet("color: #38bdf8; font-weight: bold; padding-right: 2px;")
        title.setToolTip(
            "Scroll mouse wheel anywhere on HUD to scale subtitle font size (70% - 160%)"
        )
        lay.addWidget(title)

        self.grp = QtWidgets.QButtonGroup(self)
        for name, code in LANGS.items():
            btn = QtWidgets.QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(code == state["lang"])
            btn.clicked.connect(lambda _, c=code: self.ctl.set_lang(c))
            self.grp.addButton(btn)
            lay.addWidget(btn)

        self.peek_btn = QtWidgets.QPushButton("👁 Peek")
        self.peek_btn.setCheckable(True)
        self.peek_btn.setToolTip(
            "Ghost mode (15% opacity) — or hold Alt / right-click HUD"
        )
        self.peek_btn.toggled.connect(self.ctl.set_peek_toggle)
        lay.addWidget(self.peek_btn)

        self.copy_btn = QtWidgets.QPushButton("📋 Copy")
        self.copy_btn.setToolTip("Copy latest translated subtitle to clipboard")
        self.copy_btn.clicked.connect(self.ctl.copy_last_caption)
        lay.addWidget(self.copy_btn)

        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.ctl.set_paused)
        lay.addWidget(self.pause_btn)

        self.status = QtWidgets.QLabel("🎧 Listening…")
        self.status.setMinimumWidth(165)
        self.set_status_color("green")
        lay.addWidget(self.status)

        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setObjectName("closeBtn")
        close_btn.clicked.connect(self.ctl.shutdown)
        lay.addWidget(close_btn)

        # Auto-dim the HUD after 3 seconds of inactivity to minimize distraction.
        self.fade_timer = QtCore.QTimer(self)
        self.fade_timer.setSingleShot(True)
        self.fade_timer.timeout.connect(lambda: self.setWindowOpacity(0.50))
        self.fade_timer.start(3200)

    def enterEvent(self, _):
        self.fade_timer.stop()
        self.setWindowOpacity(0.98)

    def leaveEvent(self, _):
        self.fade_timer.start(2800)

    def wheelEvent(self, e):
        delta = 0.08 if e.angleDelta().y() > 0 else -0.08
        scale = self.ctl.overlay.adjust_font_scale(delta)
        self.set_status_color("green")
        self.status.setText(f"🔠 Font Size: {int(scale * 100)}%")

    def set_status_color(self, level):
        colors = {
            "green": "rgba(110, 231, 183, 245)",
            "amber": "rgba(251, 191, 36, 245)",
            "red": "rgba(248, 113, 113, 250)",
        }
        c = colors.get(level, colors["green"])
        self.status.setStyleSheet(
            f"color: {c}; font: 600 8.5pt 'Sans Serif'; padding-left: 4px;"
        )

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._drag = e.globalPos() - self.frameGeometry().topLeft()
        elif e.button() == QtCore.Qt.RightButton:
            self.peek_btn.setChecked(not self.peek_btn.isChecked())

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPos() - self._drag)

    def mouseReleaseEvent(self, _):
        self._drag = None


# ----------------------------------------------------------------------------- Controller
class AudioController(QtCore.QObject):
    def __init__(self, app, proc):
        super().__init__()
        self.app = app
        self.proc = proc
        self.peek_toggled = False
        self.last_caption = ""

        screen = QtGui.QGuiApplication.primaryScreen()
        geom = screen.geometry() if screen else QtCore.QRect(0, 0, 1920, 1080)

        self.overlay = SubtitleOverlay(geom)
        self.hud = GlassHUD(self)
        self.overlay.show()
        self.hud.adjustSize()
        self.hud.move(
            max(geom.left() + 10, geom.left() + geom.width() - self.hud.width() - 24),
            geom.top() + 32,
        )
        self.hud.show()
        self.hud.raise_()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)

    def set_lang(self, code):
        state["lang"] = code
        _clear_queue(chunks)
        _clear_queue(texts)
        self.last_caption = ""
        if not state["paused"]:
            self.overlay.set_text("🎧 Listening…")
            self.hud.set_status_color("green")
            self.hud.status.setText(f"🎧 Listening ({code.upper()})")

    def set_peek_toggle(self, checked):
        self.peek_toggled = checked
        self._update_modifiers()

    def _update_modifiers(self):
        mods = QtWidgets.QApplication.queryKeyboardModifiers()
        ctrl = bool(mods & QtCore.Qt.ControlModifier)
        shift = bool(mods & QtCore.Qt.ShiftModifier)
        alt = bool(mods & QtCore.Qt.AltModifier)
        want_peek = self.peek_toggled or (alt and not ctrl and not shift)
        if want_peek != self.overlay.peek_mode:
            self.overlay.set_peek(want_peek)

    def copy_last_caption(self):
        if not self.last_caption:
            self.hud.set_status_color("amber")
            self.hud.status.setText("📋 No lines to copy yet")
            return
        QtWidgets.QApplication.clipboard().setText(self.last_caption)
        self.hud.set_status_color("green")
        self.hud.status.setText("📋 Copied to clipboard!")

    def set_paused(self, paused):
        state["paused"] = paused
        _clear_queue(chunks)
        _clear_queue(texts)
        if paused:
            self.overlay.set_text("")
            self.hud.set_status_color("amber")
            self.hud.status.setText("⏸ Paused")
        else:
            self.overlay.set_text(self.last_caption or "🎧 Listening…")
            self.hud.set_status_color("green")
            self.hud.status.setText(f"🎧 Listening ({state['lang'].upper()})")

    def shutdown(self):
        state["running"] = False
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.app.quit()

    def tick(self):
        if not state["running"]:
            self.shutdown()
            return
        self._update_modifiers()
        latest = None
        try:
            while True:
                latest = texts.get_nowait()
        except queue.Empty:
            pass
        if latest is not None and not state["paused"]:
            out, ms = latest
            self.last_caption = out
            self.overlay.set_text(out)
            level = "green" if ms < 400 else ("amber" if ms <= 800 else "red")
            self.hud.set_status_color(level)
            self.hud.status.setText(f"⚡ Whisper: {ms}ms")


def main():
    model = load_model()
    proc = start_ffmpeg()

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    ctl = AudioController(app, proc)

    def on_signal(*_):
        state["running"] = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    threading.Thread(target=capture, args=(proc,), daemon=True).start()
    threading.Thread(target=transcribe, args=(model,), daemon=True).start()

    try:
        rc = app.exec_()
    finally:
        ctl.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
