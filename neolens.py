#!/usr/bin/env python3
"""NeoLens - Ultra-low latency (<0.5s SLA) full-screen translator.

Runs locally on GPU using RapidOCR and CTranslate2 (NLLB-200 int8_float16).
"""

import argparse
import ctypes
from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

HERE = Path(__file__).resolve().parent
model_dir = str(HERE / "models" / "nllb-200-ct2")


def _preload_cuda_libs():
    """Preload local CUDA and cuDNN shared libraries so CTranslate2 can resolve CUDA dependencies."""
    patterns = [
        str(HERE / "bin" / "llama" / "libcu*.so*"),
        os.path.join(
            sys.prefix, "lib", "python*", "site-packages", "nvidia", "*", "lib", "*.so*"
        ),
    ]
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cuda_libs()

import ctranslate2
import cv2
import mss
import numpy as np
import transformers
from huggingface_hub import snapshot_download
from PyQt5 import QtCore, QtGui, QtWidgets
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

# ----------------------------------------------------------------------------- Configuration & Presets
CT2_REPO = "entai2965/nllb-200-distilled-600M-ctranslate2"
TOK_REPO = "facebook/nllb-200-distilled-600M"

LANGS = {"JA": "ja", "KO": "ko", "ZH": "zh", "RU": "ru"}
NLLB_CODE = {"ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans", "ru": "rus_Cyrl"}
SCRIPT_RE = {
    "ja": re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]"),
    "ko": re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]"),
    "zh": re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]"),
    "ru": re.compile(r"[\u0400-\u04ff]"),
}

THUMB = 4
DIFF_T = 25
DIFF_N = 15
SLA_CAP_MS = 500.0

PRESET_ORDER = ["movie", "fast", "hq"]
PRESETS = {
    "movie": {
        "key": "movie",
        "label": "🎬 Movie / Sub",
        "short": "Movie",
        "budget_ms": 150.0,
        "roi_mode": "subtitles",
        "limit_side_len": 800,  # Capped at 800 (down from 960) to keep OCR < 100ms and avoid 25% auto-downgrade
        "thresh": 0.25,
        "box_thresh": 0.35,
        "text_score": 0.40,
        "unclip_ratio": 1.6,
        "beam_size": 1,
        "max_decoding_length": 48,
        "repetition_penalty": 1.0,
        "tick_ms": 70,
        "stable_ticks": 1,
        "hide_ms": 25,
    },
    "fast": {
        "key": "fast",
        "label": "⚡ Fast Read",
        "short": "Fast Read",
        "budget_ms": 320.0,
        "roi_mode": "full",
        "limit_side_len": 1440,
        "thresh": 0.22,
        "box_thresh": 0.28,
        "text_score": 0.30,
        "unclip_ratio": 1.6,
        "beam_size": 1,
        "max_decoding_length": 80,
        "repetition_penalty": 1.15,
        "tick_ms": 100,
        "stable_ticks": 2,
        "hide_ms": 55,
    },
    "hq": {
        "key": "hq",
        "label": "🔍 HQ Manga",
        "short": "HQ Manga",
        "budget_ms": 500.0,
        "roi_mode": "full",
        "limit_side_len": 1920,
        "thresh": 0.20,
        "box_thresh": 0.22,
        "text_score": 0.22,
        "unclip_ratio": 1.7,
        "beam_size": 2,
        "max_decoding_length": 110,
        "repetition_penalty": 1.2,
        "tick_ms": 115,
        "stable_ticks": 2,
        "hide_ms": 65,
    },
}


def resolve_preset(name_or_key):
    if isinstance(name_or_key, dict):
        return name_or_key
    s = str(name_or_key or "fast").lower().strip()
    if s in PRESETS:
        return PRESETS[s]
    for k, cfg in PRESETS.items():
        if k in s or cfg["short"].lower() in s or cfg["label"].lower() in s:
            return cfg
    return PRESETS["fast"]


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str

    @property
    def w(self):
        return max(1, self.x1 - self.x0)

    @property
    def h(self):
        return max(1, self.y1 - self.y0)


@dataclass
class RenderBlock:
    x0: int
    y0: int
    x1: int
    y1: int
    src: str
    en: str
    bg: tuple
    fg: tuple


# ----------------------------------------------------------------------------- Core Engine
class Engine:
    """Preloads OCR and CTranslate2 models on the main thread to prevent threading deadlocks."""

    def __init__(self):
        print("1/3 [NeoLens] Loading NLLB-200 CTranslate2 (int8_float16 on CUDA)...")
        local_model_dir = str(
            Path(__file__).resolve().parent / "models" / "nllb-200-ct2"
        )
        if not Path(local_model_dir).exists():
            snapshot_download(CT2_REPO, local_dir=local_model_dir)
        try:
            self.translator = ctranslate2.Translator(
                local_model_dir,
                device="cuda",
                compute_type="int8_float16",
                inter_threads=2,
            )
            print("    -> CTranslate2 running on CUDA (int8_float16)")
        except Exception as e:
            print(f"    -> CUDA fallback to CPU ({e})")
            self.translator = ctranslate2.Translator(
                local_model_dir, device="cpu", compute_type="int8"
            )

        print("2/3 [NeoLens] Loading Tokenizer...")
        self.tokenizers = {}
        for code, nllb in NLLB_CODE.items():
            try:
                tok = transformers.AutoTokenizer.from_pretrained(
                    TOK_REPO, src_lang=nllb, local_files_only=True
                )
            except Exception:
                try:
                    tok = transformers.AutoTokenizer.from_pretrained(
                        TOK_REPO, src_lang=nllb
                    )
                except Exception:
                    tok = transformers.AutoTokenizer.from_pretrained(
                        local_model_dir, src_lang=nllb, local_files_only=True
                    )
            self.tokenizers[code] = tok

        print("3/3 [NeoLens] Loading RapidOCR Engine...")
        self.ocrs = {}
        self.cache = {}
        self.preset_key = "fast"
        self.limit_scale = 1.0  # Automatically scaled down by 25% when total latency exceeds 500ms
        ocr_ja = self.get_ocr("ja")
        # Warm up OCR detector, recognizer, and CUDA translator to eliminate first-frame cold-start latency.
        dummy = np.full((48, 160, 3), 240, dtype=np.uint8)
        ocr_ja._load_det_model()
        ocr_ja._load_rec_model()
        ocr_ja(dummy)
        ocr_ja.recognize_txt([dummy])
        for _ in range(2):
            self.translate_batch(
                "ja", ["テスト", "設定メニューを開く"], preset="movie", use_cache=False
            )
        self.cache.clear()
        print("[NeoLens] READY! All models warmed up.\n")

    def set_preset(self, preset_key):
        cfg = resolve_preset(preset_key)
        self.preset_key = cfg["key"]
        self.limit_scale = 1.0
        return cfg

    def get_ocr(self, lang, preset=None):
        key = "zh" if lang in ("ja", "zh") else lang
        cfg = resolve_preset(preset or self.preset_key)
        eff_limit = max(
            512, int(round((cfg["limit_side_len"] * self.limit_scale) / 32) * 32)
        )

        if key not in self.ocrs:
            params = {
                "Global.log_level": "error",
                "Global.use_cls": False,
                "EngineConfig.onnxruntime.use_cuda": True,
                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
                "Det.model_type": ModelType.TINY,
                "Det.limit_type": "max",
                "Det.limit_side_len": eff_limit,
                "Global.text_score": cfg["text_score"],
                "Det.thresh": cfg["thresh"],
                "Det.box_thresh": cfg["box_thresh"],
                "Det.unclip_ratio": cfg["unclip_ratio"],
            }
            if key == "ko":
                params.update(
                    {
                        "Rec.lang_type": LangRec.KOREAN,
                        "Rec.ocr_version": OCRVersion.PPOCRV5,
                        "Rec.model_type": ModelType.MOBILE,
                    }
                )
            elif key == "ru":
                params.update(
                    {
                        "Rec.lang_type": LangRec.ESLAV,
                        "Rec.ocr_version": OCRVersion.PPOCRV5,
                        "Rec.model_type": ModelType.MOBILE,
                    }
                )
            self.ocrs[key] = RapidOCR(params=params)

        ocr = self.ocrs[key]
        ocr.cfg.Det.limit_side_len = eff_limit
        ocr.cfg.Det.thresh = cfg["thresh"]
        ocr.cfg.Det.box_thresh = cfg["box_thresh"]
        ocr.cfg.Det.unclip_ratio = cfg["unclip_ratio"]
        ocr.cfg.Global.text_score = cfg["text_score"]
        ocr.text_score = cfg["text_score"]
        if getattr(ocr, "text_det", None) is not None:
            ocr.text_det.limit_side_len = eff_limit
            ocr.text_det.postprocess_op.thresh = cfg["thresh"]
            ocr.text_det.postprocess_op.box_thresh = cfg["box_thresh"]
            ocr.text_det.postprocess_op.unclip_ratio = cfg["unclip_ratio"]
        return ocr

    def translate_batch(self, lang, texts, preset=None, use_cache=True):
        if not texts:
            return []
        cfg = resolve_preset(preset or self.preset_key)
        tok = self.tokenizers[lang]
        out = [None] * len(texts)
        todo_idx, todo_txt = [], []

        for i, t in enumerate(texts):
            k = (lang, cfg["beam_size"], re.sub(r"\s+", "", t))
            if use_cache and k in self.cache:
                out[i] = self.cache[k]
            else:
                todo_idx.append(i)
                todo_txt.append(t)

        if todo_txt:
            src_tokens = [
                tok.convert_ids_to_tokens(
                    tok.encode(t, truncation=True, max_length=128)
                )
                for t in todo_txt
            ]
            prefix = [["eng_Latn"]] * len(todo_txt)
            results = self.translator.translate_batch(
                src_tokens,
                target_prefix=prefix,
                beam_size=cfg["beam_size"],
                max_decoding_length=cfg["max_decoding_length"],
                repetition_penalty=cfg["repetition_penalty"],
            )
            for idx, raw_t, res in zip(todo_idx, todo_txt, results):
                hyp = res.hypotheses[0][1:]
                en = tok.decode(
                    tok.convert_tokens_to_ids(hyp), skip_special_tokens=True
                ).strip()
                k = (lang, cfg["beam_size"], re.sub(r"\s+", "", raw_t))
                self.cache[k] = en
                out[idx] = en
        return out

    @staticmethod
    def _trim_active_bbox(img):
        """Trim blank screen margins using a fast (<0.8ms) quarter-scale gradient bounding box."""
        H, W = img.shape[:2]
        if H < 64 or W < 64:
            return 0, 0, W, H
        g = img[::4, ::4, 1].astype(np.int16)
        edges = (
            np.abs(g[1:, :-1] - g[:-1, :-1]) + np.abs(g[:-1, 1:] - g[:-1, :-1])
        ) > 16
        ys, xs = np.nonzero(edges)
        if len(xs) == 0:
            return 0, 0, W, H
        pad = 28
        x0 = max(0, int(xs.min()) * 4 - pad)
        y0 = max(0, int(ys.min()) * 4 - pad)
        x1 = min(W, int(xs.max()) * 4 + pad + 4)
        y1 = min(H, int(ys.max()) * 4 + pad + 4)
        if (x1 - x0) < 32 or (y1 - y0) < 24:
            return 0, 0, W, H
        return x0, y0, x1, y1

    def scan_image(
        self,
        img,
        lang="ja",
        preset=None,
        custom_roi=None,
        dirty_rect=None,
        panel_rect=None,
        use_cache=True,
    ):
        """Run ROI cropping, RapidOCR, box clustering, and CTranslate2 translation within the <500ms SLA."""
        t_start = time.monotonic()
        cfg = resolve_preset(preset or self.preset_key)
        work = img[:, :, :3].copy()
        H, W = work.shape[:2]

        if panel_rect is not None:
            px0, py0, px1, py1 = panel_rect
            work[max(0, py0) : min(H, py1), max(0, px0) : min(W, px1)] = 0

        # 1. Determine the ROI from custom selection or the active preset policy.
        if custom_roi is not None:
            rx0, ry0, rx1, ry1 = custom_roi
            rx0, ry0 = max(0, int(rx0)), max(0, int(ry0))
            rx1, ry1 = min(W, int(rx1)), min(H, int(ry1))
        elif cfg["roi_mode"] == "subtitles":
            sub_y0, sub_y1 = int(0.65 * H), int(0.90 * H)
            rx0, ry0, rx1, ry1 = int(0.10 * W), sub_y0, int(0.90 * W), sub_y1
            if dirty_rect is not None:
                dx0, dy0, dx1, dy1 = dirty_rect
                if (dx1 - dx0) * (dy1 - dy0) < (rx1 - rx0) * (ry1 - ry0):
                    rx0, ry0, rx1, ry1 = (
                        max(0, dx0),
                        max(0, dy0),
                        min(W, dx1),
                        min(H, dy1),
                    )
        elif dirty_rect is not None:
            dx0, dy0, dx1, dy1 = dirty_rect
            rx0, ry0, rx1, ry1 = max(0, dx0), max(0, dy0), min(W, dx1), min(H, dy1)
        else:
            rx0, ry0, rx1, ry1 = 0, 0, W, H

        roi_img = work[ry0:ry1, rx0:rx1]
        tx0, ty0, tx1, ty1 = self._trim_active_bbox(roi_img)
        active_img = np.ascontiguousarray(roi_img[ty0:ty1, tx0:tx1])
        ox, oy = rx0 + tx0, ry0 + ty0

        # 2. Run OCR using the active preset thresholds.
        ocr = self.get_ocr(lang, cfg)
        eff_limit = ocr.cfg.Det.limit_side_len
        t_ocr0 = time.monotonic()
        res = ocr(
            active_img,
            text_score=cfg["text_score"],
            box_thresh=cfg["box_thresh"],
            unclip_ratio=cfg["unclip_ratio"],
        )
        t_ocr = (time.monotonic() - t_ocr0) * 1000.0

        boxes = []
        if res and res.boxes is not None and res.txts is not None:
            for box, txt, sc in zip(res.boxes, res.txts, res.scores):
                txt = (txt or "").strip()
                if sc < cfg["text_score"] or not txt or not SCRIPT_RE[lang].search(txt):
                    continue
                xs, ys = box[:, 0], box[:, 1]
                boxes.append(
                    Box(
                        int(xs.min()) + ox,
                        int(ys.min()) + oy,
                        int(xs.max()) + ox,
                        int(ys.max()) + oy,
                        txt,
                    )
                )

        # 3. Cluster detected text boxes using orientation-aware geometry.
        groups = group_boxes(boxes, lang)

        # 4. Translate clustered text batches via CTranslate2.
        t_tr0 = time.monotonic()
        translations = self.translate_batch(
            lang, [g[4] for g in groups], preset=cfg, use_cache=use_cache
        )
        t_tr = (time.monotonic() - t_tr0) * 1000.0

        blocks = []
        for (x0, y0, x1, y1, src), en in zip(groups, translations):
            if not en:
                continue
            bg, fg = sample_style(work, x0, y0, x1, y1)
            blocks.append(RenderBlock(x0, y0, x1, y1, src, en, bg, fg))

        t_total = (time.monotonic() - t_start) * 1000.0
        downgraded = False
        if t_total > SLA_CAP_MS:
            # Reduce Det.limit_side_len by 25% for subsequent frames to maintain the <500ms SLA.
            self.limit_scale = max(0.5, self.limit_scale * 0.75)
            downgraded = True

        metrics = {
            "preset": cfg["short"],
            "budget_ms": cfg["budget_ms"],
            "ocr_ms": t_ocr,
            "trans_ms": t_tr,
            "total_ms": t_total,
            "limit_side_len": eff_limit,
            "downgraded": downgraded,
            "count": len(blocks),
        }
        return blocks, metrics


# ----------------------------------------------------------------------------- Grouping & Color Sampling
def sample_style(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1, :3]
    if crop.size == 0:
        return (250, 250, 250), (15, 15, 15)
    border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]], axis=0)
    bg_bgr = np.median(border, axis=0)
    lum = 0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2]
    if lum > 135:
        return (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0])), (15, 15, 20)
    return (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0])), (245, 245, 250)


def group_boxes(boxes, lang):
    """Cluster text boxes using orientation-aware rules for vertical manga and horizontal UI/subtitles."""
    if not boxes:
        return []
    used = [False] * len(boxes)
    clusters = []

    for i, b in enumerate(boxes):
        if used[i]:
            continue
        used[i] = True
        cluster = [b]
        changed = True
        while changed:
            changed = False
            cx0 = min(x.x0 for x in cluster)
            cy0 = min(x.y0 for x in cluster)
            cx1 = max(x.x1 for x in cluster)
            cy1 = max(x.y1 for x in cluster)
            avg_char = max(10.0, float(np.median([min(x.w, x.h) for x in cluster])))
            cluster_is_vert = (lang == "ja") and any(x.h > x.w * 1.3 for x in cluster)

            for j, o in enumerate(boxes):
                if used[j]:
                    continue
                dx = max(0, max(cx0 - o.x1, o.x0 - cx1))
                dy = max(0, max(cy0 - o.y1, o.y0 - cy1))
                cand_is_vert = (lang == "ja") and (o.h > o.w * 1.3)

                if cluster_is_vert and (cand_is_vert or o.w <= avg_char * 1.5):
                    # Vertical manga speech bubbles (Japanese with h > w * 1.3):
                    # merge adjacent columns right-to-left within proximity thresholds.
                    can_merge = dx < avg_char * 0.95 and dy < avg_char * 0.8
                elif not cluster_is_vert and not cand_is_vert:
                    # Horizontal text (web UI, subtitles):
                    # merge boxes within strict horizontal and vertical gap thresholds.
                    can_merge = dx < avg_char * 0.75 and dy < avg_char * 0.6
                else:
                    can_merge = False

                if can_merge:
                    used[j] = True
                    cluster.append(o)
                    changed = True
        clusters.append(cluster)

    merged = []
    for cl in clusters:
        x0 = min(x.x0 for x in cl)
        y0 = min(x.y0 for x in cl)
        x1 = max(x.x1 for x in cl)
        y1 = max(x.y1 for x in cl)
        is_vert = (lang == "ja") and any(x.h > x.w * 1.3 for x in cl)
        if is_vert:
            col_w = max(12, int(np.median([x.w for x in cl])))
            cl.sort(key=lambda b: (-round(b.x1 / col_w), b.y0))
            txt = "".join(x.text for x in cl)
        else:
            row_h = max(10, int(np.median([x.h for x in cl])))
            cl.sort(key=lambda b: (round(b.y0 / max(10, int(row_h * 0.7))), b.x0))
            joiner = "" if lang in ("ja", "zh") else " "
            txt = joiner.join(x.text for x in cl)
        merged.append((x0, y0, x1, y1, txt))
    return merged


# ----------------------------------------------------------------------------- Background Worker
class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(list, dict)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.busy = False
        self._lock = threading.Lock()

    def process(self, img, lang, preset_key, custom_roi, dirty_rect, panel_rect):
        with self._lock:
            if self.busy:
                return
            self.busy = True
        threading.Thread(
            target=self._run,
            args=(img, lang, preset_key, custom_roi, dirty_rect, panel_rect),
            daemon=True,
        ).start()

    def _run(self, img, lang, preset_key, custom_roi, dirty_rect, panel_rect):
        try:
            blocks, metrics = self.engine.scan_image(
                img,
                lang=lang,
                preset=preset_key,
                custom_roi=custom_roi,
                dirty_rect=dirty_rect,
                panel_rect=panel_rect,
                use_cache=True,
            )
            self.finished.emit(blocks, metrics)
        except Exception as e:
            self.finished.emit(
                [],
                {
                    "error": str(e),
                    "ocr_ms": 0.0,
                    "trans_ms": 0.0,
                    "total_ms": 0.0,
                    "budget_ms": 320.0,
                },
            )
        finally:
            with self._lock:
                self.busy = False


# ----------------------------------------------------------------------------- Interactive ROI Selector
class ROISelector(QtWidgets.QWidget):
    roi_selected = QtCore.pyqtSignal(object)  # Emits (x0, y0, x1, y1) or None

    def __init__(self, geom):
        super().__init__(
            None,
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.X11BypassWindowManagerHint,
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setGeometry(geom["left"], geom["top"], geom["width"], geom["height"])
        self.origin = None
        self.current = None

    def select_start(self):
        self.origin = None
        self.current = None
        self.show()
        self.raise_()
        self.activateWindow()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.origin = e.pos()
            self.current = e.pos()
            self.update()
        elif e.button() == QtCore.Qt.RightButton:
            self.hide()
            self.roi_selected.emit(None)

    def mouseMoveEvent(self, e):
        if self.origin is not None:
            self.current = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and self.origin is not None:
            end = e.pos()
            x0, x1 = sorted((self.origin.x(), end.x()))
            y0, y1 = sorted((self.origin.y(), end.y()))
            self.origin = None
            self.current = None
            self.hide()
            if (x1 - x0) >= 32 and (y1 - y0) >= 24:
                self.roi_selected.emit((x0, y0, x1, y1))
            else:
                self.roi_selected.emit(None)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.hide()
            self.roi_selected.emit(None)

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(10, 14, 24, 95))
        if self.origin is not None and self.current is not None:
            r = QtCore.QRect(self.origin, self.current).normalized()
            p.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            p.fillRect(r, QtCore.Qt.transparent)
            p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            pen = QtGui.QPen(QtGui.QColor(56, 189, 248, 245), 2.0, QtCore.Qt.DashLine)
            p.setPen(pen)
            p.drawRect(r)
        p.end()


# ----------------------------------------------------------------------------- Overlay & Transparent Glass HUD
class Overlay(QtWidgets.QWidget):
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
        self.setGeometry(geom["left"], geom["top"], geom["width"], geom["height"])
        self.blocks = []
        self.painted_rects = []
        self.hidden_for_grab = False
        self.peek_mode = False
        self.bilingual = False
        self.sub_dock = False  # When True (Movie mode), renders above original subtitles without hiding the overlay
        self.font_scale = 1.0
        self.hovered_idx = -1
        self.roi_rect = None
        self.font = QtGui.QFont("Sans Serif")
        self.font.setWeight(QtGui.QFont.DemiBold)

    def set_blocks(self, blocks):
        self.blocks = blocks
        if not blocks:
            self.painted_rects = []
            self.hovered_idx = -1
        self.update()

    def set_peek(self, peek):
        if self.peek_mode != peek:
            self.peek_mode = peek
            self.update()

    def set_bilingual(self, bi):
        if self.bilingual != bi:
            self.bilingual = bi
            self.update()

    def adjust_font_scale(self, delta):
        new_scale = round(max(0.70, min(1.60, self.font_scale + delta)), 2)
        if new_scale != self.font_scale:
            self.font_scale = new_scale
            self.update()
        return self.font_scale

    def set_roi_rect(self, roi_rect):
        self.roi_rect = roi_rect
        self.update()

    def fit_font(self, text, w, h):
        base = int(np.sqrt((w * h) / max(1, len(text))) * 1.15 * self.font_scale)
        size = min(32, max(8, base))
        flags = QtCore.Qt.TextWordWrap
        avail_w = max(24, w - 8)
        while size > 8:
            self.font.setPixelSize(size)
            br = QtGui.QFontMetrics(self.font).boundingRect(
                0, 0, avail_w, 2000, flags, text
            )
            if br.height() <= h * max(1.0, self.font_scale) and br.width() <= max(
                26, w - 4
            ):
                break
            size -= 1
        self.font.setPixelSize(size)
        br = QtGui.QFontMetrics(self.font).boundingRect(
            0, 0, avail_w, 2000, flags, text
        )
        return (
            size,
            max(h, br.height() + 6),
            max(w, min(int(w * 1.45), br.width() + 12)),
        )

    def paintEvent(self, _):
        if self.hidden_for_grab:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)

        rects = []
        if self.roi_rect is not None:
            rx0, ry0, rx1, ry1 = self.roi_rect
            r_rect = QtCore.QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0)
            pen = QtGui.QPen(QtGui.QColor(56, 189, 248, 115), 1.5, QtCore.Qt.DashLine)
            p.setPen(pen)
            p.drawRect(r_rect)

        bg_alpha = 36 if self.peek_mode else 238
        fg_alpha = 40 if self.peek_mode else 255

        for idx, b in enumerate(self.blocks):
            bw, bh = b.x1 - b.x0, b.y1 - b.y0
            pad = 4 if (bw < 60 or bh < 24) else 6
            w = max(36, bw + pad * 2)
            h = max(20, bh + pad * 2)

            display_txt = f"{b.src}\n{b.en}" if self.bilingual else b.en
            fsize, best_h, best_w = self.fit_font(display_txt, w, h)

            # Center horizontal expansion to keep subtitle pills balanced.
            cx = (b.x0 + b.x1) / 2.0
            rx = max(2.0, cx - best_w / 2.0)
            if self.sub_dock:
                # Dock the translation box above the source text so it never obscures subtitles or requires hiding during capture.
                ry = max(4.0, b.y0 - best_h - 8.0)
            else:
                ry = b.y0 - pad
            rect = QtCore.QRectF(rx, ry, best_w, best_h)
            rects.append(
                (
                    int(rect.left()),
                    int(rect.top()),
                    int(rect.right()),
                    int(rect.bottom()),
                )
            )

            path = QtGui.QPainterPath()
            radius = 5.0 if best_w < 52 or best_h < 26 else 8.0
            path.addRoundedRect(rect, radius, radius)

            p.fillPath(path, QtGui.QColor(b.bg[0], b.bg[1], b.bg[2], bg_alpha))

            is_hov = (idx == self.hovered_idx) and not self.peek_mode
            if is_hov:
                p.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 235), 1.8))
            else:
                border_alpha = 20 if self.peek_mode else 95
                p.setPen(QtGui.QPen(QtGui.QColor(135, 140, 155, border_alpha), 1.0))
            p.drawPath(path)

            self.font.setPixelSize(fsize)
            p.setFont(self.font)
            p.setPen(QtGui.QColor(b.fg[0], b.fg[1], b.fg[2], fg_alpha))
            inset = 3 if best_w < 52 else 5
            p.drawText(
                rect.adjusted(inset, 2, -inset, -2),
                QtCore.Qt.TextWordWrap | QtCore.Qt.AlignCenter,
                display_txt,
            )
        p.end()
        self.painted_rects = rects


class GlassHUD(QtWidgets.QWidget):
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
            QPushButton#modeBtn {
                background-color: rgba(139, 92, 246, 200);
                color: #ffffff;
                border: 1px solid rgba(167, 139, 250, 185);
            }
            QPushButton#modeBtn:hover {
                background-color: rgba(167, 139, 250, 230);
            }
            QPushButton#scanBtn {
                background-color: rgba(59, 130, 246, 205);
                color: white;
            }
            QPushButton#scanBtn:hover {
                background-color: rgba(96, 165, 250, 235);
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

        title = QtWidgets.QLabel("✦ NeoLens")
        title.setStyleSheet("color: #38bdf8; font-weight: bold; padding-right: 2px;")
        title.setToolTip(
            "Scroll mouse wheel anywhere on HUD to scale overlay font size (70% - 160%)"
        )
        lay.addWidget(title)

        self.grp = QtWidgets.QButtonGroup(self)
        for name, code in LANGS.items():
            btn = QtWidgets.QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(code == ctl.lang)
            btn.clicked.connect(lambda _, c=code: self.ctl.set_lang(c))
            self.grp.addButton(btn)
            lay.addWidget(btn)

        self.mode_btn = QtWidgets.QPushButton(PRESETS[ctl.preset_key]["label"])
        self.mode_btn.setObjectName("modeBtn")
        self.mode_btn.setToolTip(
            "Cycle Mode: Movie/Sub (≤150ms) · Fast Read (≤320ms) · HQ Manga (≤500ms)"
        )
        self.mode_btn.clicked.connect(self.ctl.cycle_preset)
        lay.addWidget(self.mode_btn)

        self.area_btn = QtWidgets.QPushButton("🎯 Area")
        self.area_btn.setCheckable(True)
        self.area_btn.setToolTip(
            "Drag a rectangle to lock translation inside an ROI (click again to reset)"
        )
        self.area_btn.clicked.connect(self.ctl.toggle_area_roi)
        lay.addWidget(self.area_btn)

        self.bi_btn = QtWidgets.QPushButton("A/文")
        self.bi_btn.setCheckable(True)
        self.bi_btn.setToolTip(
            "Toggle Dual-Subtitle mode (show both Original + English inside each box)"
        )
        self.bi_btn.toggled.connect(self.ctl.set_bilingual)
        lay.addWidget(self.bi_btn)

        self.peek_btn = QtWidgets.QPushButton("👁 Peek")
        self.peek_btn.setCheckable(True)
        self.peek_btn.setToolTip(
            "Ghost mode (15% opacity) — or hold Alt / right-click HUD"
        )
        self.peek_btn.toggled.connect(self.ctl.set_peek_toggle)
        lay.addWidget(self.peek_btn)

        self.copy_btn = QtWidgets.QPushButton("📋 Copy")
        self.copy_btn.setToolTip(
            "Copy last batch of [Original -> English] lines to clipboard"
        )
        self.copy_btn.clicked.connect(self.ctl.copy_last_batch)
        lay.addWidget(self.copy_btn)

        scan_btn = QtWidgets.QPushButton("⚡ Scan")
        scan_btn.setObjectName("scanBtn")
        scan_btn.setToolTip("Force rescan now (or press Ctrl+Shift+Alt anywhere)")
        scan_btn.clicked.connect(self.ctl.force_scan)
        lay.addWidget(scan_btn)

        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.ctl.set_paused)
        lay.addWidget(self.pause_btn)

        self.status = QtWidgets.QLabel("OCR: 0ms | Trans: 0ms | Total: 0ms")
        self.status.setMinimumWidth(265)
        self.set_status_color("green")
        lay.addWidget(self.status)

        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setObjectName("closeBtn")
        close_btn.clicked.connect(QtWidgets.QApplication.quit)
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
        self.status.setText(f"🔠 Font Size: {int(scale * 100)}% (scroll HUD to adjust)")

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
class Controller(QtCore.QObject):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.lang = "ja"
        self.preset_key = "fast"
        self.paused = False
        self.peek_toggled = False
        self.custom_roi = None
        self.last_blocks = []
        self.empty_streak = 0
        self._chord_latched = False

        self.sct = mss.MSS()
        self.mon = dict(
            self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
        )

        self.overlay = Overlay(self.mon)
        self.overlay.sub_dock = self.preset_key == "movie"
        self.roi_selector = ROISelector(self.mon)
        self.roi_selector.roi_selected.connect(self.on_roi_selected)
        self.hud = GlassHUD(self)
        self.worker = Worker(self.engine)
        self.worker.finished.connect(self.on_result)

        self.overlay.show()
        self.hud.adjustSize()
        self.hud.move(
            max(
                self.mon["left"] + 10,
                self.mon["left"] + self.mon["width"] - self.hud.width() - 24,
            ),
            self.mon["top"] + 32,
        )
        self.hud.show()
        self.hud.raise_()

        self.prev_thumb = None
        self.post_thumb = None
        self.dirty_rect = None
        self.stable_ticks = 0
        self.needs_scan = True
        self.capturing = False
        self.lock_until = 0.0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(PRESETS[self.preset_key]["tick_ms"])

    @property
    def preset(self):
        return PRESETS[self.preset_key]

    def grab_screen(self):
        shot = self.sct.grab(self.mon)
        return (
            np.frombuffer(shot.bgra, np.uint8)
            .reshape(shot.height, shot.width, 4)
            .copy()
        )

    def set_lang(self, code):
        self.lang = code
        self.last_blocks = []
        self.empty_streak = 0
        self.overlay.set_blocks([])
        self.force_scan()

    def set_bilingual(self, enabled):
        self.overlay.set_bilingual(enabled)
        self.lock_until = time.monotonic() + 0.18
        QtCore.QTimer.singleShot(120, self._capture_post_thumb)

    def cycle_preset(self):
        idx = (PRESET_ORDER.index(self.preset_key) + 1) % len(PRESET_ORDER)
        self.preset_key = PRESET_ORDER[idx]
        cfg = self.engine.set_preset(self.preset_key)
        self.overlay.sub_dock = self.preset_key == "movie"
        self.hud.mode_btn.setText(cfg["label"])
        self.hud.adjustSize()
        self.timer.setInterval(cfg["tick_ms"])
        self.force_scan()

    def toggle_area_roi(self, checked):
        if checked:
            self.last_blocks = []
            self.overlay.set_blocks([])
            self.roi_selector.select_start()
        else:
            self.custom_roi = None
            self.overlay.set_roi_rect(None)
            self.hud.area_btn.setText("🎯 Area")
            self.force_scan()

    def on_roi_selected(self, roi):
        if roi is None:
            self.custom_roi = None
            self.overlay.set_roi_rect(None)
            self.hud.area_btn.blockSignals(True)
            self.hud.area_btn.setChecked(False)
            self.hud.area_btn.setText("🎯 Area")
            self.hud.area_btn.blockSignals(False)
        else:
            self.custom_roi = roi
            self.overlay.set_roi_rect(roi)
            w, h = roi[2] - roi[0], roi[3] - roi[1]
            self.hud.area_btn.setText(f"🎯 {w}×{h}")
        self.hud.adjustSize()
        self.force_scan()

    def set_peek_toggle(self, checked):
        self.peek_toggled = checked
        self._update_modifiers()

    def _update_modifiers(self):
        mods = QtWidgets.QApplication.queryKeyboardModifiers()
        ctrl = bool(mods & QtCore.Qt.ControlModifier)
        shift = bool(mods & QtCore.Qt.ShiftModifier)
        alt = bool(mods & QtCore.Qt.AltModifier)

        if ctrl and shift and alt:
            if not self._chord_latched:
                self._chord_latched = True
                self.force_scan()
            return
        self._chord_latched = False

        want_peek = self.peek_toggled or (alt and not ctrl and not shift)
        if want_peek != self.overlay.peek_mode:
            self.overlay.set_peek(want_peek)
            self.lock_until = time.monotonic() + 0.18
            QtCore.QTimer.singleShot(120, self._capture_post_thumb)

    def copy_last_batch(self):
        if not self.last_blocks:
            self.hud.status.setText("📋 No lines to copy yet")
            return
        lines = [f"{b.src} -> {b.en}" for b in self.last_blocks]
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.hud.set_status_color("green")
        self.hud.status.setText(f"📋 Copied {len(lines)} line(s) to clipboard!")

    def set_paused(self, paused):
        self.paused = paused
        self.last_blocks = []
        self.overlay.set_blocks([])
        self.hud.status.setText("⏸ Paused" if paused else "⚡ Ready")
        if not paused:
            self.force_scan()

    def force_scan(self):
        self.post_thumb = None
        self.prev_thumb = None
        self.dirty_rect = None
        self.empty_streak = 0
        self.needs_scan = True
        self.stable_ticks = self.preset["stable_ticks"] + 1

    def _active_thumb_mask(self, shape):
        th_h, th_w = shape
        m = np.zeros(shape, dtype=bool)
        if self.custom_roi is not None:
            rx0, ry0, rx1, ry1 = self.custom_roi
            m[
                max(0, ry0 // THUMB) : min(th_h, ry1 // THUMB + 1),
                max(0, rx0 // THUMB) : min(th_w, rx1 // THUMB + 1),
            ] = True
        elif self.preset_key == "movie":
            # Restrict motion detection to the lower subtitle band (74%-95% height, 12%-88% width)
            # so upper-screen motion and logos do not trigger false rescans.
            y0, y1 = max(0, int(th_h * 0.74)), min(th_h, int(th_h * 0.95))
            x0, x1 = max(0, int(th_w * 0.12)), min(th_w, int(th_w * 0.88))
            m[y0:y1, x0:x1] = True
        else:
            m[:, :] = True

        # Exclude painted overlay regions so rendering translations does not trigger a rescan.
        pad = 24
        for rx0, ry0, rx1, ry1 in self.overlay.painted_rects:
            bx0 = max(0, (rx0 - pad) // THUMB)
            by0 = max(0, (ry0 - pad) // THUMB)
            bx1 = min(th_w, (rx1 + pad) // THUMB + 1)
            by1 = min(th_h, (ry1 + pad) // THUMB + 1)
            m[by0:by1, bx0:bx1] = False

        g = self.hud.frameGeometry()
        hx0 = max(0, (g.left() - self.mon["left"] - pad) // THUMB)
        hy0 = max(0, (g.top() - self.mon["top"] - pad) // THUMB)
        hx1 = min(th_w, (g.right() - self.mon["left"] + pad) // THUMB + 1)
        hy1 = min(th_h, (g.bottom() - self.mon["top"] + pad) // THUMB + 1)
        m[hy0:hy1, hx0:hx1] = False
        return m

    @staticmethod
    def _same_blocks(old_blocks, new_blocks):
        if len(old_blocks) != len(new_blocks):
            return False
        for a, b in zip(old_blocks, new_blocks):
            if a.src != b.src or a.en != b.en:
                return False
            if abs(a.x0 - b.x0) > 24 or abs(a.y0 - b.y0) > 24:
                return False
        return True

    def _update_hover_inspect(self):
        if not self.last_blocks:
            return
        pos = QtGui.QCursor.pos()
        cx, cy = pos.x() - self.mon["left"], pos.y() - self.mon["top"]
        found = -1
        for idx, b in enumerate(self.last_blocks):
            if (b.x0 - 8) <= cx <= (b.x1 + 8) and (b.y0 - 8) <= cy <= (b.y1 + 8):
                found = idx
                self.hud.setToolTip(f"{b.src} → {b.en}")
                break
        if found != self.overlay.hovered_idx:
            self.overlay.hovered_idx = found
            self.overlay.update()

    def tick(self):
        self._update_modifiers()
        self._update_hover_inspect()

        now = time.monotonic()
        if (
            self.paused
            or self.capturing
            or self.worker.busy
            or self.roi_selector.isVisible()
            or self.overlay.peek_mode
            or now < self.lock_until
        ):
            return

        raw = self.grab_screen()
        th = raw[::THUMB, ::THUMB, 1].astype(np.int16)
        active_mask = self._active_thumb_mask(th.shape)

        # In Movie mode, trigger a scan immediately when the subtitle band changes (>65 thumbnail pixels)
        # without waiting for background video motion to settle.
        if self.preset_key == "movie":
            if self.post_thumb is not None and self.post_thumb.shape == th.shape:
                diff_sub = int(
                    np.count_nonzero(
                        (np.abs(th - self.post_thumb) > DIFF_T) & active_mask
                    )
                )
                if diff_sub > 65:
                    self.needs_scan = True
            if self.needs_scan:
                self.needs_scan = False
                self.start_clean_capture()
            return

        # In Fast Read and HQ Manga modes, wait for scrolling or motion to settle before scanning.
        if (
            not self.needs_scan
            and self.post_thumb is not None
            and self.post_thumb.shape == th.shape
        ):
            diff_post = int(
                np.count_nonzero((np.abs(th - self.post_thumb) > DIFF_T) & active_mask)
            )
            if diff_post < 85:
                self.prev_thumb = th
                return
            self.needs_scan = True
            self.stable_ticks = 0

        if self.prev_thumb is not None and self.prev_thumb.shape == th.shape:
            diff_step = int(
                np.count_nonzero((np.abs(th - self.prev_thumb) > DIFF_T) & active_mask)
            )
            if diff_step > 45:
                self.stable_ticks = 0
                self.needs_scan = True
            else:
                self.stable_ticks += 1
        else:
            self.stable_ticks = self.preset["stable_ticks"]

        self.prev_thumb = th

        if self.needs_scan and self.stable_ticks >= self.preset["stable_ticks"]:
            self.needs_scan = False
            self.start_clean_capture()

    def start_clean_capture(self):
        self.capturing = True
        if self.overlay.blocks and not self.overlay.sub_dock:
            self.overlay.hidden_for_grab = True
            self.overlay.repaint()
            QtWidgets.QApplication.processEvents()
            QtCore.QTimer.singleShot(self.preset["hide_ms"], self.finish_clean_capture)
        else:
            self.finish_clean_capture()

    def finish_clean_capture(self):
        img = self.grab_screen()
        was_hidden = self.overlay.hidden_for_grab
        if was_hidden:
            self.overlay.hidden_for_grab = False
            self.overlay.repaint()
        elif self.overlay.sub_dock and self.overlay.painted_rects:
            # Mask only the docked translation box (leaving a 4px margin above the original subtitle)
            # so OCR does not re-read the rendered translation.
            H, W = img.shape[:2]
            for rx0, ry0, rx1, ry1 in self.overlay.painted_rects:
                y_cut = max(0, ry1 - 4)
                if y_cut > ry0:
                    img[max(0, ry0) : min(H, y_cut), max(0, rx0) : min(W, rx1)] = 24

        self.capturing = False

        g = self.hud.frameGeometry()
        panel_rect = (
            g.left() - self.mon["left"],
            g.top() - self.mon["top"],
            g.right() - self.mon["left"],
            g.bottom() - self.mon["top"],
        )
        # Pass dirty_rect=None so scan_image applies the active preset ROI policy.
        self.worker.process(
            img, self.lang, self.preset_key, self.custom_roi, None, panel_rect
        )

    def on_result(self, blocks, metrics):
        if not self.paused:
            if not blocks and self.last_blocks and self.preset_key == "movie":
                # Debounce subtitle clearing by requiring 2 consecutive empty scans.
                self.empty_streak += 1
                if self.empty_streak >= 2:
                    self.last_blocks = []
                    self.overlay.set_blocks([])
            else:
                self.empty_streak = 0
                if not self._same_blocks(self.last_blocks, blocks):
                    self.last_blocks = blocks
                    self.overlay.set_blocks(blocks)

            if "error" in metrics:
                self.hud.set_status_color("red")
                self.hud.status.setText(f"Error: {metrics['error']}")
            else:
                ocr_ms = metrics["ocr_ms"]
                tr_ms = metrics["trans_ms"]
                tot_ms = metrics["total_ms"]
                budget = metrics["budget_ms"]
                if tot_ms <= budget:
                    self.hud.set_status_color("green")
                elif tot_ms <= SLA_CAP_MS:
                    self.hud.set_status_color("amber")
                else:
                    self.hud.set_status_color("red")
                down_tag = " ↓25%" if metrics.get("downgraded") else ""
                self.hud.status.setText(
                    f"OCR: {ocr_ms:.0f}ms | Trans: {tr_ms:.0f}ms | Total: {tot_ms:.0f}ms{down_tag}"
                )
        self.lock_until = time.monotonic() + 0.25
        QtCore.QTimer.singleShot(160, self._capture_post_thumb)

    def _capture_post_thumb(self):
        if not self.capturing and not self.paused:
            raw = self.grab_screen()
            th = raw[::THUMB, ::THUMB, 1].astype(np.int16)
            self.post_thumb = th
            self.prev_thumb = th


def run_benchmark():
    """Run a headless benchmark on a 1920x1080 synthetic image with 12px and 28px Japanese text across all presets."""
    import subprocess
    from PIL import Image, ImageDraw, ImageFont

    font_path = subprocess.run(
        ["fc-match", "-f", "%{file}", "Noto Sans CJK JP:style=Regular"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    f12 = ImageFont.truetype(font_path, 12) if font_path else ImageFont.load_default()
    f28 = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()

    pil_img = Image.new("RGB", (1920, 1080), (242, 242, 246))
    draw = ImageDraw.Draw(pil_img)
    # Render 12px column labels and a 28px Japanese subtitle line in the lower region.
    draw.text((220, 690), "ロキシーの小さな文字テスト", font=f12, fill=(18, 18, 22))
    draw.text((540, 690), "設定メニューを開く", font=f12, fill=(18, 18, 22))
    draw.text((220, 760), "今日はとても良い天気ですね。", font=f28, fill=(15, 15, 20))
    img = np.asarray(pil_img)[:, :, ::-1].copy()

    engine = Engine()
    print("=== Headless Benchmark (1920x1080, 12px & 28px JA text) ===")
    for pkey in PRESET_ORDER:
        cfg = engine.set_preset(pkey)
        blocks, m = engine.scan_image(img, lang="ja", preset=pkey, use_cache=False)
        print(
            f"[{cfg['short']:9s}] boxes={len(blocks)} | "
            f"OCR: {m['ocr_ms']:.1f}ms | Trans: {m['trans_ms']:.1f}ms | Total: {m['total_ms']:.1f}ms "
            f"(budget <= {cfg['budget_ms']:.0f}ms, cap <= 500ms)"
        )
        for b in blocks:
            print(f"    - ({b.x0},{b.y0},{b.x1},{b.y1}) {b.src} -> {b.en}")


def main():
    parser = argparse.ArgumentParser(description="NeoLens Real-Time Screen Translator")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run headless latency benchmark and exit",
    )
    args, qt_args = parser.parse_known_args()
    if args.benchmark:
        run_benchmark()
        return

    engine = Engine()
    app = QtWidgets.QApplication([sys.argv[0]] + qt_args)
    app.setQuitOnLastWindowClosed(False)
    _ctl = Controller(engine)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    keepalive = QtCore.QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(200)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
