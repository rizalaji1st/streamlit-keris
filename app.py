"""Aplikasi Streamlit untuk klasifikasi & segmentasi dhapur keris berbasis YOLO11-seg."""

from __future__ import annotations

import colorsys
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best.pt"
MAX_SIDE = 1600  # foto sangat besar dikecilkan dulu supaya inferensi tetap ringan

st.set_page_config(
    page_title="Klasifikasi & Segmentasi Keris",
    page_icon="🗡️",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat model YOLO...")
def load_model(weights_path: str, _mtime: float):
    """Muat model YOLO. `_mtime` membuat cache batal otomatis saat bobot diganti."""
    from ultralytics import YOLO

    return YOLO(weights_path)


# --------------------------------------------------------------------------- #
# Utilitas gambar
# --------------------------------------------------------------------------- #
def color_for(class_id: int) -> tuple[int, int, int]:
    """Warna stabil per kelas (golden-angle hue supaya antar kelas kontras)."""
    hue = (class_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()


def prepare_image(file) -> Image.Image:
    image = Image.open(file).convert("RGB")
    if max(image.size) > MAX_SIDE:
        scale = MAX_SIDE / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Samakan ukuran mask dengan gambar asli (size = (width, height))."""
    if (mask.shape[1], mask.shape[0]) == size:
        return mask
    resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def overlay_masks(
    image: Image.Image,
    detections: list[dict],
    alpha: float,
    show_boxes: bool,
    show_labels: bool,
) -> Image.Image:
    """Tumpuk mask semi-transparan, bounding box, dan label di atas foto asli."""
    canvas = np.asarray(image, dtype=np.float32).copy()

    for det in detections:
        mask = det.get("mask")
        if mask is None:
            continue
        color = np.array(color_for(det["class_id"]), dtype=np.float32)
        selected = mask > 0.5
        canvas[selected] = canvas[selected] * (1.0 - alpha) + color * alpha

    result = Image.fromarray(canvas.astype(np.uint8))
    if not (show_boxes or show_labels):
        return result

    draw = ImageDraw.Draw(result)
    font = load_font(max(14, round(min(result.size) * 0.028)))
    line_width = max(2, round(min(result.size) * 0.004))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = color_for(det["class_id"])
        if show_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if show_labels:
            text = f"{det['class_name']} {det['confidence']:.0%}"
            tl, tt, tr, tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tr - tl, tb - tt
            pad = max(2, line_width)
            ty = max(0.0, y1 - th - 2 * pad)
            draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color)
            draw.text((x1 + pad - tl, ty + pad - tt), text, fill=(20, 20, 20), font=font)

    return result


def cutout(image: Image.Image, detections: list[dict], feather: bool = True) -> Image.Image:
    """Potong objek dari latar: gabungan semua mask jadi PNG transparan."""
    combined = np.zeros((image.height, image.width), dtype=np.float32)
    for det in detections:
        if det.get("mask") is not None:
            combined = np.maximum(combined, det["mask"])
    if not feather:
        combined = (combined > 0.5).astype(np.float32)

    rgba = np.dstack([np.asarray(image, dtype=np.uint8), (combined * 255).astype(np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Inferensi
# --------------------------------------------------------------------------- #
def run_inference(model, image: Image.Image, conf: float, iou: float, imgsz: int) -> list[dict]:
    """Jalankan prediksi lalu ubah hasilnya jadi list dict yang mudah dipakai UI."""
    result = model.predict(
        source=np.asarray(image),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )[0]

    names = result.names
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    total_pixels = float(image.width * image.height)
    detections: list[dict] = []

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())
        mask = None
        area_ratio = None
        if masks is not None and i < len(masks):
            mask = resize_mask(masks[i].astype(np.float32), (image.width, image.height))
            area_ratio = float((mask > 0.5).sum() / total_pixels)

        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
        detections.append(
            {
                "index": i + 1,
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": float(boxes.conf[i].item()),
                "box": [x1, y1, x2, y2],
                "mask": mask,
                "area_ratio": area_ratio,
            }
        )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    for order, det in enumerate(detections, start=1):
        det["index"] = order
    return detections


def summarize_classes(detections: list[dict]) -> list[dict]:
    """Ringkas per kelas: jumlah objek, confidence tertinggi, total luas mask."""
    summary: dict[str, dict] = {}
    for det in detections:
        entry = summary.setdefault(
            det["class_name"],
            {"class_name": det["class_name"], "jumlah": 0, "confidence_max": 0.0, "luas": 0.0},
        )
        entry["jumlah"] += 1
        entry["confidence_max"] = max(entry["confidence_max"], det["confidence"])
        entry["luas"] += det["area_ratio"] or 0.0
    return sorted(summary.values(), key=lambda e: e["confidence_max"], reverse=True)


def detections_to_json(detections: list[dict], image: Image.Image, source_name: str) -> str:
    payload = {
        "file": source_name,
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "ukuran_gambar": {"width": image.width, "height": image.height},
        "jumlah_deteksi": len(detections),
        "deteksi": [
            {
                "index": d["index"],
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "confidence": round(d["confidence"], 4),
                "box_xyxy": [round(v, 2) for v in d["box"]],
                "area_ratio": round(d["area_ratio"], 5) if d["area_ratio"] is not None else None,
            }
            for d in detections
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render_result(name: str, image: Image.Image, detections: list[dict], settings: dict) -> None:
    if not detections:
        st.warning(
            "Tidak ada objek terdeteksi. Coba turunkan **Confidence threshold** di sidebar "
            "atau gunakan foto keris yang lebih jelas dan tidak terlalu jauh."
        )
        st.image(image, caption=name, width="stretch")
        return

    top = detections[0]
    summary = summarize_classes(detections)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dhapur terprediksi", top["class_name"])
    col2.metric("Confidence", f"{top['confidence']:.1%}")
    col3.metric("Objek terdeteksi", len(detections))

    tab_seg, tab_cls, tab_cut, tab_data = st.tabs(
        ["🎭 Segmentasi", "🏷️ Klasifikasi", "✂️ Objek terpotong", "📄 Data"]
    )

    with tab_seg:
        annotated = overlay_masks(
            image,
            detections,
            alpha=settings["alpha"],
            show_boxes=settings["show_boxes"],
            show_labels=settings["show_labels"],
        )
        left, right = st.columns(2)
        left.image(image, caption="Foto asli", width="stretch")
        right.image(annotated, caption="Hasil segmentasi", width="stretch")
        st.download_button(
            "⬇️ Unduh gambar hasil segmentasi",
            data=to_png_bytes(annotated),
            file_name=f"{Path(name).stem}_segmentasi.png",
            mime="image/png",
            key=f"dl-seg-{name}",
        )

    with tab_cls:
        st.caption("Confidence tertinggi per kelas pada foto ini.")
        for row in summary:
            st.markdown(
                f"**{row['class_name']}** — {row['confidence_max']:.1%} "
                f"· {row['jumlah']} objek · luas mask {row['luas']:.1%} dari gambar"
            )
            st.progress(min(1.0, row["confidence_max"]))

    with tab_cut:
        st.caption("Piksel di luar mask dibuat transparan (PNG).")
        cut = cutout(image, detections, feather=settings["feather"])
        st.image(cut, caption="Keris tanpa latar", width="stretch")
        st.download_button(
            "⬇️ Unduh PNG transparan",
            data=to_png_bytes(cut),
            file_name=f"{Path(name).stem}_objek.png",
            mime="image/png",
            key=f"dl-cut-{name}",
        )

    with tab_data:
        st.dataframe(
            [
                {
                    "#": d["index"],
                    "Kelas": d["class_name"],
                    "Confidence": f"{d['confidence']:.2%}",
                    "Luas mask": f"{d['area_ratio']:.2%}" if d["area_ratio"] is not None else "-",
                    "Box (x1, y1, x2, y2)": ", ".join(f"{v:.0f}" for v in d["box"]),
                }
                for d in detections
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "⬇️ Unduh hasil (JSON)",
            data=detections_to_json(detections, image, name),
            file_name=f"{Path(name).stem}_hasil.json",
            mime="application/json",
            key=f"dl-json-{name}",
        )


def main() -> None:
    st.title("🗡️ Klasifikasi & Segmentasi Keris")
    st.caption(
        "Unggah foto keris, model YOLO11-seg akan mengenali dhapur-nya "
        "sekaligus memetakan bentuk bilahnya."
    )

    with st.sidebar:
        st.header("⚙️ Pengaturan")
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU threshold (NMS)", 0.10, 0.95, 0.45, 0.05)
        imgsz = st.select_slider("Ukuran inferensi", [320, 416, 512, 640, 768, 960, 1280], value=640)

        st.divider()
        st.subheader("Tampilan")
        alpha = st.slider("Ketebalan warna mask", 0.0, 1.0, 0.45, 0.05)
        show_boxes = st.checkbox("Tampilkan bounding box", value=True)
        show_labels = st.checkbox("Tampilkan label", value=True)
        feather = st.checkbox("Tepi mask halus (cut-out)", value=True)

    settings = {
        "alpha": alpha,
        "show_boxes": show_boxes,
        "show_labels": show_labels,
        "feather": feather,
    }

    if not MODEL_PATH.exists():
        st.error(f"Model tidak ditemukan: `{MODEL_PATH.name}`. Letakkan file di folder aplikasi.")
        st.stop()

    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except ImportError:
        st.error("Paket `ultralytics` belum terpasang. Jalankan: `pip install -r requirements.txt`")
        st.stop()
    except Exception as exc:  # bobot rusak atau tidak kompatibel
        st.error(f"Gagal memuat model: {exc}")
        st.stop()

    class_names = [str(v) for v in getattr(model, "names", {}).values()]
    with st.sidebar:
        st.divider()
        st.caption(f"**Model:** `{MODEL_PATH.name}`")
        st.caption(f"**{len(class_names)} kelas:** " + ", ".join(class_names))

    files = st.file_uploader(
        "Unggah foto keris",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        help="Bisa lebih dari satu foto sekaligus.",
    )

    if not files:
        st.info("⬆️ Unggah minimal satu foto untuk memulai analisis.")
        return

    for tab, file in zip(st.tabs([f.name for f in files]), files):
        with tab:
            try:
                image = prepare_image(file)
            except Exception as exc:
                st.error(f"Gagal membaca `{file.name}`: {exc}")
                continue

            with st.spinner(f"Menganalisis {file.name}..."):
                detections = run_inference(model, image, conf, iou, imgsz)
            render_result(file.name, image, detections, settings)


if __name__ == "__main__":
    main()
