import base64
import io
import re

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

MP_ENTRIES_TAG = 45058


def extract_gain_map(data: bytes) -> bytes:
    """Return the gain map JPEG embedded in an Ultra HDR file (second MPF entry)."""
    im = Image.open(io.BytesIO(data))
    mp = im._getmp()  # private API, same as the notebook
    if not mp or MP_ENTRIES_TAG not in mp:
        raise ValueError("No Multi-Picture Format directory found; not an Ultra HDR JPEG.")
    entries = mp[MP_ENTRIES_TAG]
    if len(entries) < 2:
        raise ValueError("File has no secondary image; no gain map found.")
    ent = entries[1]
    start = im.info["mpoffset"] + ent["DataOffset"]
    gm = data[start:start + ent["Size"]]
    if gm[:2] != b"\xff\xd8" or gm[-2:] != b"\xff\xd9":
        raise ValueError("Secondary image is not a valid JPEG.")
    return gm


def read_hdrgm_metadata(gm_bytes: bytes) -> dict:
    """Parse hdrgm:* XMP values from the gain map JPEG (replaces the exiftool call)."""
    xmp = gm_bytes.decode("latin-1")
    meta = {}
    for key in ("Version", "GainMapMin", "GainMapMax", "Gamma",
                "OffsetSDR", "OffsetHDR", "HDRCapacityMin", "HDRCapacityMax"):
        # Either an attribute (hdrgm:Key="1.0") or an element, possibly with an rdf:Seq of per-channel values.
        m = re.search(rf'hdrgm:{key}="([^"]*)"', xmp)
        if m:
            vals = [m.group(1)]
        else:
            m = re.search(rf"<hdrgm:{key}>(.*?)</hdrgm:{key}>", xmp, re.S)
            if not m:
                continue
            vals = re.findall(r"<rdf:li>([^<]*)</rdf:li>", m.group(1)) or [m.group(1)]
        try:
            nums = [float(v) for v in vals]
        except ValueError:
            continue
        meta[key] = nums if len(nums) > 1 else nums[0]
    return meta


def log2_range(meta: dict):
    """Declared log2 gain range (stops) for pixel values 0..255; per-channel values collapse to the extremes."""
    lo = np.min(meta.get("GainMapMin", 0.0))
    hi = np.max(meta["GainMapMax"])
    return float(lo), float(hi)


def decode_log2_gain(gm_bytes: bytes, meta: dict) -> np.ndarray:
    """Per-pixel log2 gain (stops) from the gain map JPEG."""
    gain = np.asarray(Image.open(io.BytesIO(gm_bytes)))
    gmin = np.asarray(meta.get("GainMapMin", 0.0), dtype=np.float32)
    gmax = np.asarray(meta["GainMapMax"], dtype=np.float32)
    gamma = np.asarray(meta.get("Gamma", 1.0), dtype=np.float32)
    g = gain.astype(np.float32) / 255.0
    return gmin + (gmax - gmin) * g ** (1.0 / gamma)  # broadcasts if params are per-channel


def render_scaled(log2_gain: np.ndarray, lo: float, hi: float) -> str:
    """PNG data URL with the image's own min..max stretched to 0..255; a constant map is neutral gray."""
    if hi > lo:
        arr = ((log2_gain - lo) / (hi - lo) * 255).round().astype(np.uint8)
    else:
        arr = np.full(log2_gain.shape, 127, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No file uploaded."), 400
    data = f.read()
    try:
        gm_bytes = extract_gain_map(data)
        meta = read_hdrgm_metadata(gm_bytes)
        if "GainMapMax" not in meta:
            raise ValueError("Gain map is missing hdrgm:GainMapMax metadata.")
        lo, hi = log2_range(meta)
        log2_gain = decode_log2_gain(gm_bytes, meta)
        actual_lo, actual_hi = float(log2_gain.min()), float(log2_gain.max())
        width, height = Image.open(io.BytesIO(gm_bytes)).size
    except Exception as e:
        return jsonify(error=str(e)), 422
    return jsonify(
        gain_map="data:image/jpeg;base64," + base64.b64encode(gm_bytes).decode(),
        log2_min=lo,
        log2_max=hi,
        gain_map_scaled=render_scaled(log2_gain, actual_lo, actual_hi),
        actual_min=actual_lo,
        actual_max=actual_hi,
        gain_map_size=[width, height],
        meta=meta,
    )


def main():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
