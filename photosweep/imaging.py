"""Image reading: open any photo the library holds (HEIC included), read its
camera EXIF, and make the small JPEG the model and the grid both use.

Cheap signals live here too. A camera make/model in EXIF almost rules out a
screenshot or a saved meme; a `Screenshot_` filename almost rules one in. They
are shown beside the model's verdict, never used to override it.
"""
import io
import os
import re

from PIL import Image, ImageFile, ImageFilter, ImageOps, ImageStat

try:  # HEIC/HEIF from iPhones; the scanner degrades to "error" on those without it
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # pragma: no cover - present in the deployed venv
    pass

# Panoramas and scans are legitimately large; a decompression bomb is not.
Image.MAX_IMAGE_PIXELS = 250_000_000

# Decode a cut-off or slightly damaged JPEG as far as it goes, the way IrfanView
# and the phone gallery do, rather than refusing it: the missing part comes out
# grey. Nine real photos sat in Errors as "truncated" or "broken data stream"
# (2026-11-18). A file with no image header at all still fails to open.
ImageFile.LOAD_TRUNCATED_IMAGES = True

_HINTS = [
    ("screenshot", re.compile(r"screen\s*shot|screenshot|screen_?recording|^scr_\d", re.I)),
    ("facebook", re.compile(r"^FB_IMG_", re.I)),
    ("messaging", re.compile(r"whatsapp|-WA\d{4}|^signal-|telegram|messenger", re.I)),
    ("app-download", re.compile(r"^ReactNativeBlobUtilTmp|^received_|^download", re.I)),
    ("ios-saved", re.compile(r"^image\d*\.", re.I)),
]

# EXIF tags (baseline TIFF ids, so no lookup table is needed)
_MAKE, _MODEL, _SOFTWARE = 0x010F, 0x0110, 0x0131


def name_hint(filename):
    """A short label for what the filename suggests, or None."""
    base = os.path.basename(filename)
    for label, rx in _HINTS:
        if rx.search(base):
            return label
    return None


def camera_of(exif):
    """'Make Model' from an Image.Exif, or None when no camera wrote this file.
    Model often repeats the make ('Apple' / 'iPhone 12'); join without doubling."""
    if not exif:
        return None
    make = str(exif.get(_MAKE, "") or "").strip().strip("\x00")
    model = str(exif.get(_MODEL, "") or "").strip().strip("\x00")
    if not make and not model:
        return None
    if make and model.lower().startswith(make.lower()):
        return model[:80]
    return f"{make} {model}".strip()[:80]


def open_image(path, target_px):
    """Open, honour EXIF orientation, and convert to RGB. For JPEGs, draft()
    lets the decoder skip straight to a reduced scale, which is most of the cost
    on 12-50 MP originals. Returns (image, camera, original_width, original_height)."""
    img = Image.open(path)
    width, height = img.size
    try:
        exif = img.getexif()
    except Exception:  # corrupt EXIF must not cost the whole image
        exif = None
    camera = camera_of(exif)
    if img.format == "JPEG":
        img.draft("RGB", (target_px * 2, target_px * 2))
    try:
        img.seek(0)  # first frame of a GIF or a multi-image HEIC
    except EOFError:
        pass
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    return img, camera, width, height


_LAPLACIAN = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


def sharpness(img):
    """How sharp the sharpest part of the picture is, contrast-normalised.
    Low means shaken, out of focus or smeared; healthy photos sit around 70.

    The Laplacian variance of the best of a 4x4 grid of tiles, so a sharp
    subject on a soft background still scores as sharp, divided by the whole
    frame's grey variance, so a dim but in-focus shot is not mistaken for a
    blurred one. Global Laplacian variance alone ranked plain sky and a jet in
    an overcast sky as the blurriest photos in the library; this did not.
    Calibrated 2026-11-17 on 3,708 labelled thumbnails: the lowest ~2% (under
    ~15) were nearly all motion-blurred or out-of-focus, and by ~20 they were
    soft old point-and-shoot frames worth keeping.

    Always computed from the stored 512 px thumbnail, so the scanner and the
    backfill give byte-identical scores for the same file."""
    g = img.convert("L")
    g.thumbnail((256, 256), Image.Resampling.LANCZOS)
    w, h = g.size
    if w < 8 or h < 8:
        return None
    lap = g.filter(_LAPLACIAN)
    best = 0.0
    for ty in range(4):
        for tx in range(4):
            # one pixel in from each tile edge: the kernel's own border is not detail
            box = (tx * w // 4 + 1, ty * h // 4 + 1, (tx + 1) * w // 4 - 1, (ty + 1) * h // 4 - 1)
            if box[2] > box[0] and box[3] > box[1]:
                best = max(best, ImageStat.Stat(lap.crop(box)).var[0])
    sd = ImageStat.Stat(g).stddev[0]
    return round(best / (sd * sd + 1) * 100, 2)


def dhash(img, n=16):
    """A difference hash of what the picture shows, as n*n bits in hex: the
    image shrunk to (n+1) x n grey, one bit per pixel for "brighter than its
    right-hand neighbour". Re-encoding, resizing, a format change and lost EXIF
    barely move it, so two files of one picture land a few bits apart while
    their bytes, names and sizes all differ.

    Measured 2026-11-21 on 31,980 library thumbnails at n=16 (256 bits): copies
    of one picture sit at 0-12 bits, burst frames and near-repeats at ~15-40,
    and unrelated photos become common only from ~55. Computed from the stored
    512 px thumbnail, like sharpness(), so the scanner and the backfill agree."""
    g = img.convert("L").resize((n + 1, n), Image.Resampling.LANCZOS)
    px = g.get_flattened_data() if hasattr(g, "get_flattened_data") else list(g.getdata())
    bits = 0
    for y in range(n):
        row = y * (n + 1)
        for x in range(n):
            bits = (bits << 1) | (px[row + x] > px[row + x + 1])
    return f"{bits:0{n * n // 4}x}"


def dhash_flipped(img):
    """dhash() of the picture flipped left to right, so a mirrored copy (a
    selfie and its flipped forward) lands as close to it as a plain copy
    would. Taken from the pixels rather than worked out from dhash(): that
    shortcut breaks wherever neighbours are exactly equal (a blown sky, a
    white wall, a black screen), measured 2026-11-23 at 66 bits on a
    synthetic pair where the real flip is a few.

    On the library a WhatsApp selfie and its flipped copy sit 126 bits apart
    as they are and 14 apart with one flipped; about 440 look-alike pairs fall
    within 24 that way, and no random pair of 200,000 did within 20."""
    return dhash(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))


def jpeg_bytes(img, max_px, quality=82):
    """A copy scaled to fit max_px on its long side, as JPEG bytes."""
    copy = img.copy()
    copy.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    copy.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
