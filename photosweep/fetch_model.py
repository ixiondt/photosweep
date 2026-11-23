"""Download the image classifier's model, pinned to one revision and checked.

    python -m photosweep.fetch_model /app/models/siglip-b16-int8.onnx

SigLIP B/16 (google/siglip-base-patch16-224, Apache-2.0), the int8 ONNX
export of its image encoder by Xenova. Fetched at one commit and refused
unless its size and SHA-256 match, so a changed or tampered file never loads.
Standard library only: the Dockerfile runs it before anything is installed.
"""
import hashlib
import os
import sys
import urllib.request

REVISION = "4649052661e53c7000355844105f8a1792088239"
URL = ("https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/"
       f"{REVISION}/onnx/vision_model_quantized.onnx")
SIZE = 99_499_129
SHA256 = "ef14a954f3d57e1806666432bd9785004c1dc27100aa260eee0cb0f10a5de058"


def verify(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return os.path.getsize(path) == SIZE and h.hexdigest() == SHA256


def fetch(dest):
    if os.path.isfile(dest) and verify(dest):
        print(f"{dest} already present and verified")
        return
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(URL, headers={"User-Agent": "photosweep-fetch-model"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if not verify(tmp):
        os.remove(tmp)
        raise SystemExit(f"downloaded model does not match: expected {SIZE} bytes, sha256 {SHA256}")
    os.replace(tmp, dest)
    print(f"{dest} fetched and verified")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    fetch(sys.argv[1])
