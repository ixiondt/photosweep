FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    ORT_DISABLE_TELEMETRY=1 \
    DATA_DIR=/data \
    PICS_ROOT=/photos

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The image classifier's model (stage 2), pinned and checksummed: about 100 MB.
# Its own layer, so a code change does not download it again.
COPY photosweep/fetch_model.py /tmp/fetch_model.py
RUN python /tmp/fetch_model.py /app/models/siglip-b16-int8.onnx && rm /tmp/fetch_model.py
COPY photosweep ./photosweep

# 33 is www-data, the user that owns a stock Nextcloud's data folder, so the
# photos are readable as they are. Change PUID/PGID in .env if yours differs.
RUN mkdir -p /data && chown 33:33 /data
USER 33:33
VOLUME /data
EXPOSE 8087

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8087/healthz', timeout=3)"

# uvicorn reads FORWARDED_ALLOW_IPS itself (default 127.0.0.1): set it to your
# reverse proxy's address so https is seen and the cookie is marked Secure.
CMD ["uvicorn", "photosweep.web:app", "--host", "0.0.0.0", "--port", "8087", "--workers", "1"]
