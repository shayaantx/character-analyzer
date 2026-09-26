# CUDA 12 + cuDNN 9 runtime, matching onnxruntime-gpu >= 1.19
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/opt/insightface

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# insightface compiles a C extension; drop the compiler afterwards to keep the image small.
# Some dependencies pull in the GUI build of OpenCV (needs X11 libs) which clobbers the
# headless one, so remove every variant and reinstall headless only. Same for the CPU-only
# onnxruntime, which shadows onnxruntime-gpu (both install the same module).
RUN pip3 install -r requirements.txt \
 && pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless \
 && pip3 install "opencv-python-headless>=4.8" \
 && pip3 uninstall -y onnxruntime onnxruntime-gpu \
 && pip3 install "onnxruntime-gpu>=1.19" \
 && python3 -c "import onnxruntime as o; p = o.get_available_providers(); print(p); assert 'CUDAExecutionProvider' in p" \
 && apt-get purge -y --auto-remove build-essential python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Bake the face model into the image so containers don't download it every run
RUN python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='$INSIGHTFACE_HOME', providers=['CPUExecutionProvider'])" \
 && chmod -R a+rX "$INSIGHTFACE_HOME"

# Readable by the non-root user the container runs as, whatever the host file mode is
COPY --chmod=644 analyze.py .

# Run from /data so the defaults (./videos, ./refs, ./output) point at the mounted folders
WORKDIR /data
ENTRYPOINT ["python3", "/app/analyze.py"]
