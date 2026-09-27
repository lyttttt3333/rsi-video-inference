FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /workspace/SANA-Video-2.0
COPY model ./model
COPY infer_native.py ./infer.py
COPY LICENSE ./LICENSE
RUN mkdir -p weights outputs inputs \
    && ln -s /weights/gemma/snapshots/current weights/gemma \
    && ln -s /weights/ltx-vae/vae weights/vae \
    && ln -s /weights/sana-checkpoint/checkpoints/SANA_Video_2.0_5B_720p.pth weights/SANA_Video_2.0_5B_720p.pth \
    && python -c 'import importlib.util; assert importlib.util.find_spec("diffusers") is None'
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
ENTRYPOINT ["python", "infer.py"]
