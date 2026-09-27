FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /workspace/MiniMax-H3
COPY model ./model
COPY infer_native.py ./infer.py
COPY LICENSE ./LICENSE
RUN mkdir -p weights outputs inputs \
    && for part in transformer transformer_ref vae audio_vae scheduler audio_scheduler; do \
        ln -s /weights/h3-diffusers/$part weights/$part; done \
    && for part in tokenizer processor text_encoder; do \
        ln -s /weights/h3-fl2va/$part weights/$part; done \
    && python -c 'import importlib.util; assert importlib.util.find_spec("diffusers") is None'
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "infer.py"]
