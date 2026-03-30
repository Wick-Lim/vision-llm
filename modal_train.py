"""Modal GPU training script for vector path diffusion model."""

import modal

app = modal.App("vision-llm-train")

# Image with dependencies + local font for consistency
FONT_PATH = "/root/fonts/AppleSDGothicNeo.ttc"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "fonttools", "Pillow", "numpy<2")
    .env({"VISION_LLM_FONT": FONT_PATH})
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("fonts", remote_path="/root/fonts")
)

# Volume for checkpoints persistence
volume = modal.Volume.from_name("vision-llm-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/root/checkpoints": volume},
)
def train_gpu(
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 3e-4,
    max_len: int = 128,
    max_chars: int = 500,
):
    import sys
    sys.path.insert(0, "/root/project")

    from src.train import train

    train(
        font_path=FONT_PATH,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_len=max_len,
        max_chars=max_chars,
        device="cuda",
        save_dir="/root/checkpoints",
        model_dim=256,
    )

    volume.commit()


@app.local_entrypoint()
def main():
    train_gpu.remote(
        epochs=5000,
        batch_size=64,
        lr=3e-4,
        max_len=128,
        max_chars=50,
    )

    # Download checkpoints locally
    import os
    os.makedirs("checkpoints", exist_ok=True)
    for entry in volume.listdir("/"):
        if entry.path.endswith(".pt"):
            print(f"Downloading {entry.path}...")
            with open(f"checkpoints/{entry.path}", "wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
    print("Done! Checkpoints saved to ./checkpoints/")
