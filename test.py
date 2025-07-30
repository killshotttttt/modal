from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

# Define the container image with all necessary dependencies
# Updated diffusers commit SHA to ensure compatibility with FLUX.1-dev
diffusers_commit_sha = "main"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --compile-bytecode --index-strategy unsafe-best-match accelerate~=1.8.1 git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha} huggingface-hub[hf-transfer]~=0.33.1 Pillow~=11.2.1 safetensors~=0.5.3 transformers~=4.53.0 sentencepiece~=0.2.0 torch==2.7.1 optimum-quanto==0.2.7 fastapi[standard]==0.115.4 python-multipart==0.0.12 --extra-index-url https://download.pytorch.org/whl/cu128"
    )
)

# 🚀 Switched to the text-to-image FLUX.1-dev model
MODEL_NAME = "black-forest-labs/FLUX.1-dev"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("flux-app-secrets", required_keys=["HF_TOKEN"])]

image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)

app = modal.App("flux-dev-fastapi")

with image.imports():
    import torch
    import os
    # 🚀 Use FluxDevPipeline for the text-to-image model
    from diffusers import FluxDevPipeline
    from PIL import Image
    from fastapi import FastAPI, Form, HTTPException
    from fastapi.responses import Response


@app.cls(
    image=image,
    cpu="0.5",
    memory="2GiB",
    gpu="L40s",
    volumes=volumes,
    secrets=secrets,
    scaledown_window=120,
    timeout=10 * 60,  # 10 minutes
)
class FluxModel:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")

        dtype = torch.bfloat16
        self.device = "cuda"

        # 🚀 Use FluxDevPipeline to load the model
        self.pipe = FluxDevPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        ).to(self.device)

    @modal.method()
    def inference(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
    ) -> bytes:
        # Use provided seed or generate a random one
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)

        # 🚀 The pipe call no longer needs an initial image
        image = self.pipe(
            width=width,
            height=height,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="pil",
            generator=generator,
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


# Create a separate FastAPI app to handle the web interface
@app.function(image=image, volumes=volumes, secrets=secrets, cpu="0.5", memory="2GiB")
@modal.asgi_app()
def fastapi_app():
    from fastapi import Depends, FastAPI, Form, HTTPException, status
    from fastapi.responses import Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    web_app = FastAPI(
        title="Flux Dev Image Generator",
        description="Generate images using the FLUX.1-dev model",
        version="1.0.0",
    )

    # 🚀 Endpoint changed to /generate_image
    @web_app.post("/generate_image")
    async def generate_image(
        prompt: str = Form(..., description="Text prompt for image generation"),
        token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        width: int = Form(1024, description="Width of the generated image"),
        height: int = Form(1024, description="Height of the generated image"),
        guidance_scale: float = Form(
            3.5, description="Guidance scale (higher = more prompt adherence)"
        ),
        num_inference_steps: int = Form(20, description="Number of inference steps"),
        seed: int = Form(None, description="Random seed for reproducible results"),
    ):
        """
        Generate an image using the FLUX.1-dev model.

        - **prompt**: Text description of the desired image.
        - **width**: Image width (default: 1024).
        - **height**: Image height (default: 1024).
        - **guidance_scale**: Controls prompt adherence (default: 3.5).
        - **num_inference_steps**: Number of denoising steps (default: 20).
        - **seed**: Optional seed for reproducible results.
        """

        if os.environ.get("BEARER_TOKEN", False):
            if not token or token.credentials != os.environ["BEARER_TOKEN"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        try:
            # 🚀 No longer need image validation, as we are generating from scratch
            model = FluxModel()
            result_bytes = model.inference.remote(
                width=width,
                height=height,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )

            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={"Content-Disposition": "inline; filename=generated_image.png"},
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating image: {str(e)}"
            )

    return web_app


# ## CLI Interface
# Run with: modal run your_script_name.py --prompt "your prompt here"
@app.local_entrypoint()
def main(
    prompt: str = "A cinematic shot of a Corgi wearing a wizard hat, Studio Ghibli style.",
    output_path: str = "/tmp/generated_image.png",
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 20,
    seed: Optional[int] = None,
):
    """
    Test the Flux model locally via CLI for text-to-image generation.
    """
    print(f"🎨 Generating image with prompt: {prompt}")

    # 🚀 No longer need to read an input image
    model = FluxModel()
    output_image_bytes = model.inference.remote(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    output_file = Path(output_path)
    print(f"🎨 Saving output image to {output_file}")
    output_file.write_bytes(output_image_bytes)

    print("✅ Image generation completed successfully!")
