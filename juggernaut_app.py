from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

# Use "main" to get the latest development version of diffusers
diffusers_commit_sha = "main"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("uv")
    # The existing dependencies are sufficient for SDXL models
    .run_commands(
        f"uv pip install --system --compile-bytecode --index-strategy unsafe-best-match accelerate~=1.8.1 git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha} huggingface-hub[hf-transfer] Pillow~=11.2.1 safetensors~=0.5.3 transformers~=4.53.0 sentencepiece~=0.2.0 torch==2.7.1 optimum-quanto==0.2.7 fastapi[standard]==0.115.4 python-multipart==0.0.12 --extra-index-url https://download.pytorch.org/whl/cu128"
    )
)

# ✅ CHANGED: Model name updated to Juggernaut-XL v9
MODEL_NAME = "RunDiffusion/Juggernaut-XL-v9"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

# Juggernaut is a public model and does not require a token
secrets = []

image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": str(CACHE_DIR),
    }
)

app = modal.App("juggernaut-xl-fastapi")

with image.imports():
    import torch
    import os
    # ✅ CHANGED: Use the Stable Diffusion XL pipeline
    from diffusers import StableDiffusionXLPipeline
    from fastapi import FastAPI, Form, HTTPException
    from fastapi.responses import Response

@app.cls(
    image=image,
    cpu="0.5",
    memory="2GiB",
    gpu="L40s", # SDXL models are large, a powerful GPU is recommended
    volumes=volumes,
    secrets=secrets,
    scaledown_window=120,
    timeout=10 * 60,
)
class JuggernautXLModel:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")
        # ✅ CHANGED: Use float16 for better performance with SDXL
        dtype = torch.float16
        self.device = "cuda"

        # ✅ CHANGED: Instantiate the StableDiffusionXLPipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
            cache_dir=CACHE_DIR,
        ).to(self.device)
        print("✅ Model loaded successfully.")

    @modal.method()
    def inference(
        self,
        prompt: str,
        negative_prompt: str = "worst quality, low quality, bad anatomy, ugly, blurry",
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28,
        seed: Optional[int] = None,
    ) -> bytes:
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)

        print("🎨 Generating image...")
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        print("✅ Image generated.")

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()
        return image_bytes

@app.function(image=image, volumes=volumes, secrets=secrets, cpu="0.5", memory="2GiB")
@modal.asgi_app()
def fastapi_app():
    web_app = FastAPI(
        title="Juggernaut-XL Image Generator",
        description="Generate images using the Juggernaut-XL v9 model",
        version="1.0.0",
    )

    @web_app.post("/generate_image")
    async def generate_image(
        prompt: str = Form(..., description="Text prompt for image generation"),
        negative_prompt: str = Form("worst quality, low quality, bad anatomy, ugly, blurry", description="Negative prompt to avoid certain features"),
        width: int = Form(1024, description="Width of the generated image"),
        height: int = Form(1024, description="Height of the generated image"),
        guidance_scale: float = Form(7.0, description="Guidance scale (higher = more prompt adherence)"),
        num_inference_steps: int = Form(28, description="Number of inference steps"),
        seed: int = Form(None, description="Random seed for reproducible results"),
    ):
        try:
            model = JuggernautXLModel()
            result_bytes = model.inference.remote(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
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
            print(f"Error during image generation: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error generating image: {str(e)}"
            )

    return web_app

@app.local_entrypoint()
def main(
    prompt: str = "photo of a beautiful golden retriever puppy, cinematic, dramatic lighting, 8k, photorealistic",
    output_path: str = "/tmp/juggernaut_image.png",
):
    print(f"🎨 Generating image with prompt: {prompt}")
    model = JuggernautXLModel()
    output_image_bytes = model.inference.remote(prompt=prompt)
    output_file = Path(output_path)
    print(f"💾 Saving output image to {output_file}")
    output_file.write_bytes(output_image_bytes)
    print("✅ Image generation completed successfully!")
