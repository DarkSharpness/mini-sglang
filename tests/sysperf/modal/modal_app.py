import modal
import os
from pathlib import Path

# Load HF token from .env
hf_token = None
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

# Build image
image_env = {"PATH": "/root/.cargo/bin:$PATH", "CUDA_VISIBLE_DEVICES": "0"}
if hf_token:
    image_env["HF_TOKEN"] = hf_token

# Use CUDA base image - Modal GPU images have CUDA pre-installed
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .run_commands(
        "export PATH=/root/.local/bin:$PATH && "
        "uv pip install --system 'torch>=2.0.0' 'transformers<=4.57.3' accelerate msgpack "
        "'sgl_kernel>=0.3.17.post1' 'flashinfer-python>=0.5.3' pyzmq "
        "'apache-tvm-ffi>=0.1.4' 'nvidia-cutlass-dsl==4.3.1'"
    )
    .env({
        **image_env,
        "PATH": "/usr/local/cuda/bin:/root/.cargo/bin:/root/.local/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "CUDA_HOME": "/usr/local/cuda"
    })
)

image = cuda_image

app = modal.App("minisgl-benchmarks", image=image)

@app.function(image=image, gpu="A10G", timeout=3600)
def run_offline_benchmark():
    """Run the offline benchmark from benchmark/offline/bench.py"""
    import subprocess
    import sys
    from pathlib import Path
    
    # Clone repo
    repo = Path("/root/minisglang")
    if not repo.exists():
        subprocess.run(["git", "clone", "https://github.com/lamng3/mini-sglang.git", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "kernel_fused_copy"], check=False)
    
    # Set up Python path
    python_path = str(repo / "python")
    sys.path.insert(0, python_path)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": python_path}
    
    # Run benchmark (output goes to stdout)
    bench_script = repo / "benchmark" / "offline" / "bench.py"
    result = subprocess.run(
        [sys.executable, "-u", str(bench_script)],
        cwd=str(repo),
        env=env
    )
    return result.returncode


@app.local_entrypoint()
def main():
    """Run offline benchmark on Modal GPU."""
    run_offline_benchmark.remote()
