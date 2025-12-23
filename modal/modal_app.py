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

# Profile script will be read from cloned repo at runtime

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
    from datetime import datetime
    from pathlib import Path
    
    # Clone repo
    repo = Path("/root/minisglang")
    if not repo.exists():
        subprocess.run(["git", "clone", "https://github.com/sgl-project/mini-sglang.git", str(repo)], check=True)
    
    # Set up Python path
    python_path = str(repo / "python")
    sys.path.insert(0, python_path)
    
    # Set PYTHONPATH for subprocess
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": python_path}
    
    # Create logs directory
    logs_dir = repo / "modal" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"offline_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Run the benchmark script and save output
    bench_script = repo / "benchmark" / "offline" / "bench.py"
    with open(log_file, "w") as f:
        result = subprocess.run(
            [sys.executable, "-u", str(bench_script)],
            cwd=str(repo),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    print(f"✓ Logs saved to: {log_file}")
    return result.returncode


@app.function(image=image, gpu="A10G", timeout=3600)
def profile_prepare_replay():
    """Profile prepare_for_replay() copy operations for Fused Copy optimization"""
    import subprocess
    import sys
    from datetime import datetime
    from pathlib import Path
    
    # Clone repo
    repo = Path("/root/minisglang")
    if not repo.exists():
        subprocess.run(["git", "clone", "https://github.com/sgl-project/mini-sglang.git", str(repo)], check=True)
    
    # Read script from cloned repo (should be committed to repo)
    repo_script = repo / "modal" / "profile_prepare_replay.py"
    if not repo_script.exists():
        raise RuntimeError(
            f"Profile script not found at {repo_script}. "
            "Please commit profile_prepare_replay.py to the repo:\n"
            "  git add modal/profile_prepare_replay.py\n"
            "  git commit -m 'Add profile script for Modal benchmarks'"
        )
    script_content = repo_script.read_text()
    print(f"✓ Read script from cloned repo: {repo_script} ({len(script_content)} bytes)")
    
    # Set up Python path
    python_path = str(repo / "python")
    sys.path.insert(0, python_path)
    
    # Write script to temp location
    script_file = Path("/tmp/profile_prepare_replay.py")
    script_file.write_text(script_content)
    
    # Create logs directory
    logs_dir = repo / "modal" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"prepare_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Set PYTHONPATH for subprocess
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": python_path}
    
    # Run profiling script and save output
    with open(log_file, "w") as f:
        result = subprocess.run(
            [sys.executable, "-u", str(script_file)],
            cwd=str(repo),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    print(f"✓ Profiling logs saved to: {log_file}")
    return result.returncode


@app.local_entrypoint()
def main(profile: bool = False):
    """Run benchmarks on Modal GPU.
    
    Args:
        profile: If True, run prepare_for_replay profiling. Otherwise run offline benchmark.
    """
    if profile:
        profile_prepare_replay.remote()
    else:
        run_offline_benchmark.remote()
