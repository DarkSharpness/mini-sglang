# Mini-sglang

## How to install

```bash
git submodule update --init --recursive
pip install -e ".[dev]"
cd 3rdparty/tvm-ffi
pip install -e .
```

Or use `uv` to manage the dependencies:

```bash
git submodule update --init --recursive
uv sync --extra=dev
```
