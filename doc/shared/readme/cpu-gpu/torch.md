On Linux, the default PyTorch packages on PyPI include CUDA libraries of several GB. `nerva-torch` currently
runs on the CPU only, so the much smaller CPU-only build of PyTorch is sufficient. Install it before `nerva-torch`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for other platforms.
