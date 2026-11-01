# Shared documentation

The files in this directory are copied to the Nerva repositories by `scripts/sync_docs.py`:

| Directory | Copied to |
|-----------|-----------|
| `python` | `doc/asciidoc/include` of nerva-jax, nerva-numpy, nerva-tensorflow and nerva-torch |
| `readme` | `README.md` of nerva-jax, nerva-numpy, nerva-tensorflow and nerva-torch (rendered from a template) |

See [`doc/README.md`](../README.md) for how the documentation is organized, built, published and checked, and for
what to keep in mind before changing it.
