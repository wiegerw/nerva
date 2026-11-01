# Shared documentation

The Nerva repositories are self-contained: each of them contains its complete documentation. Documentation that is
used by several repositories is maintained here, and copied to the repositories with

```bash
python scripts/sync_docs.py            # copy the files to the sibling checkouts ../nerva-<name>
python scripts/sync_docs.py --check    # report differences, exit with status 1 if there are any
```

The script expects the repositories to be checked out next to this one, e.g. `../nerva-torch` (use `--root` to
specify another directory). The check runs in the "Shared documentation" workflow
(`.github/workflows/docs-sync.yml`) on every push and pull request; it can also be started by hand from the Actions
tab ("Run workflow"). Do not edit the copies in the repositories directly: edit the
files here, and run the script.

| Directory | Copied to | Contents |
|-----------|-----------|----------|
| `python` | `doc/asciidoc/include` of nerva-jax, nerva-numpy, nerva-tensorflow and nerva-torch | The AsciiDoc manual of the Python backends |

The C++ repositories (nerva-rowwise and nerva-colwise) have their own copies of some of these files, such as `io.adoc`,
`introduction.adoc` and `matrix-operations.adoc`. They will be added here later.

## Differences between repositories

The shared files are identical in all repositories. Differences are expressed with AsciiDoc attributes that each
repository defines in `doc/asciidoc/settings.adoc`:

| Attribute | Meaning | Example (nerva-torch) |
|-----------|---------|-----------------------|
| `library` | The name of the library | `_Nerva-Torch Python Library_` |
| `module` | The name of the Python package | `nerva_torch` |
| `mlptool` | The name of the command line tool for training MLPs | `mlp.py` |
| `repo-url` | The URL for links to files in the repository | `https://github.com/wiegerw/nerva-torch/blob/main` |
| `prepare-data-script` | The location of the script that downloads MNIST and CIFAR-10 | `examples/mnist_cifar10/prepare_data.py` |
| `datasets-dir` | The directory where the examples expect the downloaded datasets | `examples/mnist_cifar10/data` |

In source blocks, attributes are only replaced if the block has `subs="attributes+"`, e.g.
`[source,bash,subs="attributes+"]`.
