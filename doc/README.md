# Documentation of the Nerva repositories

This guide is for maintainers. It describes how the documentation of the Python backends (nerva-jax, nerva-numpy,
nerva-tensorflow and nerva-torch) is organized, built and published, what the CI checks, and what to keep in mind
before changing it. The C++ repositories (nerva-rowwise and nerva-colwise) are not covered yet; see
[Not covered yet](#not-covered-yet).

## What documentation exists

Each Python backend `nerva-<b>` contains:

| Documentation | Location | Notes |
|---------------|----------|-------|
| README | `README.md` | Generated from [`shared/readme`](shared/readme), do not edit in the backend |
| Manual (AsciiDoc) | `doc/asciidoc/nerva-<b>.adoc` | The main file; it includes the files below |
| | `doc/asciidoc/include/*.adoc` | Copied from [`shared/python`](shared/python), do not edit in the backend |
| | `doc/asciidoc/settings.adoc` | Repository-specific attributes, see [below](#differences-between-repositories-in-the-shared-asciidoc-files) |
| | `doc/latex/nerva.bib` | Bibliography, used by asciidoctor-bibtex |
| API reference (Sphinx) | `doc/sphinx/index.rst`, `doc/sphinx/conf.py` | Generated from the docstrings with autosummary. Every module must be listed in `index.rst`. Framework imports are mocked in `conf.py` (`autodoc_mock_imports`) |
| Landing page | `doc/index.html` | Links to the manual and the API reference |

The manual also includes code from the backends themselves, through tagged regions (`# tag::name[]` ...
`# end::name[]`):

| File | Tags |
|------|------|
| `examples/mlp_construction.py` | `construct1`, `construct2` |
| `examples/train_shapes.py` | `train` |
| `examples/train_shapes_cli.sh` | `doc` |
| `examples/mnist_cifar10/train_cifar10_cli.sh` | `doc` |
| `src/nerva_<b>/training.py` | `sgd`, `sgd_plain` |

## Sources of truth

Several files are identical (up to the backend name) in the four backends. They are maintained in one place, and
copied into each repository by a script, so that every repository stays self-contained. The copies are committed.

| Source | Copies | Script | Checked in CI by |
|--------|--------|--------|------------------|
| `nerva/doc/shared/python/*.adoc` | `doc/asciidoc/include/` | `nerva/scripts/sync_docs.py` | `docs-sync.yml` in nerva |
| `nerva/doc/shared/readme/` | `README.md` (rendered) | `nerva/scripts/sync_docs.py` | `docs-sync.yml` in nerva |
| `nerva-sympy/templates/backend_tests/` | `tests/` (shared test files) | `nerva-sympy/scripts/sync_backend_tests.py` | `tests.yml` in nerva-sympy |
| `nerva-sympy/templates/backend_src/` | `src/nerva_<b>/shapes_dataset.py` | `nerva-sympy/scripts/sync_backend_tests.py` | `tests.yml` in nerva-sympy |

Generated READMEs start with an HTML comment, and the shared AsciiDoc files with an AsciiDoc comment, that names
their source. Both scripts expect the repositories to be checked out next to each other (e.g. `../nerva-torch`; use
`--root` for another directory). Without options they write the copies; with `--check` they only report
differences (as a unified diff) and exit with status 1 if there are any:

```bash
python scripts/sync_docs.py --check              # in nerva
python scripts/sync_backend_tests.py --check     # in nerva-sympy
```

## Building the documentation locally

The build scripts are in the `doc` directory of each backend. They write to `output_directory/` in the root of the
repository.

| Output | Command | Requirements |
|--------|---------|--------------|
| `output_directory/nerva-<b>.html` (manual) | `bash doc/build_asciidoc_documentation.sh` | [Asciidoctor](https://asciidoctor.org) with `asciidoctor-bibtex` and `rouge` (`gem install asciidoctor asciidoctor-bibtex rouge`) |
| `output_directory/sphinx/` (API reference) | `bash doc/build_sphinx_documentation.sh` | `pip install sphinx sphinx-rtd-theme`; the script installs them if needed |

Both scripts also copy `doc/index.html` to `output_directory/`.

## Publication

Each backend has two workflows that publish the documentation to GitHub Pages, at `https://wiegerw.github.io/nerva-<b>/`:

| Workflow | Builds | Environment |
|----------|--------|-------------|
| `deploy-asciidoc.yml` | The manual | Container `asciidoctor/docker-asciidoctor` |
| `deploy-sphinx.yml` | The API reference | Python with Sphinx; nerva-jax uses the container `ghcr.io/wiegerw/jax-sphinx` |

They run on every push to `main`, and can be started by hand from the Actions tab ("Run workflow"). They publish
`output_directory/` to the `gh-pages` branch with `keep_files: true`, so the two workflows do not remove each
other's files.

## What the CI checks

| Repository | Workflow | Checks |
|------------|----------|--------|
| `nerva-<b>` | `tests.yml` | The tests, on all supported Python versions and with the minimum versions of the dependencies |
| nerva-sympy | `tests.yml` | The symbolic tests, and (with the four backends checked out) the equation checks, the one-layer test cases and `sync_backend_tests.py --check` |
| nerva | `docs-sync.yml` | `sync_docs.py --check` |

The workflows that check other repositories (in nerva-sympy and nerva) check out the branch with the same name in
those repositories if it exists, and `main` otherwise.

**The documentation is not built on pull requests.** A broken include, a renamed tag, or an AsciiDoc or Sphinx error
only shows up in the deploy workflows, after merging into `main`.

## Before changing documentation

1. **Find the source of truth.** Check the tables above, or the comment at the top of the file. Never edit a
   generated copy: the next sync overwrites it, and the CI check fails.
2. **Keep shared files generic.** Express differences between repositories with attributes in `settings.adoc`
   (AsciiDoc), or with tokens and per-backend fragments (README), see [below](#details-of-the-shared-files).
   Attributes in source blocks are only replaced with `subs="attributes+"`, e.g. `[source,bash,subs="attributes+"]`.
3. **Mind the included code.** Tagged regions in examples and source files are part of the manual. When you rename,
   move or delete such a file or tag, update the include in the shared AsciiDoc files. Include paths are relative to
   `doc/asciidoc/include`.
4. **New modules** must be added to `doc/sphinx/index.rst` of each backend.
5. **Sync, and use the same branch name everywhere.** After editing a source, run the sync script, and commit the
   result in every affected repository on a branch with the same name, so that the cross-repository CI checks match.
   Merge the pull requests together.
6. **Build locally** before merging (see [Building the documentation locally](#building-the-documentation-locally)),
   since the CI does not build the documentation on pull requests.

## Details of the shared files

### Differences between repositories in the shared AsciiDoc files

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

### The README template

`shared/readme/README.md` is a template. For each backend, `sync_docs.py` replaces the following tokens:

| Token | Replaced by | Example (nerva-torch) |
|-------|-------------|-----------------------|
| `nerva-BACKEND` | The name of the repository and the PyPI package | `nerva-torch` |
| `nerva_BACKEND` | The name of the Python package | `nerva_torch` |
| `{{FRAMEWORK}}` | The name of the framework | `PyTorch` |
| `{{CPU_GPU}}` | The contents of `shared/readme/cpu-gpu/<backend>.md` | Installation of the CPU-only PyTorch |

Text that differs between the backends in other ways should be put in a separate file per backend, like
`shared/readme/cpu-gpu/<backend>.md`, with a token in the template.

## Not covered yet

- nerva-rowwise and nerva-colwise have their own copies of some shared files (`io.adoc`, `introduction.adoc`,
  `matrix-operations.adoc`, `mlp-options.adoc`, `inspect-npz.adoc`), which have diverged from the versions in
  `shared/python`. To share them, add the repositories to `TARGETS` in `scripts/sync_docs.py` and define the
  attributes `prepare-data-script` and `datasets-dir` in their `settings.adoc`.
- `mlp-options.adoc` and `nerva-python-data-handling.adoc` in `shared/python` are not included by the manuals of
  the Python backends.
