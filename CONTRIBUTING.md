# Contributing

Issues and pull requests are welcome.

```bash
pip install -e ".[all,dev]"
ruff check .
pytest            # GUI tests need a display: xvfb-run -a pytest on headless Linux
```

- Keep the pipeline UI-agnostic. The GUI and CLI only call `quantabench.chem.pipeline`.
- New engines go in `chem/engines.py`. Return an ASE calculator, declare supported elements and GPU
  capability, and explain unavailability in `available()` instead of raising on import.
- Heavy optional packages (torch, aimnet, dxtb, ovito) must be imported lazily so the core install
  works without them.
- Paths: use `pathlib`; everything must run on Windows and Linux (macOS later).
- Never commit data, results, model weights, or `settings.json`.
