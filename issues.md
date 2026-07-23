# neuralCAD-Edit End-to-End Test Issues

Issues found while running the hackathon pipeline as a participant (Windows, OpenAI, headless CadQuery).

| # | Issue | Where | Fix / Workaround |
|---|-------|-------|------------------|
| 1 | Dataset download link in README is still `onedrive link pending` — no automated download | [README.md](README.md) | User must manually obtain and paste dataset into `data/edit_192_external` (confirmed present: `mongita_db/`, `parquets/val_edit_all.parquet`, `breps/`) |
| 2 | `environment.yml` pins `vllm` (no native Windows wheels) and `pyobjc-framework-Cocoa` (macOS only) | [environment.yml](environment.yml) | Use `uv` with a targeted dependency set instead of `conda env create` |
| 3 | `environment.yml` omits `torch`, `transformers`, `pillow` required by DINO/CLIP metrics | [environment.yml](environment.yml), [src/utils/extract_features.py](src/utils/extract_features.py) | Install explicitly via `uv add` |
| 4 | `run_all.sh` hardcodes Mac paths (`PYTHON_PATH=/Users/perrett/...`) and all pipeline stages are commented out | [src/scripts/run_all.sh](src/scripts/run_all.sh) | Run stages directly in PowerShell with `PYTHONPATH` set to repo root |
| 5 | `models_dir.paths` is empty in config — ingest silently does nothing | [src/config/edit_192_external.json](src/config/edit_192_external.json) | Set `models_dir.paths` to harness output directory |
| 6 | Claude VLM rater (`claude-sonnet-4.5_edit-rating-6-img`) has `"skip": false` — runs live API calls during benchmark eval | [src/config/edit_192_external.json](src/config/edit_192_external.json) | Set `"skip": true` for quantitative-only eval |
| 7 | Parquet `val_edit_all.parquet` has 192 rows but only 48 have `request_text` (text-conditioned edits per PDF) | [data/edit_192_external/parquets/val_edit_all.parquet](data/edit_192_external/parquets/val_edit_all.parquet) | Created filtered `val_edit_text.parquet` (48 rows) for harness input |
| 8 | Pre-shipped dataset DB already contains 48 `gpt-5.2_cadquery-script` edits | `data/edit_192_external/mongita_db` | New harness run will add fresh edits; eval uses latest per user/request |
| 9 | `run_cadquery_script` hardcodes `"python"` subprocess — system Python lacks `cadquery`, producing empty stdout and no `tmp.step` | [src/vlms/base_vlm.py](src/vlms/base_vlm.py) | Changed to `sys.executable`; include stderr in program output |
| 10 | `cadquery_rendering.render_to_png` only supports Linux (Xw) and macOS (Cocoa/AppKit) — fails on Windows with `No module named 'AppKit'` | [src/utils/cadquery_rendering.py](src/utils/cadquery_rendering.py) | Added Windows VTK path via `cadquery.vis.show` |
| 11 | `cadquery_convert.py` crashes on Windows console with `UnicodeEncodeError` on ✓/✗ status characters | [src/scripts_preprocess/cadquery_convert.py](src/scripts_preprocess/cadquery_convert.py) | Replace unicode status chars with ASCII `OK`/`ERR`/`SKIP` |
| 12 | `probreg` missing from `environment.yml` — chamfer eval fails at import | [src/utils/evals_feature_geometric.py](src/utils/evals_feature_geometric.py) | `uv add probreg` |
| 13 | Config `breps_collection` is `breps_v2` but shipped GT breps live in `breps` (288 docs); eval can't find breps | [src/config/edit_192_external.json](src/config/edit_192_external.json) | Set `breps_collection` to `breps` |
| 14 | `transformers` 5.x `get_image_features` returns `BaseModelOutputWithPooling`, not a tensor — CLIP extract crashes | [src/utils/extract_features.py](src/utils/extract_features.py) | Added `_clip_feature_tensor()` helper for cross-version compatibility |
