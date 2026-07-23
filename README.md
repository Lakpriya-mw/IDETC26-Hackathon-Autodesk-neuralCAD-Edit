# IDETC 2026 Hackathon
# Autodesk neuralCAD-Edit Problem

This repo contains the code for the Autodesk neuralCAD-Edit hackathon problem, based on the 3D CAD editing dataset and benchmark introduced in the paper: [neuralCAD-Edit: An Expert Benchmark for Multimodal-Instructed 3D CAD Model Editing](https://autodeskailab.github.io/neuralCAD-Edit/)

![fig_01](img/fig_1.png)


We provide:
- [The problem statement PDF.](./Autodesk%20One-Page-Problem-Statement-CIE-Hackathon_2026.pdf)
- [The introductory presentation.]()
- A dataset of 48 text-based editing requests and associated edits.
- Code for accessing the data.
- Notebooks for visualising and analysing the data.
- Harness code which allows foundation models to perform edits with iterative cadquery script refinement.
- All outputs of the foundation models we run in the paper and report results on.
- All automatic, human and VLM evaluations.

Instructions are provided below, but please get in touch if you have any questions.

### Setup

1. Clone this repo
2. Install dependencies (recommended, cross-platform):

```bash
uv sync
```

Run scripts with `uv run python ...` or set `PYTHONPATH` to the repo root.

**Conda fallback** (macOS/Linux): `conda env create -f environment.yml`

**Config notes:**
- Set `models_dir.paths` in `src/config/edit_192_external.json` to your harness output directory before ingesting model results.
- The database uses the `breps` collection by default (`breps_collection` in config). Do not set it to `breps_v2` unless your data uses that collection.

### Download and visualise data

> **⚠️ Important:**  
> This hackathon problem uses data that is **significantly different** from the original dataset.  
>  
> **Use the data linked below for this competition, do _not_ use previous versions of the dataset.**

1. Download and extract the pre-computed database zip into `data/edit_192_external` (or another location and update `storage_dir` in `src/config/edit_192_external.json`). The slimmed hackathon dataset contains 48 text-conditioned edit requests with paper-matching DINO features, human GT edits, and foundation-model baselines.
2. To keep only text-conditioned requests in the database, run `uv run python src/scripts/filter_dataset_text_only.py --config src/config/edit_192_external.json`.
3. Try using `src/notebooks/visualise_examples.ipynb` to look at some of the data.
### Repo structure

- `src/config` - config jsons
- `src/harnesses` - contains the cadquery harness
- `src/notebooks` - notebook for visualising the database etc
- `src/scripts` - for running stages of the evaluation pipeline
- `src/scripts/benchmark_inference` - scripts to run a foundation model in a harness
- `src/scripts_grundtruth` - used to export/ingest for human labelling on AWS groundtruth
- `src/scripts_preprocess` - cadquery_convert.py for headless STEP to STL/PNG export
- `src/utils` - contains database, feature extraction, vlm rating etc.
- `src/vlms` - a base VLM class with provider specific files which inherit

### Running your own foundation model/harness

- You can either modify ours (see the cadquery harness in `src/vlms/base_vlm.py`, `src/harnesses/cadquery_script.py` and `src/scripts_benchmark_inference/run_harness.py` ). You can update model settings in the config, and an example of how to run is in `src/scripts/run_all.sh` (macOS/Linux) or `src/scripts/run_all.ps1` (Windows).
- Or you can write your own.
- Either way, you must produce a .step file and a settings.json with the appropriate data in. You can see what we do in `src/scripts_benchmark_inference/run_harness.py`
- You must also provide a single topright isometric view, and 6 orthographic views (top, bottom, front, back, left, right), and ensure they are in the correct file structure to be ingested. Additionally, some metrics require .stls. If the harness does not natively output these files, use `src/scripts_preprocess/cadquery_convert.py` to export them headless from `.step` files.
- An single example model output ready to be ingested is in `example_data`. As long as your model output matches this file/folder structure, it will be OK.
- Add your output path to `models_dir` in the config.
- Ingest and run the evaluation in `src/scripts/run_all.sh` (macOS/Linux) or `src/scripts/run_all.ps1` (Windows)
- You can then use `src/notebooks/leaderboard.ipynb` to display the results.
- Note that these give you the automatic metrics only. We'll gladly run the human evals for you.

### Database Schema

The dataset/benchmark is organised in a local mongita database with the following schemas

- requests: contains all information about the request (text instruction, etc).
- edits: contains all the information about an edit (screengrabs, edit events, etc.)
- users: humans/ML models who have created requests, edits, or evaluations
- breps: contains all the information about a brep: .step, .stl, iso and 6-orthographic images, dino v2 features etc.
- ratings: ratings performed on an edit by a user

All objects (e.g. step files) live outside the database in the file tree, and are pointed to by their relative filepaths from the database. See `src/notebooks/visualise_examples.ipynb` for example access patterns.

![Database Schema](img/database_schema.svg)

### Citation

```bibtex
@inproceedings{perrett2026neuralcadedit,
  title={neuralCAD-Edit: An Expert Benchmark for Multimodal-Instructed 3D CAD Model Editing},
  author={Perrett, Toby and Bouchard, Matthew and McCarthy, William},
  booktitle={arXiv preprint arXiv:2604.16170},
  year={2026}
}
```