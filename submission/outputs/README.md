# An Agentic harness for IDETC Hackathon 2026 - Autodesk Challenge

Team name: Design Syndicate 

Team members: Lakpriya Weragoda, Ali Hamza

---
## Environment

Python **3.12** (cadquery does not support 3.14). From the repo root:

```bash
conda create -y -n neuralcad-edit python=3.12
conda activate neuralcad-edit
pip install -r lw_solution/requirements.txt
```

Set up the Anthropic API key.

---

## Running and testing the harness

> Safe test of the workflow (no API calls), & tells you what is missing,

```bash
python lw_solution/scripts/selftest.py
```
> Run the actual harness. Look at scripts/run_harness.py for details about possible arguments

```bash
python lw_solution/scripts/run_harness.py --limit 1
```

> Evaluate generated outputs

```bash
python lw_solution/scripts/run_all.ps1
```

(or `bash lw_solution/scripts/run_all.sh`) — self-test, all 48 tasks, backfill
`.stl` + views, ingest, score, plots.

---
