# Running This Project on Google Colab

The repository at `github.com/caltinuzengi/CS540-ethics-in-data-ai-project` is the project root — `notebooks/`, `src/`, and `data/` live directly inside it.

---

## Step 1 — Clone the Repository

```python
!git clone https://github.com/caltinuzengi/CS540-ethics-in-data-ai-project.git
%cd CS540-ethics-in-data-ai-project
```

---

## Step 2 — Install Dependencies

Colab runs its own Python kernel — no virtual environment is needed.  
Use `uv` with `--system` to install packages directly into Colab's active Python environment:

```python
!pip install uv -q
!uv pip install --system -r requirements.txt
```

> **Why not `uv sync`?** `uv sync` creates a `.venv` that Colab's kernel cannot use.  
> `--system` installs into the kernel's Python directly, no restart required.

---

## Step 3 — Make `src/` Importable

```python
import sys, os
sys.path.append(os.getcwd())   # adds the repo root (where src/ lives) to the path
```

> Include this at the top of every notebook session.  
> It replaces the local `sys.path.append(str(Path.cwd().parent))` line that is already in each notebook.

---

## Step 4 — Load the OULAD Dataset

The dataset is gitignored and must be added manually. Two options:

### Option A — Mount Google Drive (Recommended)

Upload the OULAD CSV files to a Google Drive folder once, then copy them each session:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
from pathlib import Path

src = Path('/content/drive/MyDrive/OULAD')   # ← adjust to your Drive path
dst = Path('/content/CS540-ethics-in-data-ai-project/data')
dst.mkdir(exist_ok=True)

for f in src.glob('*.csv'):
    shutil.copy(f, dst / f.name)

print("CSV files ready:", [f.name for f in dst.glob('*.csv')])
```

### Option B — Upload a Zip File

```python
from google.colab import files
uploaded = files.upload()   # select your OULAD zip

import zipfile
with zipfile.ZipFile(list(uploaded.keys())[0], 'r') as z:
    z.extractall('/content/CS540-ethics-in-data-ai-project/data/')
```

---

## Step 5 — Open Notebooks

### Option A — Open directly from GitHub in Colab

| Notebook | Colab Link |
|---|---|
| `01_data_exploration` | [Open]() |
| `02_base_model` | [Open]() |
| `03_transparency_explainability` | [Open]() |
| `04_fairness` | [Open]() |
| `05_privacy` | [Open]() |
| `06_awareness_literacy` | [Open]() |

> When opening from GitHub, Colab shows a read-only copy. Save a personal copy with  
> **File → Save a copy in Drive** before making any changes.

### Option B — Run from the cloned repo (after Step 1)

After cloning, the notebooks are at `/content/CS540-ethics-in-data-ai-project/notebooks/`.  
You can open them from the Colab file browser (left panel 📁).

---

## Step 6 — Run in Order

Always run `01` and `02` first — they produce the shared model artifacts that all other notebooks depend on.

| Step | Notebook | Outputs |
|---|---|---|
| 1 | `01_data_exploration.ipynb` | — (EDA only) |
| 2 | `02_base_model.ipynb` | `data/base_model.pkl`, `X_train.pkl`, `X_test.pkl`, `y_test.pkl` |
| 3+ | Individual principle notebooks | — |

---

## If Your Runtime Resets

Colab sessions are not persistent. After a reset, redo **Steps 1–4** and re-run `02_base_model.ipynb`  
to regenerate the `.pkl` artifacts. Your Drive files are always safe.

---

## All-in-One Setup Cell

Paste this into the first cell of any session for a one-click setup:

```python
# ── Clone ────────────────────────────────────────────────
!git clone https://github.com/caltinuzengi/CS540-ethics-in-data-ai-project.git
%cd CS540-ethics-in-data-ai-project

# ── Dependencies ─────────────────────────────────────────
!pip install uv -q
!uv pip install --system -r requirements.txt

# ── Path ─────────────────────────────────────────────────
import sys
sys.path.append('/content/CS540-ethics-in-data-ai-project')

# ── Data (Google Drive) ──────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import shutil
from pathlib import Path

src = Path('/content/drive/MyDrive/OULAD')   # ← adjust to your Drive path
dst = Path('data')
dst.mkdir(exist_ok=True)
for f in src.glob('*.csv'):
    shutil.copy(f, dst / f.name)

print("Setup complete. Run notebooks in order: 01 → 02 → your module.")
```
