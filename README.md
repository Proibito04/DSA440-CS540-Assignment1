# CS540 — AI Ethics in Early-Warning Systems for Student Outcome Prediction

**Group:** Altınüzengi, Balcı, Günindi, Balzano — Sabancı University  
**Course:** CS540 Data & AI Ethics  
**Dataset:** [OULAD](https://analyse.kmi.open.ac.uk/open-dataset)

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management (fast, pip-compatible).

**1. Install uv (one-time)**

```bash
pip install uv
```

**2. Install dependencies and create the virtual environment**

```bash
# From the code/ directory — reads pyproject.toml, creates .venv and installs everything:
uv sync
```

**3. Activate the virtual environment**

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**4. Register the kernel for Jupyter**

```bash
uv run python -m ipykernel install --user --name cs540-gp --display-name "Python cs540-gp"
```

**5. Launch Jupyter**

```bash
jupyter notebook
```

> When opening any notebook, make sure the kernel is set to **"Python cs540-gp"** (top-right in Jupyter or via *Kernel → Change Kernel* in VS Code).

## Dataset

Download OULAD from https://analyse.kmi.open.ac.uk/open-dataset and place the CSV files into `data/`.

Required files:
- `studentInfo.csv`
- `studentRegistration.csv`
- `studentAssessment.csv`
- `studentVle.csv`
- `assessments.csv`
- `courses.csv`
- `vle.csv`

## Notebook Usage

### Step 1 — Run the Group notebooks first (everyone must do this)

The first two notebooks are the shared foundation. **All individual notebooks depend on the artifacts they produce**, so they must be run before working on any principle-specific notebook.

#### `01_data_exploration.ipynb` — EDA (Group)

Explores the raw OULAD dataset. Produces:
- Class distribution of `final_result` (Pass / Fail / Withdrawn / Distinction)
- Missing value report and automated fill for `imd_band`, `age_band`
- Demographic breakdown by gender, disability, IMD band — used as the fairness baseline
- VLE (click activity) and assessment score distributions
- Feature correlation heatmap

No output files are saved here. The purpose is to build shared understanding of the data.

#### `02_base_model.ipynb` — Base Model (Group)

Trains the shared Random Forest classifier and saves the model and test data for downstream notebooks.

**Why a shared base model?**  
Each principle module (Transparency, Fairness, Privacy, Awareness) analyses or modifies *the same* model so that results are comparable across modules. If each person trained their own model, cross-module comparisons (e.g. fairness before/after explainability masking) would be meaningless.

Produces the following files into `data/` (gitignored):

| File | Contents |
|---|---|
| `data/base_model.pkl` | Trained `RandomForestClassifier` |
| `data/X_train.pkl` | Training features |
| `data/X_test.pkl` | Test features |
| `data/y_test.pkl` | Test labels |

> **Run this notebook every time you pull a significant update to `src/data_loader.py` or `src/model.py`**, as changes there will affect model behaviour.

---

### Step 2 — Individual principle notebooks

Each notebook loads the shared artifacts from `data/` and focuses on one ethical principle. They can be run independently of each other (but always after Step 1).

| Notebook | Principle | Owner | Key inputs |
|---|---|---|---|
| `03_transparency_explainability.ipynb` | Transparency & Explainability | Çağrı | `base_model.pkl`, `X_train.pkl`, `X_test.pkl` |
| `04_fairness.ipynb` | Fairness & Non-Discrimination | Agit | `base_model.pkl`, `X_test.pkl`, `y_test.pkl` |
| `05_privacy.ipynb` | Privacy | Edoardo | `base_model.pkl`, `X_train.pkl`, `X_test.pkl` |
| `06_awareness_literacy.ipynb` | Awareness & Literacy | Yasemin | `base_model.pkl`, `X_test.pkl` |

**Loading shared artifacts (common pattern for all principle notebooks):**

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))  # makes src/ importable

import joblib
data_dir = Path.cwd().parent / "data"

model   = joblib.load(data_dir / "base_model.pkl")
X_train = joblib.load(data_dir / "X_train.pkl")
X_test  = joblib.load(data_dir / "X_test.pkl")
y_test  = joblib.load(data_dir / "y_test.pkl")
```

---

### Step 3 — Integrated pipeline (Group, final)

`07_integrated_pipeline.ipynb` runs the full ethical pipeline end-to-end in a single notebook. It will be assembled after all individual notebooks are complete.

## Project Structure

```
code/
├── data/                  # OULAD CSVs + model artifacts (gitignored)
├── notebooks/             # Jupyter notebooks (run in order)
├── src/                   # Shared Python modules
│   ├── data_loader.py     # Data loading & feature engineering (Group)
│   ├── model.py           # Model training & evaluation (Group)
│   ├── explainability.py  # SHAP + LIME + dual-view (Çağrı)
│   ├── fairness.py        # Fairness metrics + mitigation (Agit)
│   ├── privacy.py         # Feature masking + k-anonymity (Edoardo)
│   └── awareness.py       # Annotation + literacy layer (Yasemin)
├── plans/                 # Local planning docs (gitignored)
├── requirements.txt
└── README.md
```
