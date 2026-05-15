# CS540 — AI Ethics in Early-Warning Systems for Student Outcome Prediction

**Group:** Altınüzengi, Balcı, Günindi, Balzano — Sabancı University  
**Course:** CS540 Data & AI Ethics  
**Dataset:** [OULAD](https://analyse.kmi.open.ac.uk/open-dataset)

## Setup

```bash
pip install -r requirements.txt
```

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

## Notebook Order

| Notebook | Description | Owner |
|---|---|---|
| `01_data_exploration.ipynb` | EDA, missing values, class distribution | Group |
| `02_base_model.ipynb` | Feature engineering, RF/XGB training, baseline metrics | Group |
| `03_transparency_explainability.ipynb` | SHAP + LIME, dual-view output | Çağrı |
| `04_fairness.ipynb` | Fairness metrics, bias mitigation | Agit |
| `05_privacy.ipynb` | Privacy-aware preprocessing, feature masking | Edoardo |
| `06_awareness_literacy.ipynb` | Awareness layer, UI annotations | Yasemin |
| `07_integrated_pipeline.ipynb` | Full ethical pipeline — **FINAL SUBMISSION** | Group |

## Project Structure

```
code/
├── data/                  # OULAD CSVs (gitignored)
├── notebooks/             # Individual principle notebooks
├── src/                   # Shared Python modules
│   ├── data_loader.py
│   ├── model.py
│   ├── explainability.py  (Çağrı)
│   ├── fairness.py        (Agit)
│   ├── privacy.py         (Edoardo)
│   └── awareness.py       (Yasemin)
├── plans/                 # Local planning docs (gitignored)
├── requirements.txt
└── README.md
```
