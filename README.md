# PathSnap – Map-Matching and Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University (RR campus) – Team PT-008

| Name | SRN | GitHub |
|---|---|---|
| Allen Anish | PES1UG24AM029 | allen27anish-svg |
| Aman Priyansh | PES1UG24AM030 | AMAN-PRIYANSH |
| Amar Patil | PES1UG24AM031 | amarpatil2004 |
| Amith B A | PES1UG24AM032 | amithba |

PathSnap matches noisy GPS points to the road network:
raw files → candidate road segments per GPS point → features → classifier probability per
candidate → HMM/Viterbi picks one connected route → FastAPI (Laptop 1) → browser UI (Laptop 2).

## Code map – which file does what (in pipeline order)

| Step | Deliverable component | Code | Notebook (runs it, shows results) | Results |
|---|---|---|---|---|
| 1 | Data processing: read + inspect | `pathsnap/data_loading.py` | `notebooks/01_EDA.ipynb` | `results/01_eda/` |
| 2a | Distances in metres | `pathsnap/geometry.py` | 01, 02 | |
| 2 | Preprocessing: projection, candidate generation, label alignment | `pathsnap/preprocessing.py` | `notebooks/02_Preprocessing_and_Feature_Engineering.ipynb` | `results/02_preprocessing/` |
| 3 | Feature engineering | `pathsnap/feature_engineering.py` | 02 | `results/02_preprocessing/` |
| 4 | Model training + evaluation (7 models, plots, confusion matrices) | training loop in notebook 03, metrics in `pathsnap/metrics.py` | `notebooks/03_Model_Training_and_Comparison.ipynb` | `results/03_models/`, `models/*.joblib` |
| 5 | Bayesian hyperparameter optimisation | search space `configs/bayes_search_space.json` | `notebooks/04_Bayesian_Optimisation.ipynb` | `results/04_bayes_opt/`, `models/final_model.joblib` |
| 6 | HMM and Viterbi route reconstruction | `pathsnap/hmm_viterbi.py` | `notebooks/05_HMM_Viterbi.ipynb` | `results/05_hmm/`, `models/hmm_config.json` |
| 7 | End-to-end prediction + server-side scoring | `pathsnap/pipeline.py` | 05 | |
| 8 | API (endpoints + model loading) | `api/app.py` | – | logs in the Laptop 1 window |
| 9 | UI | `ui/index.html` | – | – |
| 10 | Deployment | `deployment/*.bat`, `requirements.txt` | – | – |
| – | Visualisations (Folium maps + plots) | plotting code inside notebooks 01, 02, 03, 05 | – | `results/maps/*.html`, `results/*/*.png` |

## Main results (student-test, 10 trips)
| Method | Top-1 accuracy | Route continuity |
|---|---|---|
| Nearest road segment (baseline) | 53.7 % | – |
| HistGradientBoosting (tuned), classifier only | 87.1 % | 93.5 % |
| Classifier + HMM/Viterbi | **89.7 %** | **97.8 %** |

Full tables: `results/03_models/consolidated_comparison_table.csv`,
`results/04_bayes_opt/untuned_vs_tuned.csv`, `results/05_hmm/student_test_hmm_vs_classifier.csv`.

## Setup (Windows, Laptop 1)
1. Install Python 3.10 or newer.
2. Double-click `deployment\INSTALL_requirements.bat` (or `py -m pip install -r requirements.txt`).
3. Data (not committed): put the folders `train\` and `student_test\` in `data\`, or keep the
   organisers' `StudentFolder\Student folder\` next to this repository, or set `PATHSNAP_DATA`.

## Reproduce all experiments
Double-click `deployment\RUN_ALL_NOTEBOOKS.bat`. It runs notebooks 01–05 in order and writes
`logs\run_all.txt` (about 20–25 minutes). Random seed 42 everywhere.

## Deployment on two laptops (same Wi-Fi or hotspot)
**Laptop 1 – API:** `deployment\laptop1_START_API.bat`. It prints this laptop's IP address and runs
`uvicorn api.app:app --host 0.0.0.0 --port 8000`. Keep the window open: every request is logged there.
It serves the `evaluator_test` trips (edit `PATHSNAP_PARTITION` inside the file to use `student_test`) and first stops any older API still using port 8000.
Click "Allow" if Windows Firewall asks.

**Laptop 2 – UI:** copy the `ui` and `deployment` folders, then:
1. `deployment\laptop2_CHECK_CONNECTION.bat` – type Laptop 1's IP; JSON means the network works.
2. `deployment\laptop2_START_UI.bat` – type `http://<Laptop 1 IP>:8000`, click **Load trips**, pick a trip.

Ground-truth routes stay on Laptop 1. The prediction path loads a trip without its route file;
the route is read only afterwards, on the server, to compute the scores shown in the UI.

## Use of AI tools
AI assistants were used for code generation and suggestions during development. Every experiment
was run on the team's own laptops (see `logs/`), the team reviewed the results reported here, and
each member can explain the files listed in the code map.
