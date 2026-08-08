# 🌳 06 — Hyperparameter Tuning for Random Forest

> Going beyond `.fit()` with default settings — systematically searching for the Random Forest configuration that actually performs best, and visualizing the entire hyperparameter landscape instead of just reporting one final number.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/status-learning%20project-yellow)

---

## ⚡ TL;DR — 30-Second Summary

- **Problem:** A default `RandomForestClassifier` picks its own hyperparameters — but "default" rarely means "best." How much is actually on the table by tuning them properly?
- **Approach:** Built a baseline Random Forest on a synthetic classification dataset, then used `GridSearchCV` to exhaustively search across 100 combinations of `max_features` and `n_estimators` (5-fold cross-validation, 500 total model fits).
- **Result:** Baseline model scored **0.85** accuracy on the held-out test set; the best tuned combination (`max_features=2`, `n_estimators=160`) reached a **0.86** cross-validated score.
- **What makes this more than "just run GridSearchCV":** the entire hyperparameter search space is visualized as a 2D contour plot and a 3D surface plot — turning 100 abstract number combinations into a landscape you can actually *see*, including where accuracy plateaus and where it doesn't.

---

## 💼 Skills Demonstrated

| Competency | Where it shows up |
|---|---|
| **Hyperparameter Tuning** | `GridSearchCV` with 5-fold cross-validation across a 2-parameter grid (100 combinations) |
| **Baseline-First Methodology** | Established an untuned baseline before tuning, so improvement can actually be measured against something |
| **Data Visualization** | 2D contour and 3D surface plots (Plotly) to visualize how accuracy responds across the entire hyperparameter grid, not just at the single best point |
| **Working with Synthetic Data** | Using `make_classification` to generate a controlled dataset — useful for isolating and studying tuning behavior without real-world data noise getting in the way |
| **Critical Evaluation** | Recognizing where the baseline-vs-tuned comparison needs a caveat (see [Notes & Limitations](#-notes--limitations)) rather than treating every metric at face value |

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Baseline Model](#-baseline-model)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Visualizing the Hyperparameter Landscape](#-visualizing-the-hyperparameter-landscape)
- [Results & Key Insights](#-results--key-insights)
- [Notes & Limitations](#-notes--limitations)
- [Tech Stack](#️-tech-stack)
- [How to Run](#️-how-to-run)
- [Project Structure](#️-project-structure)
- [Acknowledgments & Credits](#-acknowledgments--credits)
- [Author](#-author)

---

## 📖 Project Overview

Most introductory ML projects stop at "train a model, report the accuracy." This project goes one step further: it asks *how much of that accuracy number is actually the algorithm, and how much is just the default settings scikit-learn happened to ship with?* Using a Random Forest classifier as the test subject, this project establishes a baseline, then systematically searches for better hyperparameters with `GridSearchCV`, and visualizes the full search space to build real intuition for how `max_features` and `n_estimators` interact.

---

## 🎯 Objectives

- Understand what hyperparameters are, and how they differ from the parameters a model *learns* on its own.
- Build a baseline Random Forest model to establish a "before tuning" reference point.
- Use `GridSearchCV` to systematically search a hyperparameter grid rather than guessing values by hand.
- Visualize the relationship between hyperparameter values and model accuracy using contour and 3D surface plots.
- Practice interpreting cross-validated results critically, not just reading off the single best score.

---

## 📂 Dataset

This project uses a **synthetic** dataset generated with `sklearn.datasets.make_classification()` rather than a real-world dataset — a deliberate choice, since the goal here is to isolate and study tuning behavior without the extra noise of real data cleaning or messy features.

| Attribute | Value |
|---|---|
| Samples | 200 |
| Features | 10 (all informative — `n_redundant=0`) |
| Classes | 2 (binary classification) |
| Generation seed | `random_state=42` (reproducible dataset generation) |

Split into an 80/20 train/test set: **160 training rows**, **40 test rows**.

---

## 🔄 Project Workflow

```text
Generate Synthetic Dataset (make_classification)
      │
      ▼
Train/Test Split (80/20)
      │
      ▼
Baseline Random Forest
      ├── Train with a fixed, untuned configuration
      └── Evaluate on the held-out test set
      │
      ▼
Hyperparameter Tuning (GridSearchCV)
      ├── Search grid: max_features (1–5) × n_estimators (10–200, step 10)
      ├── 5-fold cross-validation → 500 total model fits
      └── Identify best-performing combination
      │
      ▼
Visualize the Search Space
      ├── Build a DataFrame of every combination's accuracy
      ├── Pivot into a max_features × n_estimators grid
      ├── 2D Contour Plot
      └── 3D Surface Plot
```

---

## 🎯 Baseline Model

A `RandomForestClassifier` was trained with a fixed, untuned configuration (`n_estimators=100`) as the reference point before any tuning:

| Metric | Score |
|---|---|
| Test Set Accuracy | **0.85** |

> 📎 Note: the baseline was configured with `max_features=100` — since the dataset only has 10 features, scikit-learn internally caps this at the actual feature count, so in practice this baseline uses *all* available features at each split.

---

## 🎛️ Hyperparameter Tuning

`GridSearchCV` was used to search across two key Random Forest hyperparameters:

```python
max_features_range = np.arange(1, 6, 1)        # 1 to 5
n_estimators_range = np.arange(10, 210, 10)     # 10 to 200

param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
```

This tests **5 × 20 = 100 unique combinations**, each validated with 5-fold cross-validation — **500 total model fits** in one search.

**Best combination found:**

| Parameter | Best Value |
|---|---|
| `max_features` | 2 |
| `n_estimators` | 160 |
| Cross-Validated Accuracy | **0.86** |

---

## 📊 Visualizing the Hyperparameter Landscape

Rather than only reporting the single best combination, every one of the 100 tested combinations and their accuracy scores was kept, reshaped into a `max_features` × `n_estimators` grid, and plotted two ways:

- **2D Contour Plot** — shows accuracy as a color gradient across the grid, making it easy to spot which regions of the hyperparameter space perform well versus poorly at a glance.
- **3D Surface Plot** — the same data as a literal landscape, with accuracy as the height (z-axis) — useful for seeing peaks, plateaus, and how sensitive accuracy is to each parameter.

Together, these make one thing visible that a single "best params" printout can't: accuracy climbs quickly with `max_features` from 1 to 2, then largely **plateaus** — pushing `n_estimators` higher past a certain point gives diminishing returns rather than steady improvement.

---

## 📈 Results & Key Insights

| Model | Configuration | Accuracy |
|---|---|---|
| Baseline | `n_estimators=100`, all features considered | 0.85 (test set) |
| Tuned | `max_features=2`, `n_estimators=160` | 0.86 (cross-validated) |

- **The tuning gain here is modest** — about one percentage point. That's a realistic and honest outcome: on a clean, well-behaved synthetic dataset, a default Random Forest is often already close to what careful tuning can achieve. The value of this project isn't a dramatic accuracy jump; it's understanding *how* to search and *how* to visualize that search properly, so the same process pays off more on messier, real-world data.
- **`max_features` mattered more than `n_estimators`** in this search — the contour plot shows accuracy rising sharply as `max_features` goes from 1 to 2, while adding more trees (`n_estimators`) past roughly 50–100 gave comparatively little additional benefit.

---

## ⚠️ Notes & Limitations

- **The baseline (0.85) and tuned (0.86) scores aren't measured on exactly the same basis.** The baseline accuracy comes from a single evaluation on the held-out test set, while the tuned score is the mean 5-fold cross-validation score from `GridSearchCV` on the training data only. For a fully apples-to-apples comparison, the best tuned model should also be re-evaluated on the same `X_test`/`y_test` split used for the baseline.
- **No fixed `random_state` in the `train_test_split` call.** Combined with the small dataset (200 samples), this means the exact baseline accuracy can shift somewhat between re-runs — worth pinning down for a fully reproducible comparison.
- **This is a synthetic, well-separated dataset.** The relatively small tuning gain (0.85 → 0.86) is specific to this controlled setup; real-world datasets with more noise and imbalance often show a bigger gap between default and tuned hyperparameters.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Machine Learning | `scikit-learn` (`RandomForestClassifier`, `GridSearchCV`) |
| Data Handling | `pandas`, `numpy` |
| Visualization | `Plotly` (contour & 3D surface plots) |
| Environment | Jupyter Notebook / Google Colab |

---

## ▶️ How to Run

### Option 1 — Run on Google Colab (Recommended)

1. Open `hyperparameter_tuning_Random_Forest.ipynb` in Google Colab.
2. Run all notebook cells from top to bottom.
3. The dataset is generated synthetically inside the notebook — no external data or downloads needed.

### Option 2 — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/RoihansLab/Machine-Learning-Projects.git
cd Machine-Learning-Projects/06_hyperparameter_tuning_Random_Forest
```

**2. Install the required Python libraries**
```bash
pip install pandas numpy scikit-learn plotly jupyter
```

**3. Launch Jupyter Notebook**
```bash
jupyter notebook hyperparameter_tuning_Random_Forest.ipynb
```

Then open the notebook and run all cells sequentially.

---

## 🗂️ Project Structure

```text
06_hyperparameter_tuning_Random_Forest/
│
├── hyperparameter_tuning_Random_Forest.ipynb   → Baseline model, GridSearchCV tuning,
│                                                   and hyperparameter landscape visualization
└── README.md                                    → You are here
```

---

## 🙏 Acknowledgments & Credits

A heartfelt thank you to **Data Professor** on YouTube. The fundamental understanding and hands-on practice behind this project were made possible by his excellent, clearly explained tutorial. Supporting and crediting the creators who make quality Machine Learning education freely accessible is something I take seriously as part of this developer community.

📺 [Hyperparameter Tuning of Machine Learning Model in Python](https://www.youtube.com/watch?v=jUxhUgkKAjE&list=PLtqF5YXg7GLltQSLKSTnwCcHqTZASedbO&index=3)

---

## 👤 Author

**Roihan Saputra**
*Aspiring Data Scientist / Machine Learning Engineer*
GitHub: [https://github.com/RoihansLab](https://github.com/RoihansLab)

Open to feedback, collaboration, or a conversation about this project — feel free to reach out via GitHub.
