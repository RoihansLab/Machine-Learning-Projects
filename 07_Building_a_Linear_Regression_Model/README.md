# 📈 07 — Building a Linear Regression Model

> Predicting continuous, real-world numbers — disease progression and house prices — using Linear Regression with Python and `scikit-learn`. This project doesn't just fit a model; it unpacks the math and the intuition behind every coefficient it produces.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/status-learning%20project-yellow)

---

## ⚡ TL;DR — 30-Second Summary

- **Problem:** Predict a continuous number — not a category — from a set of numeric features, using the simplest, most interpretable model in the ML toolbox.
- **Approach:** Train and evaluate a `LinearRegression` model on two classic regression benchmarks: the **Diabetes** dataset (predicting disease progression) and the **Boston Housing** dataset (predicting median home value).
- **Result:** R² of **0.50** on Diabetes and **0.78** on Boston Housing — a concrete, side-by-side look at how the same algorithm performs very differently depending on how much signal is actually in the data.
- **What makes this more than "just calling `.fit()`":** a dedicated breakdown of the math behind the model — what the intercept and coefficients actually *mean*, illustrated with a plain-language analogy rather than left as abstract Greek letters.

---

## 💼 Skills Demonstrated

| Competency | Where it shows up |
|---|---|
| **Regression Fundamentals** | Fitting and interpreting a `LinearRegression` model on two independent, real-world datasets |
| **Model Evaluation** | Using MSE and R² correctly, and explaining what each one actually communicates about model quality |
| **Mathematical Intuition** | Translating the regression equation and its coefficients into a plain-language mental model, not just running the fit |
| **Data Sourcing** | Loading a dataset directly from `scikit-learn` versus fetching one from an external CSV source |
| **Critical Awareness** | Flagging real methodology and dataset caveats (see [Notes & Limitations](#-notes--limitations)) instead of only reporting the headline metrics |

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Datasets](#-datasets)
- [The Math Behind the Model](#-the-math-behind-the-model)
- [Project Workflow](#-project-workflow)
- [Results & Key Insights](#-results--key-insights)
- [Notes & Limitations](#-notes--limitations)
- [Tech Stack](#️-tech-stack)
- [How to Run](#️-how-to-run)
- [Project Structure](#️-project-structure)
- [Acknowledgments & Credits](#-acknowledgments--credits)
- [Author](#-author)

---

## 📖 Project Overview

This project focuses on building a Machine Learning model using the **Linear Regression** algorithm with Python and `scikit-learn`, applied to two independent regression problems: predicting **diabetes disease progression** and predicting **median house prices**. Rather than treating Linear Regression as a black box, this project pairs each model with an explanation of *why* it produces the predictions it does — grounded in the actual coefficients the model learns.

---

## 🎯 Objectives

- Understand how Linear Regression maps input features to a continuous target value.
- Practice a full regression workflow: load → split → train → predict → evaluate → visualize.
- Correctly interpret **MSE** (Mean Squared Error) and **R² Score** as evaluation metrics.
- Build genuine intuition for what a model's **intercept** and **coefficients** represent — not just print them.
- Compare model performance across two structurally different datasets to see how data quality and feature richness affect results.

---

## 📂 Datasets

### 1. Diabetes Dataset
Loaded directly via `sklearn.datasets.load_diabetes()`.

| Attribute | Value |
|---|---|
| Instances | 442 patients |
| Features | 10 (age, sex, BMI, average blood pressure, and six blood serum measurements) |
| Target | A quantitative measure of disease progression one year after baseline |

### 2. Boston Housing Dataset
Loaded from a CSV mirror (`BostonHousing.csv`, sourced from Data Professor's public dataset repository) rather than `scikit-learn`'s built-in loader.

| Attribute | Value |
|---|---|
| Instances | 506 housing records |
| Features | 13 (crime rate, zoning, industry, proximity to the Charles River, pollution, rooms per dwelling, age, distance to employment centers, tax rate, and more) |
| Target | `medv` — median value of owner-occupied homes (in $1,000s) |

> ⚠️ See [Notes & Limitations](#-notes--limitations) for an important caveat specific to this dataset.

---

## 🧮 The Math Behind the Model

At its core, Linear Regression assumes the target value is a **weighted sum** of the input features, plus one baseline value:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

That's the whole model. The "learning" part is just `scikit-learn` finding the values of $\beta_0, \beta_1, \dots, \beta_n$ that make this equation fit the training data as closely as possible.

**An analogy that makes it click — think of a taxi fare:**

- **Intercept ($\beta_0$) — the "opening fare":** This is the baseline prediction when every single feature is exactly zero. Just like a taxi meter that starts at a fixed amount before you've traveled a single kilometer, the intercept is the model's starting point before any feature has had a chance to push the prediction up or down.
- **Coefficients ($\beta_1, \beta_2, \dots, \beta_n$) — the "rate per kilometer":** Each coefficient tells you how much the prediction changes for a one-unit increase in that specific feature, holding everything else constant. A **positive** coefficient pushes the prediction up as the feature increases; a **negative** coefficient pulls it down.

Putting real numbers from the trained Boston Housing model into the equation makes this concrete:

$$\text{medv} \approx -0.11(\text{crim}) + 0.05(\text{zn}) + 0.03(\text{indus}) + \dots + 39.02$$

Here, `crim` (per-capita crime rate) has a **negative** coefficient — as crime rate goes up, predicted home value goes down, exactly as intuition would suggest. The `39.02` at the end is the intercept: the model's baseline home value estimate before any feature adjusts it.

---

## 🔄 Project Workflow

The same four-step process is applied independently to both datasets:

```text
1. Generate / Load Dataset
      └── Diabetes: sklearn.datasets.load_diabetes()
      └── Boston Housing: pd.read_csv() from a hosted CSV
      │
      ▼
2. Split Dataset
      └── train_test_split (80% train / 20% test)
      │
      ▼
3. Modeling
      ├── Instantiate LinearRegression()
      ├── Fit on training data
      └── Predict on test data
      │
      ▼
4. Evaluate & Visualize
      ├── Mean Squared Error (MSE)
      ├── R² Score
      └── Actual vs. Predicted scatter plot
```

---

## 📊 Results & Key Insights

| Dataset | MSE | R² Score | Intercept |
|---|---|---|---|
| Diabetes | 3427.87 | **0.50** | 153.23 |
| Boston Housing | 17.87 | **0.78** | 39.02 |

- **Boston Housing fit noticeably better than Diabetes** (R² 0.78 vs. 0.50). This isn't a modeling mistake — it reflects the data itself: Boston Housing has more features (13 vs. 10) that are more directly and linearly related to price (like number of rooms or crime rate), while disease progression is a biologically noisier target that ten baseline measurements can only partially explain.
- **R² of 0.50 on the Diabetes dataset means the model explains about half of the variance** in disease progression — a realistic, unremarkable result for a simple linear model on complex biological data, not a failure of the code.
- The **Actual vs. Predicted scatter plots** for both models show points clustering around the diagonal (where actual = predicted) more tightly for Boston Housing than for Diabetes — a visual confirmation of the R² gap above.

---

## ⚠️ Notes & Limitations

A couple of things worth being upfront about:

- **No fixed `random_state` in either `train_test_split` call.** This means the exact train/test split — and therefore the exact MSE/R² numbers — will vary slightly each time the notebook is re-run. For a more reproducible benchmark, both splits should be pinned with a `random_state` (e.g., `random_state=42`).
- **The Boston Housing dataset has a documented ethical issue.** `scikit-learn` deprecated and eventually removed its built-in `load_boston()` loader (as of version 1.2) because the dataset's authors engineered a variable ("B") based on an assumption that racial self-segregation positively affected housing prices — a premise the maintainers explicitly flagged as ethically problematic. This project sources the data from a CSV mirror instead of the deprecated loader, and uses it purely as a well-known, freely available benchmark for practicing regression mechanics — not as an endorsement of how the original dataset was constructed. For any real analysis of housing prices, a dataset like `fetch_california_housing` (scikit-learn's suggested replacement) would be the more responsible choice.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Machine Learning | `scikit-learn` |
| Data Handling | `pandas` |
| Visualization | `matplotlib`, `seaborn` |
| Environment | Jupyter Notebook / Google Colab |

---

## ▶️ How to Run

### Option 1 — Run on Google Colab (Recommended)

1. Open `Building_a_Linear_Regression_Model.ipynb` in Google Colab.
2. Run all notebook cells from top to bottom.
3. The Diabetes dataset loads automatically from `scikit-learn`. The Boston Housing dataset is included in this folder as `BostonHousing.csv`, so it's already available locally — the notebook's `wget` cell (originally used to fetch it) can be skipped if the file is already present.

### Option 2 — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/RoihansLab/Machine-Learning-Projects.git
cd Machine-Learning-Projects/07_Building_a_Linear_Regression_Model
```

**2. Install the required Python libraries**
```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
```

**3. Launch Jupyter Notebook**
```bash
jupyter notebook Building_a_Linear_Regression_Model.ipynb
```

Then open the notebook and run all cells sequentially.

---

## 🗂️ Project Structure

```text
07_Building_a_Linear_Regression_Model/
│
├── Building_a_Linear_Regression_Model.ipynb   → Full workflow for both datasets: loading, splitting,
│                                                  training, evaluation, and visualization
├── BostonHousing.csv                           → Boston Housing dataset, committed directly to the repo
└── README.md                                   → You are here
```

---

## 🙏 Acknowledgments & Credits

A heartfelt thank you to **Data Professor** on YouTube. The fundamental understanding and hands-on practice behind this project were made possible by his excellent, clearly explained tutorial. Supporting and crediting the creators who make quality Machine Learning education freely accessible is something I take seriously as part of this developer community.

📺 [Machine Learning in Python: Building a Linear Regression Model](https://www.youtube.com/watch?v=R15LjD8aCzc)

---

## 👤 Author

**Roihan Saputra**
*Aspiring Data Scientist / Machine Learning Engineer*
GitHub: [https://github.com/RoihansLab](https://github.com/RoihansLab)

Open to feedback, collaboration, or a conversation about this project — feel free to reach out via GitHub.
