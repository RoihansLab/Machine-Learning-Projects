# 📦 End-to-End Retail Demand Forecasting

An end-to-end Machine Learning pipeline designed to forecast retail inventory demand using historical sales data and transaction parameters. This project covers exploratory data analysis (EDA), feature engineering, model optimization via hyperparameter tuning, and deployment as an interactive web application.

---

## 🚀 Project Overview
Accurate demand forecasting is critical in retail to prevent stockouts and overstocking. This project utilizes an **XGBoost Regressor** trained on historical transactional attributes (such as Pricing, Competitor Pricing, Discounts, Inventory Levels, Promotions, and Categories) to predict product demand precisely.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python 3.10
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning Model:** XGBoost (`XGBRegressor`)
- **Optimization & Tuning:** Randomized Search Cross-Validation (`RandomizedSearchCV`)
- **Data Visualization:** Matplotlib, Seaborn
- **Deployment Framework:** Streamlit
- **Serialization:** Pickle

---

## 📂 Project Structure
```text
10_End_to_End_Retail_Demand_Forecasting/
│
├── 10_analisis.ipynb                   # Exploratory Data Analysis (EDA) & Feature Engineering
├── 10_Machine_Learning.ipynb           # Model building, training, tuning, and evaluation
├── app.py                              # Streamlit web application for real-time inference
├── demand_forecasting.csv              # Raw retail dataset (76,000 rows)
├── preprocessed_demand_forecasting.csv # Processed dataset after feature engineering
├── xgboost_demand_model_RSCV.pkl       # Trained XGBoost model artifact
├── label_encoder.pkl                   # Serialized LabelEncoder for categorical features
├── pyrightconfig.json                  # Python type checking configuration
└── README.md                           # Project documentation
```

---

## 📊 Model Performance & Evaluation
To ensure the model is robust and generalizable, it was evaluated using a comprehensive suite of business and technical metrics on the test data:

- **R² Score:** 0.428 (Indicates stable pattern recognition without severe overfitting compared to the train set's 0.504)
- **MAE (Mean Absolute Error):** 26.99 units (Average deviation per prediction)
- **RMSE (Root Mean Squared Error):** 35.55 units (Penalizes extreme outliers)
- **MAPE (Mean Absolute Percentage Error):** 38.88% (Relative accuracy margin)

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RoihansLab/Machine-Learning-Projects.git
   ```

2. **Navigate to the project folder:**
   ```bash
   cd Machine-Learning-Projects/10_End_to_End_Retail_Demand_Forecasting
   ```

3. **Install dependencies:**
   Make sure you have the required libraries installed in your environment:
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit
   ```

4. **Run the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

---

## 👤 Author
**Roihan Saputra**  
*D4 Informatics Engineering Student at Politeknik Negeri Semarang (Polines)*  
[GitHub Profile](https://github.com/RoihansLab) | [LinkedIn Profile](#)
