# Linear Regression Tutorial in Python: Bangalore Rental Price Prediction (scikit-learn)

**Python machine learning tutorial** for beginners: train a **linear regression** model with **pandas** and **scikit-learn** to predict **Bangalore rental prices**. A short, runnable college lab for **supervised learning** fundamentals.

> **Lab 3** in the [Python learning path](https://github.com/saurabhahuja71/learning-path#2-python--data--apis) · Audience: beginners in ML · Time: ~45–90 minutes · Level: beginner → intermediate

## What is this project?

A single-script ML demo:

1. Load `bangalore_rentals.csv`  
2. Clean missing values  
3. **One-hot encode** categorical columns (`area`, `location`)  
4. Train/test split  
5. Fit `sklearn.linear_model.LinearRegression`  
6. Report **RMSE**, **MAE**, and a sample prediction  

**SEO keywords:** *linear regression python tutorial*, *scikit-learn beginner example*, *Bangalore house rent prediction*, *pandas one-hot encoding*, *RMSE MAE regression*, *machine learning college lab*.

## Prerequisites

- Python **3.9+**
- Ability to create a virtualenv
- Basic NumPy/pandas curiosity (Lab 4 notebooks go deeper)

## Quick start

```bash
git clone https://github.com/saurabhahuja71/sample-regression-model.git
cd sample-regression-model

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install pandas scikit-learn numpy

python bangalore_rental_regression.py
```

Example output shape:

```text
RMSE: 5141.68
MAE: 4426.51
Predicted rent for sample: 23810.61
```

(Exact numbers depend on the CSV and sklearn version.)

## Project structure

```text
sample-regression-model/
├── bangalore_rental_regression.py  # training script
├── bangalore_rentals.csv           # dataset
└── README.md
```

## What you will learn

| Concept | In this lab |
|---------|-------------|
| Supervised learning | Predict continuous `rent` |
| Feature prep | Median fill, one-hot encoding |
| Train/test split | Generalization check |
| Metrics | RMSE, MAE |
| scikit-learn API | `fit` / `predict` |

## How the pipeline works

```text
CSV → clean → encode categoricals → numeric features
    → train_test_split → LinearRegression.fit
    → predict → RMSE / MAE
```

Key libraries: **pandas**, **scikit-learn** (`OneHotEncoder`, `LinearRegression`, metrics).

## Lab exercises

1. Print feature importances via model coefficients (largest absolute weights).  
2. Try `Ridge` or `RandomForestRegressor` and compare RMSE.  
3. Add a simple CLI: `python bangalore_rental_regression.py --test-size 0.3`.  
4. Plot actual vs predicted with matplotlib.  
5. Continue with multi-day notebooks in [datascienceandmachinelearning](https://github.com/saurabhahuja71/datascienceandmachinelearning).

## Learning path

| # | Lab | Focus |
|---|-----|--------|
| 1–2 | FastAPI / Flask apps | Software side of Python |
| **3 (this)** | sample-regression-model | First ML model end-to-end |
| 4 | [datascienceandmachinelearning](https://github.com/saurabhahuja71/datascienceandmachinelearning) | NumPy, pandas, viz notebooks |

Hub: [learning-path](https://github.com/saurabhahuja71/learning-path)

## FAQ — Regression ML for students

**Is linear regression “real ML”?**  
Yes — it is the right first supervised model before neural nets.

**Why is error still thousands of rupees?**  
Rent is noisy; linear models are limited. The goal is the **workflow**, not Kaggle rank.

**Do I need a GPU?**  
No.

## Topics / SEO tags

`python` `machine-learning` `linear-regression` `scikit-learn` `pandas` `bangalore` `rent-prediction` `tutorial` `beginner` `data-science` `rmse` `college`

## Author

[Saurabh Ahuja](https://github.com/saurabhahuja71) · [learning-path](https://github.com/saurabhahuja71/learning-path)

## License

Educational sample. Dataset is for learning; validate before any real pricing use.
