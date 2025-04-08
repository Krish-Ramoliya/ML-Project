# ML-Project

# 🛍️ Mall Customer Segmentation

This project applies **K-Means Clustering** to segment mall customers based on their **Annual Income** and **Spending Score**. The goal is to identify different types of customers to help the mall improve marketing strategies and customer experiences.

---

## 📁 Dataset

The dataset used is the [Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial) from Kaggle. It contains the following fields:

- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

---

## 🧰 Tech Stack

- **Python 3.x**
- **Libraries:**
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - (optional) `plotly` for interactive plots

---

## 📊 Features

- Perform Exploratory Data Analysis (EDA)
- Use Elbow Method to find optimal number of clusters
- Apply **K-Means Clustering**
- Visualize customer clusters
- Identify business insights from segments

---

## 📂 Project Structure

ML-Project/
├── data/                  # Includes Mall_Customers.csv
├── notebooks/             # For Jupyter Notebooks
├── src/                   # Python scripts
│   └── __init__.py
├── models/                # Will hold trained models
├── docs/                  # For documentation
├── README.md              # Project overview
├── requirements.txt       # Required libraries
├── LICENSE                # MIT License


## 📈 Output Sample

- After clustering, you will get clear customer segments such as:
- High income, high spending (Premium customers)
- Low income, low spending
- High income, low spending (Potential targets)
- Low income, high spending (Value seekers)


## 📦 Installation
1. Clone the repository:

```bash
git clone https://github.com/Krish-Ramoliya/ML-Project.git
cd ML-Project

2. Install dependencies:
pip install -r requirements.txt
