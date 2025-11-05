# Time-Series Forecasting with Machine Learning, Linear Programming, and Stochastic Programming

**Course:** SEP 6DA3 – Data Analytics and Big Data  
**Institution:** McMaster University  
**Semester:** Fall 2025  
**Team:** Group 6  
**Contributors:** Andi Dong, Zhiyu Hu, Foram Brahmbhatt, Jiarui Yang, Linghe Shen, Shannon Chen  
**Instructor:** Prof. Pedro Tondo  

---

## ⚡ Project Overview
This project integrates **Machine Learning (ML)** and **Optimization (LP/SP)** for energy demand forecasting and cost minimization using PJM’s hourly electricity consumption dataset.

We implemented two models — **Random Forest (RF)** and **Transformer** — to predict short-term electricity demand, compared their forecasting accuracy, and applied optimization techniques to determine the most cost-effective generator scheduling plan.  
The project also extends to **Stochastic Programming (SP)** to account for uncertainty in demand across multiple scenarios.

---

## 🧠 Objectives
- Forecast hourly electricity demand using **Random Forest** and **Transformer** models.  
- Utilize **GPU acceleration** to enhance Transformer training efficiency.  
- Compare forecast accuracy using **Mean Squared Error (MSE)**.  
- Optimize generator operation using **Linear Programming (LP)** to minimize total generation cost.  
- Extend to **Stochastic Programming (SP)** to handle uncertain demand scenarios.  
- Demonstrate how **AI/ML forecasting** integrates with **optimization models** in real-world energy systems.

Dataset: [PJM Hourly Energy Consumption (Kaggle)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data)

---

## 🧩 Methodology

### 1. Data Preprocessing
- Parsed and sorted `datetime` column from **PJME_hourly.csv**.  
- Created lagged and rolling mean features to capture temporal dependencies.  
- Split dataset into **train (80%)** and **test (20%)** subsets.  

### 2. Forecasting Models
#### 🔹 Random Forest (RF)
- Trained on lag and rolling features.  
- Achieved **MSE = 1,785,677.71**, outperforming the Transformer in this dataset.  
- Provided stable, interpretable predictions with fast computation.

#### 🔹 Transformer (Deep Learning)
- Implemented a small **TransformerEncoder** for sequence-to-sequence forecasting.  
- Trained for ~50 epochs on GPU.  
- Achieved **MSE = 1,854,248.99**, slightly higher but showed smoother trend fitting.  

#### 🔹 Visualization
- **Actual vs Random Forest**  
- **Actual vs Transformer**  
- **Random Forest vs Transformer Predictions**  
  (Line plots and scatter overlays clearly show both models’ predictive alignment.)

---

## ⚙️ Energy Optimization: LP & SP

### 🔸 Linear Programming (LP)
Objective:  
Minimize total generation cost under capacity and demand constraints.

- 3 Generators defined with distinct cost and capacity values.  
- LP model formulated using **PuLP**.  
- Results:  
  - **Gen1** → Full capacity (cheapest cost per MW)  
  - **Gen2** → Partial capacity (mid-cost)  
  - **Gen3** → 0 output (highest cost per MW)

This demonstrates cost-driven dispatch logic where the solver fully utilizes low-cost units before engaging higher-cost generators.

---

### 🔸 Stochastic Programming (SP)
Introduced **three demand scenarios** to model uncertainty:  
| Scenario | Total Cost (USD) |
|-----------|------------------|
| Low Demand | 10,656.26 |
| Medium Demand | 10,926.47 |
| High Demand | 10,980.24 |

**Expected cost ≈ 10,854.32**  
The variation illustrates how demand uncertainty impacts operational cost and risk exposure.  

To mitigate variability, **demand-side management** and **energy storage** are recommended for smoothing peaks and reducing reliance on expensive units.

---

## 📊 Key Results
| Model | MSE | GPU Used | Notes |
|--------|-----|-----------|--------|
| Random Forest | **1,785,677.71** | No | Best overall accuracy |
| Transformer | 1,854,248.99 | Yes | Slightly higher error, but scalable for large datasets |

| Optimization | Description | Key Insight |
|---------------|--------------|--------------|
| Linear Programming | Cost minimization with capacity constraints | Prioritizes cheapest generators |
| Stochastic Programming | Cost optimization under uncertainty | Quantifies financial risk |

---

## 💡 Insights
- **Random Forest** delivered better predictive accuracy and efficiency for this dataset.  
- **Transformer** is promising for larger, more complex temporal data when GPU acceleration is available.  
- **LP + SP integration** bridges AI-driven forecasts with real-world operational decisions.  
- Handling **data uncertainty** through SP helps anticipate cost variability, improving grid reliability.  

---

## 🛠️ Technologies Used
- **Python 3.11**  
- **scikit-learn**, **PyTorch**, **NumPy**, **Pandas**, **Matplotlib**  
- **PuLP** for LP/SP optimization  
- **GPU acceleration (CUDA)** for Transformer training  

---

## 🧩 Repository Structure
