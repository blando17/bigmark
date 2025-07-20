#  Big Mart Sales Prediction

A machine learning project designed to predict product sales across various Big Mart outlets using historical sales and item/outlet characteristics.

---

##  Project Overview

Big Mart operates multiple retail outlets in various regions. This project aims to forecast the sales of products at different stores using machine learning, enabling better stock planning and demand forecasting.

We use regression modeling—specifically XGBoost—to predict the `Item_Outlet_Sales` target variable based on 11 input features.

---

##  Objectives

- Handle missing data effectively.
- Visualize the dataset to uncover patterns.
- Encode categorical variables properly.
- Train a predictive regression model.
- Evaluate model performance.
- Build a sales prediction system.

---

##  Technologies Used

| Tool/Library         | Purpose                         |
|----------------------|---------------------------------|
| Python               | Programming language            |
| Pandas               | Data manipulation               |
| NumPy                | Numerical operations            |
| Matplotlib & Seaborn | Data visualization              |
| scikit-learn         | Preprocessing & evaluation      |
| XGBoost              | Regression model implementation |

---

## 📂 Dataset Information

The dataset contains **8,523** rows and **12** columns.

### Important Features:
- `Item_Weight`, `Item_Visibility`, `Item_MRP`: Continuous features
- `Item_Fat_Content`, `Item_Type`, `Outlet_Type`, etc.: Categorical features
- `Item_Outlet_Sales`: Target variable (float)

---

##  Data Preprocessing

- **Missing Values**
  - `Item_Weight` → filled with **mean**
  - `Outlet_Size` → filled with **mode based on Outlet_Type**

- **Category Normalization**
  - Fixed inconsistencies in `Item_Fat_Content` (e.g., “low fat”, “LF” → “Low Fat”)

- **Label Encoding**
  - Applied `LabelEncoder` to convert categorical columns to numerical format:
    - `Item_Identifier`, `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, etc.

---

## Exploratory Data Analysis (EDA)

- Distribution plots for:
  - `Item_Weight`, `Item_Visibility`, `Item_MRP`, `Item_Outlet_Sales`
- Count plots for:
  - `Item_Fat_Content`, `Item_Type`, `Outlet_Size`, `Outlet_Establishment_Year`

---

##  Feature Engineering

- Features and target split:
  ```python
  X = big_mart_data.drop(columns='Item_Outlet_Sales')
  y = big_mart_data['Item_Outlet_Sales']
  ```

- Train-test split:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
  ```

---

##  Model Training

### Model: `XGBoost Regressor`

```python
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)
```

---

##  Model Evaluation

- **Training Data R² Score**: 0.8762
- **Testing Data R² Score**: 0.5017

Despite high performance on training data, some overfitting is observed, as seen in the test score.

---

##  Sales Prediction System

You can input the features manually to predict the sales:

```python
input_data = (662,17.50,0,0.016760,10,141.6180,9,1999,1,0,1)
prediction = regressor.predict([input_data])
print(prediction)
```

---

##  Conclusion

This project demonstrates the process of building a regression model for sales forecasting using real-world retail data. With appropriate preprocessing, visualization, and modeling techniques, we can build systems that aid in strategic business planning.

