# Machine-Learning-Practical-Application
Github repository for the Practical Application 

# Project Overview
The goal of this project is to distinguish between customers who accepted a driving coupon versus those that did not by using visualizations and probability distributions

# Data Information

* Data Folder: **data** - This folder is created to store the data file (coupons.csv)

* Data Set - Data set used in the project is in the **coupons.csv** file 
  
* Code File - Python Code is stored in the **Module_5_Activity1.ipynb** file

# Initial Data Analysis

```python
# Python Code to  read the coupons.csv stored in the data Folder using various Pyhon Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv('data/coupons.csv')

# Python Code to print the column names and their data types
print(data.info())

# Print the data frame first 5 rows in a structured format
print(data.head().to_markdown(index=False, numalign="left", stralign="left"))

```
# Dataset Analysis for missing or problematic data

```python
# Print the count and percentage of missing values for each column
missing_values = data.isnull().sum()
missing_percent = (missing_values / len(data)) * 100
print("Missing Values:\n")
print(pd.concat([missing_values, missing_percent], axis=1, keys=['Count', 'Percentage']).sort_values(by='Count', ascending=False).to_markdown(numalign="left", stralign="left"))
```

Analysis of Missing Values:
The dataset has missing values in several columns. Here's a breakdown:
- The 'car' column has 12576 missing values, which represents 99.15% of the total data.
- The 'Bar' column has 107 missing values, which represents 0.84% of the total data.
- The 'CoffeeHouse' column has 217 missing values, which represents 1.71% of the total data.
- The 'CarryAway' column has 151 missing values, which represents 1.19% of the total data.
- The 'RestaurantLessThan20' column has 130 missing values, which represents 1.02% of the total data.
- The 'Restaurant20To50' column has 189 missing values, which represents 1.49% of the total data.
