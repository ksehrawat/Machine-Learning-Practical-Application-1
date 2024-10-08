# Machine-Learning-Practical-Application-1
Github repository for the Practical Application 1 

# Project Overview
The goal of this project is to distinguish between customers who accepted a driving coupon versus those that did not by using visualizations and probability distributions

# Data Information

* Data Folder: **data** - This folder is created to store the data file (coupons.csv)

* Data Set - Data set used in the project is in the **coupons.csv** file 
  
* Code File - Python Code is stored in the **Module_5_Activity1.ipynb** file

# Analysis

```python
# Python Code to  read the coupons.csv stored in the data Folder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv('data/coupons.csv')

# Print the column names and their data types
print(data.info())
