# Machine-Learning-Practical-Application
Github repository for the Practical Application 

# Project Overview
The goal of this project is to distinguish between customers who accepted a driving coupon versus those that did not by using visualizations and probability distributions

# Data Information

* Data Folder: **data** - This folder is created to store the data file (coupons.csv)

* Data Set - Data set used in the project is in the **coupons.csv** file 
  
* Code File - Python Code is stored in the **Module_5_Activity1.ipynb** file

# Data Read to Create DataFrame in Python

```python
# Python Code to  read the coupons.csv stored in the data Folder using various Pyhon Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv('data/coupons.csv')

# Python Code to print the column names and their data types
print(data.info())


```
# Data Set Column Details

1.  `destination`: Where the driver is going (e.g., home, work).
2.  `passanger`: Who's in the car with the driver (e.g., alone, partner, kids).
3.  `weather`: Current weather conditions (e.g., sunny, rainy).
4.  `temperature`: Temperature outside (e.g., 30F, 55F).
5.  `time`: Time of day (e.g., 10AM, 2PM).
6.  `coupon`: Type of coupon (e.g., restaurant, coffee).
7.  `expiration`: How long the coupon is valid for (e.g., 2 hours, 1 day).
8.  `gender`: Gender of the driver (e.g., female, male).
9.  `age`: Age range of the driver (e.g., below 21, 26-30).
10. `maritalStatus`: Marital status of the driver (e.g., single, married).
11. `has_children`: Whether the driver has children (e.g., yes, no).
12. `education`: Educational level of the driver (e.g., high school, graduate degree).
13. `occupation`: The driver's profession (e.g., architecture, business).
14. `income`: Annual income range of the driver (e.g., below $12500, $25000-$37499).
15. `Bar`: How often the driver goes to a bar (e.g., 0, 1-3 times).
16. `CoffeeHouse`: How often the driver goes to a coffee house (e.g., 0, 4-8 times).
17. `CarryAway`: How often the driver orders take-away food (e.g., 0, less than 1 time).
18. `RestaurantLessThan20`: How often the driver eats at a less expensive restaurant (e.g., 0, 1-3 times).
19. `Restaurant20To50`: How often the driver eats at a more expensive restaurant (e.g., 0, greater than 8 times).
20. `toCoupon_GEQ5min`: Whether the coupon is at least 5 minutes away.
21.   `toCoupon_GEQ15min`: Whether the coupon is at least 15 minutes away.
22. `toCoupon_GEQ25min`: Whether the coupon is at least 25 minutes away.
23. `direction_same`: Whether the coupon is in the same direction as the driver's destination.
24. `Y`: Whether the driver accepted the coupon (0 = No, 1 = Yes).

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

# Dataset Cleanup

**The column car has a very high proportion of missing values (around 99%), rendering it practically unusable for our analysis. So, I will exclude this column from further analysis**
```python
# Drop the 'car' column
if 'car' in data.columns:
    data.drop('car', axis=1, inplace=True)
    print("Column 'car' deleted successfully.")
else:
    print("Column 'car' not found. Skipping drop.")
```
**Drop the rows with the null values from the columns CoffeeHouse, Restaurant20To50, CarryAway, RestaurantLessThan20, and Bar Columns. The percentage of the null values for these Columns is less than 2% so it will not have any significant impact**
```python
data.dropna(subset=['CoffeeHouse', 'Restaurant20To50', 'CarryAway', 'RestaurantLessThan20', 'Bar'], inplace=True)
```
**Data Cleanup to The columns Bar, CoffeeHouse, CarryAway, RestaurantLessThan20, and Restaurant20To50 Columns by converting them in numeric type in order to carry out the necessary computations later in the Analysis**
```python
# Transformation the data to Numeric Values
def transform_frequency(freq_str):
  if freq_str == 'never':
    return 0.0
  elif freq_str == 'less1':
    return 0.5
  elif freq_str == '1~3':
    return 2.0
  elif freq_str == '4~8':
    return 6.0
  elif freq_str == 'gt8':
    return 10.0
  elif pd.isna(freq_str): # Check for np.nan using pd.isna()
    return np.nan
  else:
    raise ValueError(f"Unexpected frequency value: {freq_str}")

# Apply the transformation to the relevant columns
columns_to_transform = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']
for col in columns_to_transform:
  data[col] = data[col].astype(str).apply(transform_frequency)
```
# Statstical Analysis of the Cleanup Dataset
```python
# Print descriptive statistics for all numeric columns

print("\nDescriptive Statistics for Numeric Columns:\n")
print(data.describe().to_markdown(numalign="left", stralign="left"))

# For all object type columns, print the number of distinct values and the most frequent value

print("\nObject Column Summaries:\n")
for col in data.select_dtypes(include='object'):
    print(f"Column: {col}")
    print(f"  Number of distinct values: {data[col].nunique()}")
    print(f"  Most frequent value: {data[col].mode()[0]}\n")
```
**Analysis of Descriptive Statistics:**

* Count: The number of observations (rows) for each column. It shows how much data we have for each feature.
* Mean: The average value for each column. It gives an overall idea of the central tendency of the data for each feature.
* Standard Deviation (std): How much the data varies around the mean. A higher standard deviation means more variability.
* Minimum (min): The smallest value in each column.
* 25th Percentile (25%): The value below which 25% of the data falls. This gives us a sense of the lower range of the data.
* 50th Percentile (50% or median): The value below which 50% of the data falls. It is the middle value when the data is sorted.
* 75th Percentile (75%): The value below which 75% of the data falls. It gives us a sense of the upper range of the data.
* Maximum (max): The largest value in each column.

From the descriptive statstics we can see:
- For the columns Bar, CoffeeHouse, CarryAway, RestaurantLessThan20, and Restaurant20To50, the mean, median, and other percentile values suggest how often customers generally visit these places.
- The standard deviation for these columns indicates the level of variability in the frequency of visits to these places.
- For instance, the large difference between mean and median in Restaurant20To50 may suggest skewed distribution with potential outliers.

** Correlation Calculation between all the numerical columns of the Dataset using Heatmap**
```python
plt.figure(figsize=(12, 8))

# Select only numerical columns for correlation calculation
numerical_data = data.select_dtypes(include=['number'])

sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```

