# Project Overview
The goal of this project is to distinguish between customers who accepted a driving coupon versus those that did not by using visualizations and probability distributions using Python Pandas

# Data Information

* Data Folder: **data** - This folder is created to store the data file (coupons.csv)

* Data Set - Data set used in the project is in the **coupons.csv** file 
  
* Code File - Python Code is stored in the **Module_5_Activity1.ipynb** file

# Create DataFrame in Python for the Data Set

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

**Correlation Calculation between all the numerical columns of the Dataset using Heatmap**
```python
plt.figure(figsize=(12, 8))

# Select only numerical columns for correlation calculation
numerical_data = data.select_dtypes(include=['number'])

sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```
![Heatmap](https://github.com/user-attachments/assets/31953a78-cda4-46fa-b05b-9e05b2fca9ed)

**Key observations from the heatmap:**
- There is a moderate positive correlation between RestaurantLessThan20 and CarryAway (0.36). This suggests that customers who frequently eat at inexpensive restaurants also tend to order takeout more often.
- There's a weak positive correlation between Restaurant20To50 and RestaurantLessThan20 (0.13).
- There's a very weak positive correlation between Bar and CoffeeHouse.
- The correlation between Y and other features are weak, suggesting no strong linear relationship.

# DataSet Analysis

**Overall Acceptance rate of the Coupons**
```python
overall_acceptance_rate = data['Y'].mean() * 100
print(f"Overall Acceptance Rate: {overall_acceptance_rate:.2f}%")
```
- The overall acceptance rate for all the Coupons is 56.93%

**Acceptance rate by different Coupon Types**
```python
coupon_acceptance_rates = data.groupby('coupon')['Y'].mean() * 100
coupon_acceptance_rates = coupon_acceptance_rates.sort_values(ascending=False)
print(coupon_acceptance_rates.to_markdown(numalign="left", stralign="left"))
```
| Coupon Type           | Acceptance Rate|
|:----------------------|:--------|
| Carry out & Take away | 73.7719 |
| Restaurant(<20)       | 70.9009 |
| Coffee House          | 49.6331 |
| Restaurant(20-50)     | 44.6013 |
| Bar                   | 41.1918 |


**Calcu;late the difference between Acceptance rate by different Coupon Types and the overall Coupon Acceptance Rate**
```python
for coupon, acceptance_rate in coupon_acceptance_rates.items():
  difference = acceptance_rate - overall_acceptance_rate
  print(f"Coupon: {coupon}, Acceptance Rate: {acceptance_rate:.2f}, Difference from Overall Acceptance Rate: {difference:.2f}")
```
- Coupon: Carry out & Take away, Acceptance Rate: 73.77, Difference from Overall Acceptance Rate: 16.84
- Coupon: Restaurant(<20), Acceptance Rate: 70.90, Difference from Overall Acceptance Rate: 13.97
- Coupon: Coffee House, Acceptance Rate: 49.63, Difference from Overall Acceptance Rate: -7.30
- Coupon: Restaurant(20-50), Acceptance Rate: 44.60, Difference from Overall Acceptance Rate: -12.33
- Coupon: Bar, Acceptance Rate: 41.19, Difference from Overall Acceptance Rate: -15.74

**Summary**

- The overall average coupon acceptance rate is 57%.
- 'Carry out & Take away' coupons have the highest acceptance rate at 74%, which is 17% higher than the overall average.
- 'Bar' coupons have the lowest acceptance rate at 41%, which is 16% lower than the overall average.

**What proportion of the total observations chose to accept the coupon?**
```python
# Calculate the proportion of observations who accepted the coupon
acceptance_rate = data['Y'].mean()

# Print the acceptance rate
print(f"The proportion of observations who accepted the coupon is: {acceptance_rate * 100:.2f}%")

# Create a pie chart to visualize the acceptance rate
plt.figure(figsize=(6, 6))
labels = ['Accepted', 'Not Accepted']
sizes = [acceptance_rate, 1 - acceptance_rate]
colors = ['skyblue', 'lightcoral']
explode = (0.1, 0)  # Explode the 'Accepted' slice for emphasis

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Coupon Acceptance Rate')
plt.show()
```
![download](https://github.com/user-attachments/assets/5d5dc272-695f-4ea6-9794-4e34607b822f)

**The proportion of observations who accepted the coupon is: 56.93%**

# Data Visualization

**Bar plot to visualize the coupon column**
```python
coupon_counts = data['coupon'].value_counts()

plt.figure(figsize=(10, 6))
bars = plt.bar(coupon_counts.index, coupon_counts.values)
plt.xlabel('Coupon Type')
plt.ylabel('Frequency')
plt.title('Frequency of Different Coupon Types')

# Add frequency labels to the bars
for bar in bars:
  yval = bar.get_height()
  plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.show()
```
![download (2)](https://github.com/user-attachments/assets/8a896371-d0b1-402f-8e2c-064cc3e75f4b)

**Histogram to visualize the temperature column**
```python
plt.figure(figsize=(10, 6))
sns.histplot(data['temperature'], bins=10, color='skyblue', edgecolor='black', kde = 'true')
plt.title('Distribution of Temperature')
plt.xlabel('Temperature')
plt.ylabel('Frequency')

# Annotate the bars with their values
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                     textcoords='offset points')

plt.show()

```
![download (3)](https://github.com/user-attachments/assets/44c5ad8f-94f2-4f4b-8847-59d7711587c7)


# Bar  Coupon Data Analysis

**Create a new DataFrame that contains just the Bar coupons**
```python
# Filter the dataframe to only include rows where the `coupon` column is equal to 'Bar'
bar_coupons_data = data[data['coupon'] == 'Bar'].copy()

# Print the first 5 rows of `bar_coupons_data`
print(bar_coupons_data.head().to_markdown(index=False, numalign="left", stralign="left"))
```


**Proportion of bar coupons were accepted?**
```python
# Count the number of accepted Bar coupons (Y = 1) using data frame bar_coupons_data
accepted_bar_coupons_count = (bar_coupons_data['Y'] == 1).sum()

# Print the Total Bar Coupons
print(f"Total Bar Coupons: {len(bar_coupons_data)}")

# Print the Total Accepted Bar Coupons
print(f"Total Accepted Bar Coupons: {accepted_bar_coupons_count}")

# Calculate the proportion of accepted coupons
Bar_coupon_proportion_accepted = accepted_bar_coupons_count / len(bar_coupons_data)

# Print the proportion as a percentage, rounded to two decimal places
print(f"Proportion of Accepted Bar Coupons: {Bar_coupon_proportion_accepted * 100:.2f}%")


# Create a pie chart with different colors to visualize the proportion of accepted Bar coupons
plt.figure(figsize=(8, 8))
labels = ['Accepted', 'Not Accepted']
sizes = [Bar_coupon_proportion_accepted, 1 - Bar_coupon_proportion_accepted]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)  # Explode the 'Accepted' slice for emphasis

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Bar Coupon Acceptance Rate')
plt.show()
```
![download (4)](https://github.com/user-attachments/assets/2f3ac729-635f-4ae9-8a31-1751548e3712)

- Total Bar Coupons: 1913
- Total Accepted Bar Coupons: 788
- Proportion of Accepted Bar Coupons: 41.19%


**Acceptance rate between those who went to a bar 3 or fewer times a month to those who went more**
```python
# Create a new column `bar_visit_frequency` based on the `Bar` column
bar_coupons = data[data['coupon'] == 'Bar'].copy()
bar_coupons['bar_visit_frequency'] = bar_coupons['Bar'].apply(lambda x: '3 or fewer' if x in [0, 0.5, 2] else 'more than 3')

# Calculate acceptance rates for each group
acceptance_rates = bar_coupons.groupby('bar_visit_frequency')['Y'].mean().reset_index()
print (acceptance_rates.to_markdown(index=False))

plt.figure(figsize=(8, 6))
sns.barplot(x='bar_visit_frequency', y='Y', data=acceptance_rates, palette='viridis', hue = 'bar_visit_frequency' )
plt.title('Coupon Acceptance Rate by Bar Visit Frequency')
plt.xlabel('Bar Visit Frequency')
plt.ylabel('Acceptance Rate')
plt.ylim(0, 1)  # Set y-axis limits to 0 and 1 for better visualization
plt.show()
```
| bar_visit_frequency   |        Y |
|:----------------------|---------:|
| 3 or fewer            | 0.372674 |
| more than 3           | 0.761658 |


**Acceptance rate between drivers who go to a bar more than once a month and are over the age of 25 to the all others. Is there a difference?**
```python
bar_coupons['target_group'] = 'Other'
bar_coupons.loc[(bar_coupons['Bar'].apply(lambda x: x not in [0, 0.5]) ) & (bar_coupons['age'] > '25'), 'target_group'] = 'Bar > 1 & Age > 25'

# Calculate acceptance rates for each group
acceptance_rates_target = bar_coupons.groupby('target_group')['Y'].mean().reset_index()
print(acceptance_rates_target.to_markdown(index=False))

# Visualize the comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='target_group', y='Y', data=bar_coupons, palette='viridis', hue='target_group')
plt.title('Coupon Acceptance Rate by Target Group')
plt.xlabel('Target Group')
plt.ylabel('Acceptance Rate')
plt.ylim(0, 1)
plt.show()
```
| target_group       |        Y |
|:-------------------|---------:|
| Bar > 1 & Age > 25 | 0.682809 |
| Other              | 0.337333 |

![download (5)](https://github.com/user-attachments/assets/0c4b9859-6926-49e1-bce1-67b0f12f7815)

Key Observation:
The drivers in the 'Bar > 1 & Age > 25' group seem to have a slightly higher coupon acceptance rate compared to the 'Other' group.
This suggests that drivers who are both older and frequent bar visitors might be more likely to accept bar-related coupons.

Possible Reasons:
- Habitual bar visitors might be more receptive to bar-related deals and promotions.
- Older drivers in this group might be more inclined to take advantage of opportunities to try out new bars or revisit preferred ones.
- It's possible that age and bar-visiting frequency are correlated with lifestyle factors that influence coupon acceptance.


**Compare the acceptance rate between drivers who go to bars more than once a month and had passengers that were not a kid and had occupations other than farming, fishing, or forestry.**
```python
# Create a new column 'target_group' based on the criteria
bar_coupons['target_group'] = 'Other'
bar_coupons.loc[(bar_coupons['Bar'].apply(lambda x: x not in [0, 0.5])) &
                (bar_coupons['passanger'] != 'Kid(s)') &
                (~bar_coupons['occupation'].isin(['Farming Fishing & Forestry'])),
                'target_group'] = 'Bar > 1, No Kids, Not Farming/Fishing/Forestry'

# Calculate acceptance rates for each group
acceptance_rates_target = bar_coupons.groupby('target_group')['Y'].mean().reset_index()
print(acceptance_rates_target.to_markdown(index=False))

# Visualize the comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='target_group', y='Y', data=bar_coupons, palette='viridis', hue='target_group') # Changed hue parameter
plt.title('Coupon Acceptance Rate by Target Group')
plt.xlabel('Target Group')
plt.ylabel('Acceptance Rate')
plt.ylim(0, 1)
plt.show()
```
| target_group                                   |        Y |
|:-----------------------------------------------|---------:|
| Bar > 1, No Kids, Not Farming/Fishing/Forestry | 0.709434 |
| Other                                          | 0.297903 |

![download (6)](https://github.com/user-attachments/assets/ddf5934c-e86c-4602-8692-38764ed29139)

**Key Finding:** The acceptance rate for the target group 'Bar > 1, No Kids, Not Farming/Fishing/Forestry' appears to be slightly higher than the 'Other' group. This suggests that individuals who frequent bars, don't have children as passengers, and work in fields outside of farming, fishing, or forestry might be more likely to accept Bar coupons.

**Question 6:**

Compare the acceptance rates between those drivers who:
- go to bars more than once a month, had passengers that were not a kid, and were not widowed OR
- go to bars more than once a month and are under the age of 30 OR
- go to cheap restaurants more than 4 times a month and income is less than 50K.

```python
# Group 1: Go to bars more than once a month, no kids as passengers, and not widowed
bar_coupons.loc[(bar_coupons['Bar'] > 0.5) & (bar_coupons['passanger'] != 'Kid(s)') & (bar_coupons['maritalStatus'] != 'Widowed'), 'target_group'] = 'Group 1'

# Group 2: Go to bars more than once a month and are under 30
bar_coupons.loc[(bar_coupons['Bar'] > 0.5) & (bar_coupons['age'] < '30'), 'target_group'] = 'Group 2'

# Group 3: Go to cheap restaurants more than 4 times a month and income is less than 50K
# Use the | operator for element-wise OR and isin() for efficient multiple value checks
bar_coupons.loc[(bar_coupons['RestaurantLessThan20'] > 6.0) & (bar_coupons['income'].isin(['Less than $12500', '$12500 - $24999', '$25000 - $37499', '$37500 - $49999'])), 'target_group'] = 'Group 3'

# Calculate acceptance rates for each group
acceptance_rates_target = bar_coupons.groupby('target_group')['Y'].mean().reset_index()
print(acceptance_rates_target.to_markdown(index=False))

# Visualize the comparison
plt.figure(figsize=(8, 6))
sns.barplot(x='target_group', y='Y', data=bar_coupons, palette='viridis', hue='target_group')
plt.title('Coupon Acceptance Rate by Target Group')
plt.xlabel('Target Group')
plt.ylabel('Acceptance Rate')
plt.ylim(0, 1)
plt.show()
```
| target_group   |        Y |
|:---------------|---------:|
| Group 1        | 0.671362 |
| Group 2        | 0.717687 |
| Group 3        | 0.59375  |
| Other          | 0.287786 |

![download (7)](https://github.com/user-attachments/assets/f437c207-30af-472f-b3f6-215d287ca4df)

**Key Findings**
The plot suggests that drivers who frequently visit bars, especially those without children and who are younger or not widowed, are more likely to accept a bar coupon.  Drivers who mainly frequent cheap restaurants and have lower incomes seem less inclined to take advantage of a bar coupon offer.
