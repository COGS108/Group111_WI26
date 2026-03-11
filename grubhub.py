import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu


# -------------------------
# Load and clean data
# -------------------------


grubhub = pd.read_csv('data/00-raw/grubhub.csv')
grubhub_drop_columns = ['searched_zipcode', 'searched_lat', 'searched_lng', 'searched_address','searched_metro', 'is_gh', 'latitude', 'longitude', 'loc_name', 'loc_number', 'url', 'address', 'phone','RunDate', 'restaurant_tags', 'delivery_fee_raw', 'delivery_time_raw']
grubhub = grubhub.drop(columns = grubhub_drop_columns)
grubhub.isnull().sum()
rename_cols = {'searched_state': 'state',
                'delivery_time': 'delivery_time',
                'review_rating': 'rating',}

grubhub = grubhub.rename(columns = rename_cols)
ny_grubhub = grubhub[grubhub['state'] == 'NY']
il_grubhub = grubhub[grubhub['state'] == 'IL']
ca_grubhub = grubhub[grubhub['state'] == 'CA']
ma_grubhub = grubhub[grubhub['state'] == 'MA']
nj_grubhub = grubhub[grubhub['state'] == 'NJ']
categorical_cols = ['state','searched_city','cuisines','delivery_type']
grubhub[categorical_cols] = grubhub[categorical_cols].astype('category')

# Removing any potential outliers for delivery time
Q1 = grubhub['delivery_time'].quantile(0.25)
Q3 = grubhub['delivery_time'].quantile(0.75)
IQR = Q3 - Q1

grubhub = grubhub[(grubhub['delivery_time'] >= Q1 - 1.5*IQR) &
                  (grubhub['delivery_time'] <= Q3 + 1.5*IQR)]


# -------------------------
# Visualizations
# -------------------------


# Grubhub Plots
plt.figure(figsize=(8,5))

sns.histplot(
    data=grubhub,
    x='delivery_time',
    bins=20,            
    kde=True
)

plt.title("Distribution of Delivery Time")# Create bins
grubhub['delivery_bin'] = pd.cut(grubhub['delivery_time'], bins=6)

plt.figure(figsize=(10,5));
# Grubhub Plots
# Add small jitter to rating if discrete (1-5)
grubhub['rating_jitter'] = grubhub['rating'] + np.random.uniform(-0.1, 0.1, size=len(grubhub))

# Scatter plot with regression lines per state
sns.lmplot(
    data=grubhub,
    x='delivery_time',
    y='rating_jitter',
    hue='state',           # updated column name
    height=6,
    aspect=1.5,
    scatter_kws={'alpha':0.6},
    line_kws={'linewidth':2}
)

plt.title("Delivery Time vs Rating with Regression Lines by State")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Rating")
plt.show()

# Add small jitter to ratings (1–5)
grubhub['rating_jitter'] = grubhub['rating'] + np.random.uniform(-0.1, 0.1, size=len(grubhub))
# List of states you want separate plots for
states = ['NY','NJ','CA','IL','MA']
plt.figure(figsize=(18, 10))
for i, state in enumerate(states, 1):
    plt.subplot(2, 3, i)  # 2 rows, 3 columns
    subset = grubhub[grubhub['state'] == state]    
    # Scatter plot
    sns.scatterplot(
        data=subset,
        x='delivery_time',
        y='rating_jitter',
        alpha=0.6,
        color='skyblue'
    )
    # Regression line
    sns.regplot(
        data=subset,
        x='delivery_time',
        y='rating_jitter',
        scatter=False,
        color='darkred',
        line_kws={'linewidth':2}
    )
plt.title(f"State: {state}")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

#Very little correlation between the variables
sns.heatmap(  grubhub[['rating','distance','delivery_fee','delivery_time','service_fee','review_count']].corr().round(2),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Matrix")
plt.show()


# -------------------------
# Filtered analysis
# -------------------------


#Grubhub Plot
grubhub_test = grubhub[grubhub['rating'] > 0]#removing ratings with 0 to test for any possible relationship
grubhub_test = grubhub[grubhub['rating'] > 0]
subset = grubhub_test.copy()

subset['rating_jitter'] = subset['rating'] + np.random.uniform(-0.1, 0.1, size=len(subset))
# grubhub_test['rating_jitter'] = grubhub_test['rating'] + np.random.uniform(-0.1, 0.1, size=len(grubhub_test))
states = ['NY','NJ','CA','IL','MA']
plt.figure(figsize=(18,10))
for i, state in enumerate(states, 1):
    plt.subplot(2,3,i)
    subset = grubhub_test[grubhub_test['state'] == state]
    # Scatter
    sns.scatterplot(
        data=subset,
        x='delivery_time',
        y='rating_jitter',
        alpha=0.6,
        color='skyblue'
    )
    # Regression line
    sns.regplot(
        data=subset,
        x='delivery_time',
        y='rating_jitter',
        scatter=False,
        color='darkred',
        line_kws={'linewidth':2}
    )
plt.title(f"State: {state}")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

# --- 1. Add jitter to ratings ---
grubhub['rating_jitter'] = grubhub['rating'] + np.random.uniform(-0.1, 0.1, size=len(grubhub))
# --- 2. Filter for New York state ---
grubhub_ny = grubhub[grubhub['state'] == 'NY']
# --- 3. select top cities to avoid clutter ---
top_cities = grubhub_ny['searched_city'].value_counts().nlargest(6).index
grubhub_ny_top = grubhub_ny[grubhub_ny['searched_city'].isin(top_cities)]
# --- 4. Scatter plot with regression lines per city ---
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=grubhub_ny_top,
    x='delivery_time',
    y='rating_jitter',
    hue='searched_city',
    palette='tab10',
    alpha=0.6,
    legend=False
)
# Regression lines per city
for city in top_cities:
    subset = grubhub_ny_top[grubhub_ny_top['searched_city'] == city]
    sns.regplot(
        data=subset,
        x='delivery_time',
        y='rating_jitter',
        scatter=False,
        label=f"{city} trend"
    )
plt.title("Delivery Time vs Rating for Top Cities in New York")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Rating")
plt.legend(title="City", bbox_to_anchor=(1.05,1), loc="upper left")
plt.show()

#Grubhub Plot - Isolating city
# --- 1. Add jitter to ratings ---
grubhub['rating_jitter'] = grubhub['rating'] + np.random.uniform(-0.1, 0.1, size=len(grubhub))
# --- 2. Choose the city you want to focus on ---
city_to_plot = 'Cambridge'  # replace with the exact city name in your dataset
grubhub_city = grubhub[grubhub['searched_city'] == city_to_plot]
# --- 3. Create scatter plot with regression line ---
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=grubhub_city,
    x='delivery_time',
    y='rating_jitter',
    alpha=0.6,
    color='skyblue'
)
sns.regplot(
    data=grubhub_city,
    x='delivery_time',
    y='rating_jitter',
    scatter=False,#removing ratings with 0 to test for any possible relationship
    color='darkred',
    line_kws={'linewidth':2}
)
plt.title(f"Delivery Time vs Rating in {city_to_plot}")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Rating")
plt.show()

grubhub_test = grubhub[grubhub['rating'] > 0]
#Very little correlation between the variables
sns.heatmap(
grubhub_test[['rating','distance','delivery_fee','delivery_time','service_fee','review_count']].corr().round(2),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Matrix")
plt.show()

# Remove any rating less than 3 ratings
grubhub_test_2 = grubhub[grubhub['rating'] > 3].copy()
#Grubhub Plot
sns.histplot(grubhub_test_2['rating'], bins=10, kde=True)
plt.title("Distribution of Ratings")
plt.show()

# Add jitter to rating
grubhub_test_2['rating_jitter'] = grubhub_test_2['rating'] + np.random.uniform(-0.1, 0.1, size=len(grubhub_test_2))

# States to plot
states = ['NY','NJ','CA','IL','MA']

plt.figure(figsize=(18,10))
for i, state in enumerate(states, 1):
    plt.subplot(2,3,i)
    subset = grubhub_test_2[grubhub_test_2['state'] == state].copy()
    # Scatter
    sns.scatterplot(
        data=subset,
        x='delivery_time',
        y='rating_jitter',
        alpha=0.6,
        color='skyblue'
    )
    # Regression line
    sns.regplot(
        data=subset,
        x='delivery_time',
        y='rating_jitter',
        scatter=False,
        color='darkred',
        line_kws={'linewidth':2}
    )
plt.title(f"State: {state}")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

#Grubhub Plot
# Heatmap of correlations
plt.figure(figsize=(8,6))
sns.heatmap(
grubhub_test_2[['rating','distance','delivery_fee','delivery_time','service_fee','review_count']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Matrix (Ratings > 3)")
plt.show()


# -------------------------
# Statistical Tests
# -------------------------


grubhub_test_2[['delivery_time','rating']].corr()
#Multivariable OLS test to see if it cna bypass normality and skewedness
formula = 'rating ~ delivery_time + delivery_fee + service_fee + distance + C(state)'
model = ols(formula, data=grubhub_test_2).fit()
print(model.summary())

#Testing Mann Whitney U-test
group1 = grubhub[grubhub['state'] == 'NY']['rating']
group2 = grubhub[grubhub['state'] == 'NJ']['rating']
stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
print(f"Mann-Whitney U statistic: {stat:.2f}")
print(f"P-value: {p:.4e}")
if p < 0.05:
    print("Result: Significant difference between NY and NJ ratings")
else:
    print("Result: No significant difference between NY and NJ ratings")

# Checking for Effect Size
group1 = grubhub[grubhub['state']=='NY']['rating']
group2 = grubhub[grubhub['state']=='NJ']['rating']
stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
n1 = len(group1)
n2 = len(group2)
N = n1 + n2
mean_U = n1*n2 / 2
std_U = np.sqrt(n1*n2*(n1+n2+1)/12)
Z = (stat - mean_U)/std_U
r = Z / np.sqrt(N)
print(f"Effect size r: {r:.3f}")