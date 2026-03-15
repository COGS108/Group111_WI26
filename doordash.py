import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu


# -------------------------
# Load and clean data
# -------------------------

# Reading the sataset into a dataframe
doordash = pd.read_csv('data/00-raw/doordash.csv')

# Dropping irrelevant columns and renaming them
dd_drop_col = ['searched_zipcode', 'searched_lat', 'searched_lng', 'searched_address','searched_city', 'searched_metro',
               'city_slug', 'latitude', 'longitude', 'distance', 'loc_name', 'loc_number', 'url', 'address', 'cuisines',
               'delivery_fee_raw', 'delivery_fee',  'delivery_time_raw', 'service_fee_raw', 'service_fee', 'phone', 'review_count', 'RunDate']

doordash = doordash.drop(columns = dd_drop_col)
doordash = doordash.rename(columns = rename_cols)

# Removing outliers
d_Q1 = doordash['delivery_time'].quantile(0.25)
d_Q3 = doordash['delivery_time'].quantile(0.75)
d_IQR = d_Q3 - d_Q1

doordash = doordash[(doordash['delivery_time'] >= d_Q1 - 1.5 * d_IQR) & 
    (doordash['delivery_time'] <= d_Q3 + 1.5 * d_IQR)]

# Making subsets of the df based on state
ny_doordash = doordash[doordash['state'] == 'NY']

il_doordash = doordash[doordash['state'] == 'IL']

ca_doordash = doordash[doordash['state'] == 'CA']

# -------------------------
# Visualizations
# -------------------------

# First set of scatterplots in code cell 49
doordash['rating_jitter'] = doordash['rating'] + np.random.uniform(-0.1, 0.1, size=len(doordash))

states = ['NY','NJ','CA','IL','MA']

plt.figure(figsize=(18, 10))

for i, state in enumerate(states, 1):
    plt.subplot(2, 3, i)  # 2 rows, 3 columns
    subset3 = doordash[doordash['state'] == state]
    
    # Scatter plot
    sns.scatterplot(
        data=subset3,
        x='delivery_time',
        y='rating_jitter',
        alpha=0.6,
        color='skyblue'
    )
    
    # Regression line
    sns.regplot(
        data=subset3,
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

# Second set of scatterplots analyzing "rating" >= 3 in code cell 50
dd_pos = doordash[doordash['rating'] >= 3].copy()
dd_pos['rating_jitter'] = dd_pos['rating'] + np.random.uniform(-0.1, 0.1, size=len(dd_pos))

plt.figure(figsize=(18, 10))

for i, state in enumerate(states, 1):
    plt.subplot(2, 3, i)  # 2 rows, 3 columns
    subset4 = dd_pos[dd_pos['state'] == state]
    
    # Scatter plot
    sns.scatterplot(
        data=subset4,
        x='delivery_time',
        y='rating_jitter',
        alpha=0.6,
        color='skyblue'
    )
    
    # Regression line
    sns.regplot(
        data=subset4,
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