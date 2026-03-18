import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Visualizations with long lines of code are stored here in chronological order of use.

def state_lmplot(df):
    """
    Creates a combined linear model plot comparing delivery_time to rating for every state in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
         The dataframe that contains delivery and rating data.

    Returns
    -------
    None
         Displays a singular lmplot.
    """
    # Generating the line plot
    sns.lmplot(
    data=df,
    x='delivery_time',
    y='rating_jitter',
    hue='state',           
    height=6,
    aspect=1.5,
    scatter_kws={'alpha':0.6},
    line_kws={'linewidth':2}
    )

    # Adding appropriate and clear labels
    plt.title("Delivery Time vs Rating with Regression Lines by State")
    plt.xlabel("Delivery Time (minutes)")
    plt.ylabel("Rating")
    plt.show()



def state_scatterplot(df):
    """
    Creates a scatterplot comparing delivery_time to rating for every state in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
         The dataframe that contains delivery and rating data.

    Returns
    -------
    None
         Displays a 2x3 grid of scatter plots.
    """
    
    # adding the rating jitter column for visualization
    df = df.copy()
    
    df['rating_jitter'] = df['rating'] + np.random.uniform(-0.1, 0.1, size=len(df))
    
    states = ['NY','NJ','CA','IL','MA']

    plt.figure(figsize=(18, 10))

    for i, state in enumerate(states, 1):
        plt.subplot(2, 3, i)  # 2 rows, 3 columns
        
        df_subset = df[df['state'] == state]
        
        # Generating the scatterplot
        sns.scatterplot(
        data=df_subset,
        x='delivery_time',
        y='rating_jitter',
        alpha=0.6,
        color='skyblue'
        )
    
        # Generating the regression line
        sns.regplot(
        data=df_subset,
        x='delivery_time',
        y='rating_jitter',
        scatter=False,
        color='darkred',
        line_kws={'linewidth':2}
        )

        # Adding labels 
        plt.title(f"State: {state}")
        plt.xlabel("Delivery Time (minutes)")
        plt.ylabel("Rating")

    plt.tight_layout()
    plt.show()




def mwu_test(df1, df2, name1, name2):
    """ 
    Runs a Mann-Whitney U test on the two given datasets and calculates its effect size.

    Parameters
    ----------
    df1 : pd.DataFrame
         The first dataframe.     
    df2 : pd.DataFrame
         The second dataframe.
    name1 : str
         The title of df1.
    name2: str
         The title of df2.

    Returns
    -------
    None
         Prints the statistics and effect size.
    """

    # Running the test
    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')

    print(f"Mann-Whitney U statistic: {stat:.2f}")
    print(f"P-value: {p:.4e}")

    # Printing the results using the name1 and name2 parameters
    if p < 0.05:
        print(f"Result: Significant difference between {name1} and {name2} ratings")
    else:
        print(f"Result: No significant difference between {name1} and {name2} ratings")

    # Calculating the effect size of the results
    n1 = len(df1)
    n2 = len(df2)
    N = n1 + n2

    mean_U = n1*n2 / 2
    std_U = np.sqrt(n1*n2*(n1+n2+1)/12)
    Z = (stat - mean_U)/std_U

    r = Z / np.sqrt(N)

    # Printing the effect size
    print(f"Effect size r: {r:.3f}")
    


def compare_lineplot(df1, df2, df3):  # intended to be called on (foodhub, grubhub, doordash)
    """ 
    Creates a lineplot comparing each dataframe's delivery time to the associated ratings.


    Parameters
    ----------
    df1 : pd.DataFrame
         The first dataframe.
    df2 : pd.DataFrame
         The second dataframe.
    df3 : pd.DataFrame
         The third dataframe.

    Returns
    -------
    None
         Displays a lineplot.
    """

    # Setting up the visualization
    foodhub_small = df1[['delivery_time', 'rating']].copy()
    foodhub_small['platform'] = 'FoodHub'

    grubhub_small = df2[['delivery_time', 'rating']].copy()
    grubhub_small['platform'] = 'Grubhub'

    doordash_small = df3[['delivery_time', 'rating']].copy()
    doordash_small['platform'] = 'DoorDash'

    combined = pd.concat([foodhub_small, grubhub_small, doordash_small], ignore_index=True)
    combined = combined.dropna(subset=['delivery_time', 'rating'])

    combined['time_bin'] = pd.cut(combined['delivery_time'], bins=10)
    combined['bin_mid'] = combined['time_bin'].apply(lambda x: x.mid)

    # Generating the figure
    bin_means = (
        combined.groupby(['platform', 'bin_mid'], observed=True)['rating']
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10,6))
    sns.lineplot(data=bin_means, x='bin_mid', y='rating', hue='platform', marker='o')

    # Adding labels
    plt.xlabel("Delivery Time Bin Midpoint")
    plt.ylabel("Average Rating")
    plt.title("Average Customer Ratings by Delivery Time Across Platforms")
    plt.tight_layout()
    plt.show()




def compare_hexbin_plot(df1, df2, df3):  # intended to be called on (foodhub, grubhub, doordash)
    """ 
    Creates a hexagonal bin plot comparing each dataframe's delivery time to the associated ratings.

    Parameters
    ----------
    df1 : pd.DataFrame
         The first dataframe. 
    df2 : pd.DataFrame
         The second dataframe. 
    df3 : pd.DataFrame
         The third dataframe.

    Returns
    -------
    None
         Displays a hexbin plot.
    """

    # Creating the plots for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    datasets = [
        ("FoodHub", df1),
        ("Grubhub", df2),
        ("DoorDash", df3)
    ]

    # Generating the visualization for each dataset
    for ax, (name, df) in zip(axes, datasets):

        hb = ax.hexbin(
            df["delivery_time"],
            df["rating"],
            gridsize=25,
            cmap="Blues",
            mincnt=1
        )

        ax.set_title(name)
        ax.set_xlabel("Delivery Time (minutes)")
        ax.set_ylabel("Rating")

    # create a dedicated colorbar axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(hb, cax=cbar_ax, label="Number of Orders")

    # Adding labels
    plt.suptitle("Density of Delivery Time vs Customer Ratings Across Platforms")
    plt.subplots_adjust(right=0.9)
    plt.show()