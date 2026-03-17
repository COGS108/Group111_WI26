import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    
    df = df.copy()
    
    df['rating_jitter'] = df['rating'] + np.random.uniform(-0.1, 0.1, size=len(df))
    
    states = ['NY','NJ','CA','IL','MA']

    plt.figure(figsize=(18, 10))

    for i, state in enumerate(states, 1):
        plt.subplot(2, 3, i)  # 2 rows, 3 columns
        
        df_subset = df[df['state'] == state]
        
        # Scatter plot
        sns.scatterplot(
        data=df_subset,
        x='delivery_time',
        y='rating_jitter',
        alpha=0.6,
        color='skyblue'
        )
    
        # Regression line
        sns.regplot(
        data=df_subset,
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




def compare_lineplot(df1, df2, df3):  # intended to be called on (foodhub, grubhub, doordash)
    """ 
    Creates a lineplot comparing each dataframe's delivery time to the associated ratings.


    Parameters
    ----------
    df1 : pd.DataFrame
         The first dataframe, default set to Foodhub dataframe.
         
    df2 : pd.DataFrame
         The second dataframe
    df3 : pd.DataFrame
         The third dataframe

    Returns
    -------
    None
         Displays a lineplot.
    """
    
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

    bin_means = (
        combined.groupby(['platform', 'bin_mid'], observed=True)['rating']
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10,6))
    sns.lineplot(data=bin_means, x='bin_mid', y='rating', hue='platform', marker='o')

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
    
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    datasets = [
        ("FoodHub", df1),
        ("Grubhub", df2),
        ("DoorDash", df3)
    ]

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

    plt.suptitle("Density of Delivery Time vs Customer Ratings Across Platforms")

    plt.subplots_adjust(right=0.9)

    plt.show()