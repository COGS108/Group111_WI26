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




def city_scatterplot(df=grubhub_ny_top): 
    """ 
    Creates a combined scatterplot comparing delivery_time to rating for each city in the New York Grubhub subset.


    Parameters
    ----------
    df : pd.DataFrame
         The GrubHub data subset containing delivery and rating data for New York, default set to grubhub_ny_top.

    Returns
    -------
    None
         Displays a singular scatterplot.
    """
    
    plt.figure(figsize=(10,6))

    sns.scatterplot(
        data=df,
        x='delivery_time',
        y='rating_jitter',
        hue='searched_city',
        palette='tab10',
        alpha=0.6,
        legend=False
    )

    # Regression lines per city
    for city in top_cities:
        subset = df[df['searched_city'] == city]
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




def one_city_plot(df=grubhub_city, city_to_plot='Cambridge'): 
     """ 
    Creates a singular scatterplot comparing delivery_time to rating for a specific city in the New York Grubhub subset.

    Parameters
    ----------
    df : pd.DataFrame
         The data subset containing delivery and rating data for New York, default set to grubhub_city.

    city_to_plot: str
         The name of the city, default set to 'Cambridge' as that is our city of interest.

    Returns
    -------
    None
         Displays a singular scatterplot.
    """
    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x='delivery_time',
        y='rating_jitter',
        alpha=0.6,
        color='skyblue'
    )

    sns.regplot(
        data=df,
        x='delivery_time',
        y='rating_jitter',
        scatter=False,
        color='darkred',
        line_kws={'linewidth':2}
    )

    plt.title(f"Delivery Time vs Rating in {city_to_plot}")
    plt.xlabel("Delivery Time (minutes)")
    plt.ylabel("Rating")
    plt.show()




def compare_lineplot(df1=foodhub, df2=grubhub, df3=doordash):
    """ 
    Creates a lineplot comparing each dataframe's delivery time to the associated ratings.


    Parameters
    ----------
    df1 : pd.DataFrame
         The first dataframe, default set to Foodhub dataframe.
         
    df2 : pd.DataFrame
         The second dataframe, default set to Grubhub dataframe.
         
    df3 : pd.DataFrame
         The third dataframe, default set to Doordash dataframe.

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




def compare_hexbin_plot(df1=foodhub, df2=grubhub, df3=doordash):
    """ 
    Creates a hexagonal bin plot comparing each dataframe's delivery time to the associated ratings.


    Parameters
    ----------
    df1 : pd.DataFrame
         The first dataframe, default set to Foodhub dataframe.
         
    df2 : pd.DataFrame
         The second dataframe, default set to Grubhub dataframe.
         
    df3 : pd.DataFrame
         The third dataframe, default set to Doordash dataframe.

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