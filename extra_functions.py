import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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