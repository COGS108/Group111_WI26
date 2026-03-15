import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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