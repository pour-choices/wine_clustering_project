#Importing required packages and files
import pandas as pd
import numpy as np
import os


#Removes warnings and imporves asthenics
import warnings
warnings.filterwarnings("ignore")

def pour_wine():
    """ 
    This function takes the red and white wine quality csvs, adds a type column and combines into the wine DataFrame.
    """
    
    filename = "winequality.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        red_df = pd.read_csv("winequality-red.csv")
        white_df = pd.read_csv("winequality-white.csv")

        red_df['type'] = 'red'
        white_df['type'] = 'white'

        wine_df = pd.concat([red_df, white_df], ignore_index=True)

        wine_df.to_csv(filename, index=False)
    
    return wine_df