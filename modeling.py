#Importing required packages and files
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Tools to build machine learning models and reports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor

#Removes warnings and imporves asthenics
import warnings
#warnings.filterwarnings("ignore")

def make_dummies(train, val, test, dumb_cols):
    """
    This function takes in the train, validate and test DataFrame, creates dummies and returns the DataFrames.
    """
    train = pd.get_dummies(train, columns=dumb_cols)
    val = pd.get_dummies(val, columns=dumb_cols)
    test = pd.get_dummies(test, columns=dumb_cols)
    return train, val, test

def scale_cont_columns(train, val, test, cont_columns, scaler_model = 1):
    """
    This takes in the train, validate and test DataFrames, scales the cont_columns using the
    selected scaler and returns the DataFrames.
    *** Inputs ***
    train: DataFrame
    validate: DataFrame
    test: DataFrame
    scaler_model (1 = MinMaxScaler, 2 = StandardScaler, else = RobustScaler)
    - default = MinMaxScaler
    cont_columns: List of columns to scale in DataFrames
    *** Outputs ***
    train: DataFrame with cont_columns scaled.
    val: DataFrame with cont_columns scaled.
    test: DataFrame with cont_columns scaled.
    """
    #Create the scaler
    if scaler_model == 1:
        scaler = MinMaxScaler()
    elif scaler_model == 2:
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    #Make a copy
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    
    #Fit the scaler
    scaler = scaler.fit(train[cont_columns])
    
    #Build the new DataFrames
    train_scaled[cont_columns] = pd.DataFrame(scaler.transform(train[cont_columns]),
                                                  columns=train[cont_columns].columns.values).set_index([train.index.values])

    val_scaled[cont_columns] = pd.DataFrame(scaler.transform(val[cont_columns]),
                                                  columns=val[cont_columns].columns.values).set_index([val.index.values])

    test_scaled[cont_columns] = pd.DataFrame(scaler.transform(test[cont_columns]),
                                                 columns=test[cont_columns].columns.values).set_index([test.index.values])
    #Sending them back
    return train_scaled, val_scaled, test_scaled


def train_val_test(train, val, test, target_col):
    """
    Seperates out the target variable and creates a series with only the target variable to test accuracy.
    """
    #Seperating out the target variable
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_val = val.drop(columns = [target_col])
    y_val = val[target_col]

    X_test = test.drop(columns = [target_col])
    y_test = test[target_col]
    return X_train, y_train, X_val, y_val, X_test, y_test

def find_regression_baseline(y_train):
    """
    This function shows a comparison in baselines for mean and median.
    Output is the RMSE error when using both mean and median.
    """
    
    # Train set
    bl_df = pd.DataFrame({'actual':y_train, 'mean_bl':y_train.mean(), 'median_bl':y_train.median()})
    rmse_train_mean = mean_squared_error(bl_df['actual'], bl_df['mean_bl'], squared=False)
    rmse_train_median = mean_squared_error(bl_df['actual'], bl_df['median_bl'], squared=False)
    
    
    if min(rmse_train_mean, rmse_train_median) == rmse_train_median:
        print(f'Using RMSE Median training baseline: {round(rmse_train_median,4):,.4f}')
    elif min(rmse_train_mean, rmse_train_median) == rmse_train_mean:
        print(f'Using RMSE Mean training baseline: {round(rmse_train_mean,4):,.4f}')
    
    return min(rmse_train_mean, rmse_train_median)


def find_model_scores(X_train, y_train, X_val, y_val, baseline):
    
    #List for gathering metrics
    rmse_scores = []


    """ *** Builds and fits Linear Regression Model (OLS) *** """


    lm = LinearRegression(normalize=True, positive=True)
    lm.fit(X_train, y_train)

    #Train data
    lm_preds = pd.DataFrame({'actual':y_train})
    lm_preds['pred_lm'] = lm.predict(X_train)

    #Validate data
    lm_val_preds = pd.DataFrame({'actual':y_val})
    lm_val_preds['lm_val_preds'] = lm.predict(X_val)

    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(lm_preds['actual'],
                                    lm_preds['pred_lm'],
                                    squared=False) 
    rmse_val = mean_squared_error(lm_val_preds['actual'],
                                  lm_val_preds['lm_val_preds'],
                                  squared=False) 

    #Adds score to metrics list for later comparison
    rmse_scores.append({'Model':'OLS Linear',
                    'RMSE on Train': round(rmse_train,4),
                    'RMSE on Validate': round(rmse_val,4)})
    
    """ *** Builds and fits Lasso Lars Model *** """


    lars = LassoLars(alpha=.25)
    lars.fit(X_train, y_train)

    #Train data
    ll_preds = pd.DataFrame({'actual':y_train})
    ll_preds['pred_ll'] = lars.predict(X_train)

    #Validate data
    ll_val_preds = pd.DataFrame({'actual':y_val})
    ll_val_preds['ll_val_preds'] = lars.predict(X_val)

    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(ll_preds['actual'],
                                    ll_preds['pred_ll'],
                                    squared=False)
    rmse_val = mean_squared_error(ll_val_preds['actual'],
                                  ll_val_preds['ll_val_preds'],
                                  squared=False)

    #Adds score to metrics list for later comparison
    rmse_scores.append({'Model':'Lasso Lars',
                    'RMSE on Train': round(rmse_train,4),
                    'RMSE on Validate': round(rmse_val,4)})
    
    """ *** Builds and fits Tweedie Regressor (GLM) Model *** """

    glm = TweedieRegressor(power=1, alpha=1)    
    glm.fit(X_train, y_train)

    #Train data
    glm_preds = pd.DataFrame({'actual':y_train})
    glm_preds['pred_glm'] = glm.predict(X_train)

    #Validate data
    glm_val_preds = pd.DataFrame({'actual':y_val})
    glm_val_preds['glm_val_preds'] = glm.predict(X_val)

    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(glm_preds['actual'],
                                    glm_preds['pred_glm'],
                                    squared=False) 
    rmse_val = mean_squared_error(glm_val_preds['actual'],
                                  glm_val_preds['glm_val_preds'],
                                  squared=False)

    #Adds score to metrics list for later comparison
    rmse_scores.append({'Model':'Tweedie',
                        'RMSE on Train': round(rmse_train,4),
                        'RMSE on Validate': round(rmse_val,4)})
    
    """ *** Builds and fits Polynomial regression Model *** """


    #Polynomial Regression part:
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=1)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_val)
    #X_test_degree2 = pf.transform(X_test)

    #Polynomial Regression being fed into Linear Regression:
    lm2 = LinearRegression(normalize=True)
    lm2.fit(X_train_degree2, y_train)

    #Train data
    lm2_preds = pd.DataFrame({'actual':y_train})
    lm2_preds['pred_lm2'] = lm2.predict(X_train_degree2)

    #Validate data
    lm2_val_preds = pd.DataFrame({'actual':y_val})
    lm2_val_preds['lm2_val_preds'] = lm2.predict(X_validate_degree2)

    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(lm2_preds['actual'],
                                    lm2_preds['pred_lm2'],
                                    squared=False) 
    rmse_val = mean_squared_error(lm2_val_preds['actual'],
                                  lm2_val_preds['lm2_val_preds'],
                                  squared=False)

    #Adds score to metrics list for later comparison
    rmse_scores.append({'Model':'Polynomial',
                        'RMSE on Train': round(rmse_train,4),
                        'RMSE on Validate': round(rmse_val,4)})
    
    """ *** Later comparison section to display results *** """

    #Builds and displays results DataFrame
    rmse_scores = pd.DataFrame(rmse_scores)
    rmse_scores['Difference'] = round(rmse_scores['RMSE on Train'] - rmse_scores['RMSE on Validate'],2)    

    #Results were too close so had to look at the numbers
    print(rmse_scores)

    #Building variables for plotting
    rmse_min = min([rmse_scores['RMSE on Train'].min(),
                    rmse_scores['RMSE on Validate'].min(), baseline])
    rmse_max = max([rmse_scores['RMSE on Train'].max(),
                    rmse_scores['RMSE on Validate'].max(), baseline])

    lower_limit = rmse_min * 0.8
    upper_limit = rmse_max * 1.05
    
    """ *** Builds plot to display results *** """

    
    x = np.arange(len(rmse_scores))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(facecolor="gainsboro")
    rects1 = ax.bar(x - width/2, rmse_scores['RMSE on Train'],
                    width, label='Training data', color='#4e5e33',
                    edgecolor='dimgray') #Codeup dark green
    rects2 = ax.bar(x + width/2, rmse_scores['RMSE on Validate'],
                    width, label='Validation data', color='#8bc34b',
                    edgecolor='dimgray') #Codeup light green

    # Need to have baseline input:
    plt.axhline(baseline, label="Baseline Error", c='tomato', linestyle=':')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.axhspan(0, baseline, facecolor='palegreen', alpha=0.2)
    ax.axhspan(baseline, upper_limit, facecolor='red', alpha=0.3)
    ax.set_ylabel('RMS Error')
    ax.set_xlabel('Machine Learning Models')
    ax.set_title('Model Error Scores')
    ax.set_xticks(x, rmse_scores['Model'])

    plt.ylim(bottom=lower_limit, top = upper_limit)

    ax.legend(loc='upper right', framealpha=.9, facecolor="whitesmoke",
              edgecolor='darkolivegreen')

    fig.tight_layout()
    #plt.savefig('best_model_all_features.png')
    plt.show()
    
    
def final_test(X_train, y_train, X_val, y_val, X_test, y_test, baseline):
    """
    This function takes in the target DataFrame, runs the data against the
    machine learning model selected for the final test and outputs some visuals.
    """
    
    #List to capture scores
    final_rmse_scores = []
    
    """ *** Builds and fits Linear Regression Model (OLS) *** """  
    
    lm = LinearRegression(normalize=True, positive=True)
    lm.fit(X_train, y_train)
    
    #Train data
    lm_preds = pd.DataFrame({'actual':y_train})
    lm_preds['pred_lm'] = lm.predict(X_train)
    
    #Validate data
    lm_val_preds = pd.DataFrame({'actual':y_val})
    lm_val_preds['lm_val_preds'] = lm.predict(X_val)
    
    #Test data
    lm_test_preds = pd.DataFrame({'actual':y_test})
    lm_test_preds['lm_test_preds'] = lm.predict(X_test)
    
    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(lm_preds['actual'],
                                    lm_preds['pred_lm'],
                                    squared=False)
    rmse_val = mean_squared_error(lm_val_preds['actual'],
                                  lm_val_preds['lm_val_preds'],
                                  squared=False)
    rmse_test = mean_squared_error(lm_test_preds['actual'],
                              lm_test_preds['lm_test_preds'],
                              squared=False)
    
    #Adds score to metrics list for comparison
    final_rmse_scores.append({'Model':'Linear Regression Model (OLS)',
                              'RMSE on Train': round(rmse_train,4), 
                              'RMSE on Validate': round(rmse_val,4), 
                              'RMSE on Test': round(rmse_test,4)})
    # Turn scores into a DataFrame
    final_rmse_scores = pd.DataFrame(data = final_rmse_scores)
    print(final_rmse_scores)
    
    #Create visuals to show the results
    fig, ax = plt.subplots(facecolor="gainsboro")

    plt.figure(figsize=(4,4))
    ax.set_title('Linear Regression Model (OLS) results')
    ax.axhspan(0, baseline, facecolor='palegreen', alpha=0.2)
    ax.axhspan(baseline, ymax=450000, facecolor='red', alpha=0.3)
    ax.set_ylabel('RMS Error')    

    #x_pos = [0.5, 1, 1.5]
    width = 0.25

    bar1 = ax.bar(0.5, height=final_rmse_scores['RMSE on Train'],width =width, color=('#4e5e33'), label='Train', edgecolor='dimgray')
    bar2 = ax.bar(1, height= final_rmse_scores['RMSE on Validate'], width =width, color=('#8bc34b'), label='Validate', edgecolor='dimgray')
    bar3 = ax.bar(1.5, height=final_rmse_scores['RMSE on Test'], width =width, color=('tomato'), label='Test', edgecolor='dimgray')

    # Need to have baseline input:
    ax.axhline(baseline, label="Baseline", c='tomato', linestyle=':')
    ax.set_xticks([0.5, 1.0, 1.5], ['Training', 'Validation', 'Test']) 
    ax.set_ylim(bottom=.5, top=1)
    #Zoom into the important area
    ax.legend(loc='upper right', framealpha=.9, facecolor="whitesmoke", edgecolor='darkolivegreen')