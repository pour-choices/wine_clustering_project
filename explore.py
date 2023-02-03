import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans


#Removes warnings and imporves asthenics
import warnings
warnings.filterwarnings("ignore")


def elbow_method(df, cluster_list): 
    """ 
    This function takes a DataFrame and list of continuous columns to find the inertia for 1-9 clusters and plot them.
    Note: DataFrame should be scaled prior to using this tool.
    """
    inertia0 = []

    for n in range(1,10):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(df[cluster_list])

        df['cluster_preds'] = kmeans.predict(df[cluster_list])

        inertia0.append({'n_clusters': n,
                        'inertia': kmeans.inertia_})

    inertia0 = pd.DataFrame(inertia0)

    sns.relplot(data=inertia0, x='n_clusters', y='inertia', kind='line', marker='o')
    plt.title("Elbow method")
    plt.show()
    
def display_clusters(df, cluster_list, n_clusters=3):
    """
    Intakes a DataFrame and list to cluster. List should be two columns. I have not tested it on more.
    n_clusters will work up to 9. After that there are no more colors in the dictionary.
    Result is a plot of the clusters and centroids.
    """
    
    #Creates a DataFrame of the desired columns
    X = df[cluster_list]

    #Builds the cluster object with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X[cluster_list])
    
    #Builds a column with the cluster numbers
    X['cluster_preds'] = kmeans.predict(X[cluster_list])
    
    #Builds a DataFrame of the centroids of the cluster
    cluster_df = pd.DataFrame(kmeans.cluster_centers_, columns = cluster_list)
    cluster_df.rename_axis(index='centroid')
    
    #Color dictionary for the clusters
    cb_colors = {0:'#377eb8', 1:'#ff7f00', 2:'#4daf4a',
                      3:'#f781bf', 4:'#a65628', 5:'#984ea3',
                      6:'#999999', 7:'#e41a1c', 8:'#dede00'}

    #Creates the figure and axis objects to build upon
    fig, ax = plt.subplots(facecolor='gainsboro', edgecolor='dimgray')
    
    #Groups by the clusters to plot n_cluster number of scatterplots on the same figure
    grouped = X.groupby('cluster_preds')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x=cluster_list[0], y=cluster_list[1], marker='.', label=key, color=cb_colors[key])
    #Plots the centroids
    ax.scatter(cluster_df[cluster_list[0]], cluster_df[cluster_list[1]], marker='x', color = 'red', label='Centroid')
    ax.set_xlabel(f'{cluster_list[0]}')
    ax.set_ylabel(f'{cluster_list[1]}')
    ax.set_title("Clusters and Centroids for KMeans Clustering")
    ax.legend()
    plt.show()
    
def check_p_val(p_val, h0, ha, s=None, alpha=0.05):
    """
    Checks if p value is significant or not and prints the associated string
    """
    
    #Pretty self explanitory.
    if p_val < alpha:
        print(f'We have evidence to reject the null hypothesis.')
        print(f'{ha}')
        if s != None:
            print(f'Significance level of: {round(s,4)}')
    else:
        print(f'We do not have evidence to reject the null hypothesis.')
        print(f'{h0}')
        
def correlation_test(df, target_col, alpha=0.05):
    """
    Maybe create a function that automatically seperates continuous from discrete columns.
    """
    
    list_of_cols = df.select_dtypes(include=[int, float]).columns
              
    metrics = []
    for col in list_of_cols:
        result = stats.anderson(df[col])
        #Checks skew to pick a test
        if result.statistic < result.critical_values[2]:
            corr, p_value = stats.pearsonr(df[target_col], df[col])
            test_type = '(P)'
        else:
            # I'm unsure how this handles columns with null values in it.
            corr, p_value = stats.spearmanr(df[target_col],
                                            df[col], nan_policy='omit')
            test_type = '(S)'

        #Answer logic
        if p_value < alpha:
            test_result = 'relationship'
        else:
            test_result = 'independent'

        temp_metrics = {"Column":f'{col} {test_type}',
                        "Correlation": corr,
                        "P Value": p_value,
                        "Test Result": test_result}
        metrics.append(temp_metrics)
    distro_df = pd.DataFrame(metrics)              
    distro_df = distro_df.set_index('Column')

    #Plotting the relationship with the target variable (and stats test result)
    my_range=range(1,len(distro_df.index) + 1)
    hue_colors = {'relationship': 'green', 'independent':'red'}

    plt.figure(figsize=(6,5))
    plt.axvline(0, c='tomato', alpha=.6)

    plt.hlines(y=my_range, xmin=-1, xmax=1, color='grey', alpha=0.4)
    sns.scatterplot(data=distro_df, x="Correlation",
                    y=my_range, hue="Test Result", palette=hue_colors,
                    style="Test Result")
    plt.legend(title="Stats test result")

    # Add title and axis names
    plt.yticks(my_range, distro_df.index)
    plt.title(f'Statistics tests of {target_col}', loc='center')
    plt.xlabel('Neg Correlation            No Correlation            Pos Correlation')
    plt.ylabel('Feature')
    
    #Saves plot when it has a name and uncommented
    #plt.savefig(f'{train.name}.png')