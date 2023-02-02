import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from env import get_connection
from scipy import stats
import os


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor


#Removes warnings and imporves asthenics
import warnings
warnings.filterwarnings("ignore")

"""############## -- AVAILABLE FUNCTIONS -- ##############
Acquire:
-pour_wine()
-wrangle_iris()
-wrangle_mall()
-get_telco_data()
-wrangle_zillow()

Prepare:
train_validate(df, stratify_col = None, random_seed=1969)
-get_dummies(df, dumb_columns)
-find_na(df)
-outlier_ejector(dataframe, column, k=1.5)
-outlier_detector(dataframe, column, k=1.5)
-handle_missing_values(df, prop_required_column = .4, prop_required_row = .25)

Explore:
-exploring_cats(train, target_column, alpha = 0.05)
-elbow_method(df, cluster_list)
-display_clusters(df, cluster_list, n_clusters=3)
-check_p_val(p_val, h0, ha, s=None, alpha=0.05)
-explore_relationships(feature_list, train, target_col, visuals = False)
-correlation_test(df, target_col, alpha=0.05)

Modeling:
-encode_and_dummies(df, target_column = None ,random_seed=1969)
-train_val_test(train, val, test, target_col)
-find_regression_baseline(y_train)
-scale_cont_columns(train, val, test, , cont_columns, scaler_model = 1)

############## -- ACQUIRE FUNCTIONS -- ##############"""

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


def wrangle_iris():
    """
    This function gets all data from the iris database.
    """
    filename = "iris_db.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        
        # read the SQL query into a dataframe
        query = """
        SELECT * FROM measurements 
        LEFT JOIN species USING (species_id);
        """

        df = pd.read_sql(query, get_connection('iris_db'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df
    
def wrangle_mall():
    """
    This function gets all data from the mall_customers database.
    """
    filename = "mall_customers.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        
        # read the SQL query into a dataframe
        query = """
        SELECT * FROM customers;
        """

        df = pd.read_sql(query, get_connection('mall_customers'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df


def get_telco_data():
    """
    This function reads the telco_churn data from Codeup db into a df.
    """
    filename = "telco_churn.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        
        # read the SQL query into a dataframe
        query = """
        SELECT * FROM customer_subscriptions
        LEFT JOIN customer_churn USING (customer_id)
        LEFT JOIN customer_contracts USING (customer_id)
        LEFT JOIN customer_details USING (customer_id)
        LEFT JOIN customer_payments USING (customer_id)
        LEFT JOIN customer_signups USING (customer_id)
        LEFT JOIN contract_types USING (contract_type_id)
        LEFT JOIN internet_service_types USING (internet_service_type_id)
        LEFT JOIN payment_types USING (payment_type_id);
        """
        df = pd.read_sql(query, get_connection('telco_churn'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df

def wrangle_zillow(query_num = 2):
    """
    This function reads the zillow data from Codeup db into a df.
    Changes the names to be more readable.
    Drops null values.
    """
    filename = "zillow_2017.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, parse_dates=['transactiondate'])
    else:
        
        # read the SQL query into a dataframe
        query1 = """
        SELECT taxvaluedollarcnt, bedroomcnt, bathroomcnt,
        calculatedfinishedsquarefeet, transactiondate
        FROM properties_2017
        LEFT JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusetypeid LIKE 261 AND
        transactiondate like '2017%%';
        """
        
        query2 = """
        SELECT taxvaluedollarcnt, bedroomcnt,
        bathroomcnt, calculatedfinishedsquarefeet,
        transactiondate, hashottuborspa, decktypeid,
        garagecarcnt, poolcnt, fips, latitude, longitude
        FROM properties_2017
        LEFT JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusetypeid LIKE 261 AND
        transactiondate like '2017%%';
        """
        
        query3 = """
        SELECT * FROM properties_2017
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN predictions_2017 USING (parcelid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        LEFT JOIN unique_properties USING (parcelid)
        WHERE transactiondate LIKE "2017%%";
        """

        #Selects a query
        if query_num == 1:
            query = query1
        elif query_num == 2:
            query = query2
        elif query_num == 3:
            query = query3
            
        
        df = pd.read_sql(query, get_connection('zillow'))
        
        # Remove NAs. No significant change to data. tax_values upper outliers were affected the most.
        df.rename(columns = {'bedroomcnt': 'bedrooms',
                             'bathroomcnt': 'bathrooms',
                             'calculatedfinishedsquarefeet': 'sqft',
                             'taxvaluedollarcnt':'tax_value', 
                             'hashottuborspa' : 'hottub_spa', 
                             'decktypeid': 'deck', 
                             'poolcnt': 'pool',
                             'fips':'County'}, 
                  inplace=True)
        df.County = df.County.map({6037.0:'Los Angeles', 6059.0:'Orange', 6111.0:'Ventura'})
        df['latitude'] = df['latitude'] / 10_000_000
        df['longitude'] = df['longitude'] / 100_000_000

        df['transactiondate'] = pd.to_datetime(df['transactiondate'])
        
        sqft_bins = [0, 200, 400, 600, 800, 1000, 1200, 1400,
                     1600, 1800, 2000, 2200, 2400, 2600, 2800,
                     3000, 3200, 3400, 3600, 3800, 4000, 4200,
                     4400, 4600, 4800, 5000]        
        bin_labels = [200, 400, 600, 800, 1000, 1200, 1400, 1600,
                      1800, 2000, 2200, 2400, 2600, 2800, 3000,
                      3200, 3400, 3600, 3800, 4000, 4200, 4400,
                      4600, 4800, 5000]        
        df['sqft_bins'] = pd.cut(df.sqft, bins = sqft_bins,
                                 labels = bin_labels)        
        value_bins = [0, 400000, 800000, 1200000, 1600000, 30000000]        
        value_bin_labels = ['$400k', '$800k', '$1.2m', '$1.5m', '$1.5m+']
        df['value_bins'] = pd.cut(df.tax_value, bins = value_bins,
                                  labels = value_bin_labels)
        df['hottub_spa'] = df['hottub_spa'].notna().astype('int')
        df['deck'] = df['deck'].notna().astype('int')
        df['pool'] = df['pool'].notna().astype('int')
        df['has_garages'] = df['garagecarcnt'].notna().astype('int')
        df['garagecarcnt'].fillna(0, inplace=True)
        df['num_of_features'] = df[['pool','deck','hottub_spa', 'has_garages']].sum(axis=1)
        df = df.dropna()

        cols_outliers = ['bedrooms', 'bathrooms', 'sqft', 'tax_value']
        for col in cols_outliers:
            df = df[df[col] <= df[col].quantile(q=0.99)]
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df


"""############## -- PREPARE FUNCTIONS -- ##############"""
# 20% test, 80% train_validate
# then of the 70% train_validate: 30% validate, 70% train. 
def train_validate(df, stratify_col = None, random_seed=1969):
    """
    This function takes in a DataFrame and column name for the stratify argument (defualt is None).
    It will split the data into three parts for training, testing and validating.
    """
    #This is logic to set the stratify argument:
    stratify_arg = ''
    if stratify_col != None:
        stratify_arg = df[stratify_col]
    else:
        stratify_arg = None
    
    #This splits the DataFrame into 'train' and 'test':
    train, test = train_test_split(df, train_size=.8, stratify=stratify_arg, random_state = random_seed)
    
    #The length of the stratify column changed and needs to be adjusted:
    if stratify_col != None:
        stratify_arg = train[stratify_col]
        
    #This splits the larger 'train' DataFrame into a smaller 'train' and 'validate' DataFrames:
    train, validate = train_test_split(train, train_size=.6, stratify=stratify_arg, random_state = random_seed)
    return train, validate, test


def get_dummies(df, dumb_columns):
    """
    #Creates dummy columns based on list 'dumb_columns' and drops dummy source columns
    """
    #Pandas dummies function
    df = pd.get_dummies(df, columns=dumb_columns)
    
    return df


def find_na(df):
    list_of_na = []
    for col in df:
        temp_dict = {'column_name': f'{col}' , 
                     'num_rows_missing': df[col].isna().sum(),
                     'unique_values': df_sorted[col].value_counts().sum(),
                     'pct_rows_missing': round(df[col].isna().sum() / len(df[col]),5)
                     }

        list_of_na.append(temp_dict)
    print("The effect of dropping all rows with null values:")
    df.describe() - df.dropna().describe()
    na_df = pd.DataFrame(list_of_na)
    na_df.set_index('column_name')
    return na_df

def outlier_ejector(dataframe, column, k=1.5):
    """
    This function takes in a dataframe and looks for upper outliers.
    """
    q1, q3  = dataframe[column].quantile(q=[0.25, 0.75])
    iqr = q3 - q1
    
    
    lower_bound = q1 - (k * iqr)
    upper_bound = q3 + (k * iqr)
    
    high_items = dataframe[column] > upper_bound
    low_items = dataframe[column] < lower_bound

    
    return dataframe[~low_items & ~high_items]

def outlier_detector(dataframe, column, k=1.5):
    """
    This function takes in a dataframe and looks for upper outliers.
    """
    q1, q3  = dataframe[column].quantile(q=[0.25, 0.75])
    iqr = q3 - q1
    
    
    lower_bound = q1 - (k * iqr)
    upper_bound = q3 + (k * iqr)
    
    high_items = dataframe[column] > upper_bound
    low_items = dataframe[column] < lower_bound

    
    return dataframe[low_items & high_items]

def handle_missing_values(df, prop_required_column = .4, prop_required_row = .25):
    """
    This function drops columns then rows which contain a certain amount of null values.
    """
    #Lists to hold values
    drop_cols = []
    drop_rows = []
    na_cols_not_drop = ['taxdelinquencyyear']
    
    #Finds columns with lots of na values
    for col in df:
        if (df[col].isna().sum()/len(df) > prop_required_column):
            if col in na_cols_not_drop:
                pass
            else:
                drop_cols.append(f'{col}')
    #Drops columns with lots of na values        
    df = df.drop(columns=drop_cols)
    num_rows = int(len(df.columns) * prop_required_row)
    #Drops rows with lots of na values
    df = df.dropna(thresh=num_rows) 
    
    return df

"""############## -- EXPLORE FUNCTIONS -- ##############"""

def exploring_cats(train, target_column, alpha = 0.05):
    """
    Input DataFrame and a string of the target_column name.
    Performs chi^2 test with a default alpha of 0.05 on each categorical feature.
    Prints a visualization and list of columns whos data occures exclusivly 
    in the target group or non-target group.
    """

    #Lists to hold variables
    distros = []
    drivers = []
    non_drivers = []
    chi_test_result = []
    sus_columns = []
    
    #This snags int columns and drops those that have more than 2 values.
    plot_df = train.select_dtypes(exclude=['object','bool',
                                           'float', 'datetime'])
    
    for col in plot_df:
        if len(plot_df[col].value_counts()) > 2:
            plot_df.drop(columns=col, inplace = True)
    
    #Seperating target rows
    target_df = plot_df[plot_df[target_column] == 1]

    #Warning that the below is prefered... IDK why:
    #df.loc[:,('one','second')]
    target_df.drop(columns=target_column, inplace = True)
    
    #Seperating non-target rows
    not_target = plot_df[plot_df[target_column] == 0]
    not_target.drop(columns=target_column, inplace = True)
    
    #Creating the Target Indication DataFrame

    for item in target_df:
        target = round(target_df[item].mean(),3)
        not_tar = round(not_target[item].mean(),3)

        output = {"Column" : item,
                  "Target %": target, 
                  "Not Target %": not_tar,
                  "Target Indication":(target - not_tar)}

        distros.append(output)
        
        #Checks all data points occure in one group or the other
        #Adds to a list of suspicious columns to be printed later
        if (target - not_tar) == 1.0 or (target - not_tar) == -1.0:
            sus_columns.append(item)

    #This turns the info into a DataFrame
    distro_df = pd.DataFrame(distros)              
    distro_df = distro_df.set_index('Column')

    #Seperate out columns to investigate, Target Indication = 1 or -1

    for feature in distro_df.T:

    # Let's run a chi squared to compare proportions, to have more confidence
        null_hypothesis = f'{feature} and {target_column} are independent.'
        alternative_hypothesis = f'there is a relationship between {feature} and {target_column}'

        # Setup a crosstab of observed df target to df feature
        observed = pd.crosstab(train[target_column], train[feature])

        #Stats test
        chi2, p_value, degf, expected = stats.chi2_contingency(observed)

        #Answer logic
        if p_value < alpha:
            chi_test_result.append('relationship')

        else:
            chi_test_result.append('independent')
        
    distro_df['chi_test_result'] = chi_test_result

    
    #Plotting the relationship with the target variable (and stats test result)
    my_range=range(1,len(distro_df.index) + 1)
    hue_colors = {'relationship': 'green', 'independent':'red'}

    plt.figure(figsize=(6,9))
    plt.axvline(0, c='tomato', alpha=.6)

    plt.hlines(y=my_range, xmin=-1, xmax=1, color='grey', alpha=0.4)
    sns.scatterplot(data=distro_df, x='Target Indication',
                    y=my_range, hue='chi_test_result', palette=hue_colors,
                    style='chi_test_result')
    plt.legend(title='$Chi^2$ test result')

    # Add title and axis names
    plt.yticks(my_range, distro_df.index)
    plt.title(f'Drivers of {target_column}', loc='center')
    plt.xlabel('Occures Less           Occures Evenly           Occures More')
    plt.ylabel('Feature')
    
    #Saves plot when it has a name and uncommented
    #plt.savefig(f'{train.name}.png')
    #Gives you columns which might need looking into
    if len(sus_columns) > 0:
        print(f'Columns with suspicious data to investigate: {sus_columns}')

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
        
def explore_relationships(feature_list, train, target_col, visuals = False):
    """
    This function takes in a list of features, grabs the .describe() metrics associated with the target column.
    *** Inputs ***
    feature_list: List of DataFrame column names to iterate through and compare to target column.
    train: Panda's DataFrame to explore.
    target_col: String. Title of target column.
    *** Output ***
    DataFrame with metrics to explore
    """
    metrics = []
    for feature in feature_list:
        num_items = train[feature].unique()
        num_items.sort()
        for item in num_items:
            temp_df = train[train[feature] == item][target_col].describe()
            temp_metrics = {
                'comparison' : f'{item}_{feature}',
                'count' : round(temp_df[0],0),
                'mean' : round(temp_df[1],0),
                'std' : round(temp_df[2],0),
                'min' : round(temp_df[3],0),
                '25%' : round(temp_df[4],0),
                '50%' : round(temp_df[5],0),
                '75%' : round(temp_df[6],0),
                'max' : round(temp_df[7],0)}
            metrics.append(temp_metrics)

    feature_per_item = pd.DataFrame(metrics)
    if visuals == True:
        sns.lineplot(data=feature_per_item, x='comparison', y='25%',
                             legend='brief').set(title=f'{target_col} to {feature} comparison',
                                                 xlabel =f'{feature}', ylabel = f'{target_col}')
        sns.lineplot(data=feature_per_item, x='comparison', y='mean', markers=True)
        sns.lineplot(data=feature_per_item, x='comparison', y='50%')
        sns.lineplot(data=feature_per_item, x='comparison', y='75%')
        plt.ylabel(f'{target_col}')
        plt.xlabel(f'{item}_{feature}')
        
    return feature_per_item

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

    plt.figure(figsize=(6,9))
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

"""############## -- MODELING FUNCTIONS -- ##############"""

def encode_and_dummies(df, target_column = None ,random_seed=1969):
    """
    Target column should be in a yes/no, True/False, 0/1 format.
    This function is not designed to handle null values.
    Input DataFrame and a string of the target_column name.
    Outputs train, validate and test DataFrame with binary 
    columns as 0/1 and dummy columns.
    """
    
    #Variable
    dumb_columns = []

    #Values that will be turned to an integer of  0 or 1
    values_to_encode = {'Yes': 1, 'yes': 1, 'y': 0, 'Y': 1,
                      True : 1, 'T': 1, 'True': 1, 't': 1,'true': 1,
                      'No': 0, 'no': 0, 'n': 0, 'N' : 0,
                      False : 0, 'F': 0, 'f': 0, 'False': 0, 'false':0,
                       '0': 0, '1': 1, 'Win':1, 'Lose':0, 'win':1, 'lose':0,
                       'W':1, 'L':0, 'w':1, 'l':0}

    #Seperate out object and bool data type columns into new df:
    object_df = df.select_dtypes(include=['object','bool'])
    
    #For loop to find applicable columns
    for col in object_df:
        change = False

        #Filter to check if the values are the correct length and in the values_to_encode dict
        if (len(object_df[col].value_counts()) == 2):
            for item in object_df[col].unique():
                if item in values_to_encode.keys():
                    change = True

            #Swaps out old column with the new binary column
            if change == True:
                df = df.drop(columns=col)
                df = pd.concat([df, object_df[col].replace(to_replace=values_to_encode).astype('int')],
                               axis=1)                
            else:
                dumb_columns.append(object_df[col].name)
            change = False

        #Create dummy values for columns with < 6 unique values:        
        elif (len(object_df[col].value_counts()) < 6 ):
            dumb_columns.append(object_df[col].name)
            
    #Creates dummy columns based on list 'dumb_columns' and drops dummy source columns
    dummy_df = pd.get_dummies(object_df[dumb_columns])
    df = pd.concat([df, dummy_df], axis=1)
    df.drop(columns=dumb_columns, inplace = True)    

    #This splits the dataframe into a training, validate and test set.
    
    #This is logic to set the stratify argument:
    stratify_arg = ''
    if target_column != None:
        stratify_arg = df[target_column]
    else:
        stratify_arg = None
    
    #This splits the DataFrame into 'train' and 'test':
    train, test = train_test_split(df, train_size=.7, stratify=stratify_arg,
                                   random_state = random_seed)
    
    #The length of the stratify column changed and needs to be adjusted:
    if target_column != None:
        stratify_arg = train[target_column]
        
    #This splits the larger 'train' DataFrame into a smaller 'train' and 'validate' DataFrames:
    train, validate = train_test_split(train, test_size=.4,
                                       stratify=stratify_arg,
                                       random_state = random_seed)
    return train, validate, test


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




def find_model_scores(df):
    """
    This function takes in the target DataFrame, runs the data against four
    machine learning models and outputs some visuals.
    """
    #Creates a copy so the original data is not affected
    ml_df = df.copy()

    #Drops columns not used in modeling
    ml_df = df.drop(columns=['transactiondate', 'sqft_bins',
                             'value_bins', 'County'])
    #Creates dummy columns
    ml_df = pd.get_dummies(columns=['bedrooms', 'bathrooms',
                                    'num_of_features', 'garagecarcnt'],
                           data=ml_df)
    #Splits data into train, validate and test datasets
    train, val, test = train_validate(ml_df)
    
    #Scales continuous data#Scaling the data
    train, val, test = scale_zillow(train, val, test, scaler_model = 3,
                                    cont_columns = ['sqft'])

    #Seperate target column from feature columns
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test(train, val, test, target_col)
    
    #Eastablishes the standard to beat
    baseline = find_baseline(y_train)
    
    #List for gathering metrics
    rmse_scores = []

    
    """ *** Builds and fits Linear Regression Model (OLS) *** """
    
    
    lm = LinearRegression(normalize=True)
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
                    'RMSE on Train': round(rmse_train,0),
                    'RMSE on Validate': round(rmse_val,0)})
    
    
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
                    'RMSE on Train': round(rmse_train,0),
                    'RMSE on Validate': round(rmse_val,0)})
    
    
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
                        'RMSE on Train': round(rmse_train,0),
                        'RMSE on Validate': round(rmse_val,0)})
    
    
    """ *** Builds and fits Polynomial regression Model *** """

    
    #Polynomial Regression part:
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=1)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_val)
    X_test_degree2 = pf.transform(X_test)

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
                        'RMSE on Train': round(rmse_train,0),
                        'RMSE on Validate': round(rmse_val,0)})
    
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

    #ax.bar_label(rects1, padding=4)
    #ax.bar_label(rects2, padding=4)
    fig.tight_layout()
    #plt.savefig('best_model_all_features.png')
    plt.show()

