# README


* **Title**: Pour Choices

* **Project Description**: Our project is to use statistical analysis and machine learning to find the relationships between various wine chemicals to determine an approximate quality rating. 
* **Project Goal**: Our goal is to find features which affect quality, analyize how they affect quality and develop a machine learning model to determine a quality rating.
* **Initial Hypotheses**: Chemical features will provide enough information to determine a quality score for each wine observation.
* **Project Plan**: Acquire the two csv files, combine and cache them. Perform initial exploration to determine the number and extent of null values. Find data types. See ranges of data available. Then split the data into train, validate and test sets for model integrity and explore the training set to find relationships. Perform statistical testing to find if a relationship exists and the strength of correlation. Use clustings for exploration and modeling to see if it improves performance. Establish a baseline value to beat. Select a final model and run it on the test data set.

#### **Data Dictionary** 

| Feature |	Definition |
|:--------|:-----------|
|quality| Target column. A wine rating from an unknown source which ranges from 3 to 9 in the dataset.|
|alcohol| The ratio of ethonal to water in the wine. Unit is alcohol by volume or 'ABV'.|
|density| Measured in grams/milliliter. Used to help determine alcohol content.|
|pH| A measure of how acidic/basic the liquid is. The range is 0 to 14. 7 is neutral. Less than 7 is acidic and greater than 7 indicates a base.|
|fixed acidity| Acids left after a steam distillation test. Measured in g/L.|
|volatile acidity| Acids which can be removed by steam distillation. Measured in g/L.|
|citric acid| Most commonly used as an acid supplement during the fermentation process to help winemakers boost the acidity|
|residual sugar| Natural grape sugars left in a wine after the alcoholic fermentation finishes. Usually measured in grams per litre (g/L).|
|chlorides| The amount of salt in the wine.|
|total sulfur dioxide| The portion of SO2 that is free in the wine plus the portion that is bound to other chemicals.|
|free sulfur dioxide| SO2 that is present in wine but has not yet reacted with other chemicals.|
|sulphates| A combination of sulfur dioxide molecules and sulfite ions|
|type| Indicates if the wine is classified as a red or white. |

#### **Steps to Reproduce** 
1. Install necessary python packages.
2. Clone the wine_clustering_project repository.
3. Download the red and white wine csv files from https://data.world/food/wine-quality
4. Unzip and store the red and white wine csv files in the wine_clustering_project folder.
5. Ensure the acquire.py, prepare.py, explore.py and modeling.py files are in the same folder as the blind_tastings.ipynb notebook.
6. Run the blind_tastings.ipynb notebook.

