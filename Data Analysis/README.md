# Will it be Warmer Tomorrow?
### *A Machine Learning Project by Ryan Li Jian Tang*

This is a project based on weather data from Australia. The objective is to find ways to correctly predict if the next day will be warmer depending on current factors, but not limited to, like date, wind direction, temperature, clouds, pressure. 

**Models Used:**
- Decision Trees
- Naive Bayes
- Bagging
- Boosting
- Random Forest
- Artificial Neural Network
  
A 2000 days sample of the goliath dataset was used instead to ease the training time in building models. [*Sample Data Link*](https://raw.githubusercontent.com/RyTang/Monash-Projects/main/Data%20Analysis/Machine%20Learning/WarmerTomorrow2022.csv)

## General Process Done:
1. Data Preparation
2. Data Analysis
3. Creation on Model
4. Compare Results of Models
5. Omitting Variables
6. Tune hyper-parameters on Models

To create a non-biased model, it is important to ensure that the data is processed and cleansed.  Meaning anomalous data should be highlighted and fixed. Data will then be analysed in attempts of omitting variables that do not reduce the entropy of the dataset: i.e through efforts of correlation matrix. Thereby, the models can then be trained upon the cleaned dataset. Wherefore, analysis on the results will show which model is best suited for the model.

An in-depth exploration of the process can be found in the report


