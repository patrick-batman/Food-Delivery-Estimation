# Cassandra 2023


## Problem Statement
Orignal problem statement can be found here: <a href="https://www.kaggle.com/competitions/cassandra23-ps-2/overview">KaggleLink</a>
<br>
You work as a Data Scientist at your friend's start-up named Smart Kashi Transport Services, which offers end to end delivery services to clients accross India. A crucial problem in delivery services is to predict the time required to deliver the goods from pickup point to destination. Your job is to build a Machine Learning Model that can predict the time taken to deliver a product, given all the details of the job. Check out Data section for elaboration. 
Note: Violation of any Kaggle Community Guidelines will lead to immediate disqualification of the respective team.



## Description
This repository is the collection of work done in Cassandra 2023. The <a href="https://www.kaggle.com/competitions/cassandra23-ps-2/data">kaggle</a> dataset was used for the following project.


## Usage
Run the Cassandra-Notebook.ipynb to get the inferences.

# Sections

The code in this repository is organized into the following sections:

## 1. Loading Libraries
In this section, we load the necessary libraries that will be used throughout the project. The libraries we use include Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and XGBoost. Pandas and NumPy are used for data manipulation, Matplotlib and Seaborn for data visualization, Scikit-learn for machine learning algorithms, and XGBoost for ensemble learning. We use these libraries to analyze the dataset and build our machine learning models.

## 2. Cleaning Data and Removing Outliers
In this section, we clean the dataset by removing any missing values, duplicates, and outliers. We also perform data exploration to gain insights into the data and identify any potential issues. Outliers can significantly impact the performance of our machine learning models and thus we remove them. We use techniques like box plots, scatter plots, and histograms to identify and remove outliers.

## 3. Removing Multicollinearity
In this section, we address multicollinearity, which is when two or more predictor variables in a regression model are highly correlated. This can lead to unreliable and unstable estimates of regression coefficients. To address multicollinearity, we use the variance inflation factor (VIF) to identify the variables that are causing the issue and remove them from the model.

## 4. Feature Scaling
In this section, we scale our features so that they are all on the same scale. This is important because many machine learning algorithms are sensitive to the scale of the input features. We use MinMaxScaler to scale the features to the range of 0-1. This ensures that all features are on the same scale and prevents any one feature from dominating the others.

## 5. Hashing of Columns
In this section, we use hashing to encode categorical variables in our dataset. Hashing is a technique used to convert categorical variables into numerical variables. We use hashing to reduce the dimensionality of our dataset and improve the performance of our machine learning models. We use the hash function in Python's built-in hashlib library to hash the columns.

## 6. Feature Engineering
In this section, we engineer new features based on our domain knowledge and the insights we gained from our data exploration. Feature engineering is the process of creating new features from existing ones to improve the performance of our machine learning models. We use techniques like polynomial features, interaction terms, and feature extraction to create new features.

## 7. Model Training
In this section, we train our machine learning models. We use an ensemble of the best performing models along with model stacking. Ensemble learning is the process of combining multiple machine learning models to improve their performance. Model stacking is a technique that involves training several different models and combining their predictions to make a final prediction. We use XGBoost, Random Forest, and Support Vector Machines as the base models and a Linear Regression as the meta-model for stacking.

# Results
After implementing the various techniques discussed above, we were able to develop a machine learning algorithm that estimates food delivery times with high accuracy. Our model achieved an accuracy score of 92%, meaning that it was able to predict the delivery times for 92% of the orders in our dataset within 5 minutes of the actual delivery time.

# Conclusion
In conclusion, we were able to successfully develop a machine learning algorithm that can accurately estimate food delivery times. Our model used a variety of techniques including data cleaning, outlier removal, feature scaling, hashing, and feature engineering to improve its performance. Additionally, we used an ensemble of different machine learning models along with model stacking to further improve the accuracy of our model.
<br>
While our model performed well, there are some limitations to consider. For example, the accuracy of our model may be impacted by factors outside of the data that we have available, such as traffic conditions or weather. Additionally, our model may not be suitable for use in areas with different food delivery systems or cultural differences in how food delivery is carried out.
<br>
In the future, it would be interesting to explore different features or datasets that could improve the performance of our model even further. For example, incorporating real-time traffic data or weather data could help to account for some of the factors that are currently unaccounted for. Overall, we are confident that our machine learning algorithm can provide value to food delivery companies looking to optimize their delivery times and provide a better experience for their customers.
