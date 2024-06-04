import _sqlite3 as sqlite3
import os
import pandas
import numpy as np
from sklearn.model_selection import train_test_split

# Load the given Sqlite file.
conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'database.sqlite'))
c = conn.cursor()

# Get the tablename from db
tablename = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()[0][0]

# Get the table columns from db
columns = conn.execute(f"PRAGMA table_info({tablename})").fetchall()
column_names = [c[1] for c in columns]
print(f"\nPrinting column names of table: {tablename}\n", ", ".join(column_names))

# Get feature names from column_names
feature_names = [n for n in [c[1] for c in columns] if "Cm" in n]
print(f"\nPrinting feature names of table: {tablename}\n", ", ".join(feature_names))

# Get distinct spieces from db
species = conn.execute(f"SELECT DISTINCT species FROM {tablename}").fetchall()
class_names = [s[0] for s in species]
print(f"\nPrinting all class names from table: {tablename}\n", ", ".join(class_names))


# Load data from the given Sqlite file.
dataframes = pandas.read_sql(f"SELECT * FROM {tablename}", conn)
# Close db connection
conn.close()
print("\nData loaded successfully. \n ----- DB Operations completed -----\n\n")

# Remove null values from dataset
dataframes = dataframes.dropna()

# Remove duplicates in the dataset
dataframes = dataframes.drop_duplicates(subset=column_names[:-1])

# Drop "ID" column as it is not required for the Statistical Calculations
dataframes = dataframes.drop(columns=column_names[0])

# Display Statistics of the data
print("Printing Statistics of the data: \n", dataframes.describe())

# Display number of samples of each Class to Check if the Dataset is balanced or not
""" Dataset is balanced, if all the classes have same number of samples """
print("\nPrinting number of samples of each Class to Check if the Dataset is balanced or not\n", dataframes[column_names[-1]].value_counts())

# Mean of each attribute for each class
mean_data = dataframes.groupby(column_names[-1]).mean()

# Standard deviation of each attribute for each class
stdev_data = dataframes.groupby(column_names[-1]).std()

print("\nPrinting Mean of each attribute for each class: \n", mean_data)
print("\nPrinting Standard Deviation of each attribute for each class: \n", stdev_data)

# Print the correlation matrix
correlation_matrix = dataframes.corr(numeric_only=True)
print("\nCorrelation Matrix: \n", correlation_matrix)

print("\n ----- Data analysis completed -----\n")


#######################################################################
# Calculate Gaussian Probablity Density Function
def predict_class(x_vect_data):
    # Initialize Predicted Class empty list
    y_pred = []

    # Calculate Prior Probabilities for each class
    prior = []
    for r in range(len(class_names)):
        pp = len(dataframes[dataframes[column_names[-1]] == class_names[r]]) / len(dataframes)
        prior.append(pp)

    for x in x_vect_data:
        # Initialize Likelihood Probabilities for each class as 1
        likelihood = [1.0] * len(class_names)

        for c in range(len(class_names)):
            for f in range(len(feature_names)):
                mean = mean_data[feature_names[f]][class_names[c]]
                stdev = stdev_data[feature_names[f]][class_names[c]]

                # Calculate Likelihood for each class
                likelihood[c] *= (1 / (np.sqrt(2 * np.pi) * stdev)) * np.exp(-((x[f]-mean)**2 / (2 * stdev**2)))

        # Initialize Posterior Probability for each class as 1
        posterior = [1.0] * len(class_names)
        for j in range(len(class_names)):
            posterior[j] = likelihood[j] * prior[j]

        y_pred.append(np.argmax(posterior))

    return np.array(y_pred)


#######################################################################
print("\nStarting Predictions for input data:")
# Predict the Class for the given test sample data
X_test = [[5.7, 2.9, 4.2, 1.3]]
Y_Pred = predict_class(X_test)

for c, X in enumerate(X_test):
    print(F"\nPredicted Class for Test Data {X}: {class_names[Y_Pred[c]]}")

# Split the dataset into Training Data and Testing Data
# We use 10% of the given Data as Testing Data
train_data, test_data = train_test_split(dataframes, test_size=0.1)
X_test = test_data.iloc[:, :-1].values

# Predict the Class for the given test sample data
y_prediction_list = predict_class(X_test)
for i, X in enumerate(X_test):
    print(F"Predicted Class for Test Data {X}: {class_names[y_prediction_list[i]]}")


# Predict the Class for the given test sample data
X_test = [[6.7, 3.3, 5.7, 2.5],     # Iris-virginica in csv file
          [6.0, 2.9, 4.5, 1.5],     # Iris-versicolor in csv file
          [6.2, 3.4, 5.4, 2.3],     # Iris-virginica in csv file
          [5.5, 4.2, 1.4, 0.2],     # Iris-setosa in csv file
          [6.0, 2.2, 4.0, 1.0],     # Iris-versicolor in csv file
          [4.9, 3.1, 1.5, 0.1],     # Iris-setosa in csv file
          [5.0, 3.2, 1.2, 0.2],     # Iris-setosa in csv file
          [5.5, 3.5, 1.3, 0.2]      # Iris-setosa in csv file
          ]

y_prediction_list = predict_class(X_test)
for i, X in enumerate(X_test):
    print(F"Predicted Class for Test Data {X}: {class_names[y_prediction_list[i]]}")

print("\n ----- End of PR Assignment3 Submitted by Dhaval S, from Cohort3")