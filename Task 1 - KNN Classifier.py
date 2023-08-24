#loading necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing scikit learn library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Converting data into a pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#including target
df['target'] = iris.target

# Splitting the data into training and testing sets
indV_train, indV_test, depV_train, depV_test = train_test_split(
    iris.data, iris.target, test_size=0.20, random_state=4
)

# Normalizing the data
scaler = StandardScaler()
indV_train_scaled = scaler.fit_transform(indV_train)
indV_test_scaled = scaler.transform(indV_test)

# Creating KNN classifier with optimized hyperparameters
knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan')  # Adjust parameters

# Fitting the model on the training data
knn.fit(indV_train_scaled, depV_train)

# Predicting the target labels for the test set
y_predict = knn.predict(indV_test_scaled)

# Evaluating the model
accuracy = accuracy_score(depV_test, y_predict)
print("\nAccuracy of the Model is:", accuracy)
print("\nClassification Report:\n", classification_report(depV_test, y_predict))
print("\nConfusion Matrix:\n", confusion_matrix(depV_test, y_predict))

#testing the model

#getting input from user for sepal length, sepal width, petal length and petal width
print("\n\nTesting The Model With User Input:\n")
sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

#example test data - input user
#sepal_length = 5.1
#sepal_width = 3.5
#petal_length = 1.4
#petal_width = 0.2
#result - setosa

#converting user input into an array
user_input = [sepal_length, sepal_width, petal_length, petal_width]
print(user_input)

#creating a dataframe for user input
user_input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])

#normalizing user input
user_input_scaled = scaler.transform(user_input_df)

#predict using user input
predicted_class = knn.predict(user_input_scaled)
predicted_class_name = iris.target_names[predicted_class][0]

#printing the result obtained
print("The predicted class for the given user input is:", predicted_class_name)