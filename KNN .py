from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

x = iris.data
y = iris.target

#print(x)
#print(y)
# print(iris.target_names[y])
# use this to display target names instead of their binary values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)  # checking target names of the 3 nearest plots
knn.fit(x_train, y_train)  # fitting the model using the training data

y_pred = knn.predict(x_test)
# using the predict function predict the output for the test data and store in y_predict

from sklearn import metrics

# used to predict accuracy by comparing the test value and prediction value
print("knn model accuracy: ", metrics.accuracy_score(y_test, y_pred))

sample = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]
# giving a new data and checking the prediction
preds = knn.predict(sample)

pred_species = [iris.target_names[p] for p in preds] # guess 'p' is like the 'i' that we use in for-loops commonly
# the model returns the prediction in binary form..
# the above statement converts the binary values into it's corresponding names of the target
# comment out the above line to know more
print("predictions: ", pred_species)
