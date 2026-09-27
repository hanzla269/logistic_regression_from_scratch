import numpy as np
import pandas as pd
df = pd.DataFrame({
    "Age": [22, 35, 28, 42, 19, 31, 25, 45, 29, 38, 21, 34, 27, 41, 30],

    "Gender": [
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male"
    ],

    "Device": [
        "Mobile", "Laptop", "Laptop", "Mobile", "Mobile",
        "Tablet", "Laptop", "Laptop", "Tablet", "Mobile",
        "Mobile", "Laptop", "Tablet", "Mobile", "Laptop"
    ],

    "Hours_Online": [2, 5, 4, 6, 1, 5, 3, 7, 2, 6, 1, 5, 3, 7, 4],

    "Purchased": [
        0, 1, 1, 1, 0,
        1, 0, 1, 0, 1,
        0, 1, 0, 1, 1
    ]
})
X = df[["Age", "Gender", "Device", "Hours_Online"]]
y = df["Purchased"].to_numpy()

X = pd.get_dummies(
    X,
    columns=["Gender", "Device"])
X = X.astype(int)
X = X.to_numpy()


print("Encoded X:")
print(X)

print("\nTarget y:")
print(y)
np.random.seed(42)

indices = np.arange(len(X))

np.random.shuffle(indices)

split = int(0.8 * len(X))

train_indices = indices[:split]
test_indices = indices[split:]


X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

class LogisticRegression:

    def __init__(self, learning_rate=0.01, iterations=10000):

        self.learning_rate = learning_rate
        self.iterations = iterations

        self.weights = None
        self.bias = None


    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))


    def loss(self, y, predictions):

        # Prevent log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)

        loss = -np.mean(
            y * np.log(predictions)
            + (1 - y) * np.log(1 - predictions)
        )
        return loss
    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        self.bias = 0
        for i in range(self.iterations):
          z = np.dot(X, self.weights) + self.bias
          predictions = self.sigmoid(z)
          dw = (1 / n_samples) * np.dot(
                X.T,
                (predictions - y))
          db = (1 / n_samples) * np.sum(
                predictions - y)
          self.weights = (
                self.weights
                - self.learning_rate * dw)
          self.bias = (
                self.bias
                - self.learning_rate * db)
          current_loss = self.loss(y,predictions)
          if i % 1000 == 0:
                print("Iteration:",i,"Loss:",current_loss)

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        return probabilities


    def predict(self, X):
      probabilities = self.predict_proba(X)
      predictions = (probabilities >= 0.5).astype(int)
      return predictions




model = LogisticRegression(
    learning_rate=0.01,
    iterations=10000
)



model.fit(X_train, y_train)


"""PREDICTION"""
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)


print("\nActual values:")
print(y_test)

print("\nPredicted values:")
print(y_pred)

print("\nPredicted probabilities:")
print(y_prob)


"""CONFUSION MATRIX"""

TN = np.sum(
    (y_test == 0) & (y_pred == 0)
)

FP = np.sum(
    (y_test == 0) & (y_pred == 1)
)

FN = np.sum(
    (y_test == 1) & (y_pred == 0)
)

TP = np.sum(
    (y_test == 1) & (y_pred == 1)
)


confusion_matrix = np.array([
    [TN, FP],
    [FN, TP]
])


print("\nConfusion Matrix:")
print(confusion_matrix)


"""ACCURACY"""
accuracy = (TP + TN) / (
    TP + TN + FP + FN
)


print("\nAccuracy:")
print(accuracy)
