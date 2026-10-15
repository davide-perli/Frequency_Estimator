import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from FreqEst import FrequencyEstimator

X = {
    "feature_0": np.asarray([
        [1, 5, 8, 0, 13, -5, 4],
        [2, 4, 7, 1, 12, -4, 3],
        [8, 2, 1, 9, -2, 5, 0],
    ]),

    "feature_1": np.asarray([
        [2, 4, 7, 1, 12, -4, 3],
        [1, 5, 8, 0, 13, -5, 4],
        [3, 7, 2, 8, -1, 4, 6],
    ]),
}

X_test = {
    "feature_0": np.asarray([
        [1, 5, 8, 0, 13, -5, 4],
    ]),

    "feature_1": np.asarray([
        [2, 4, 7, 1, 12, -4, 3],
    ]),
}

y = np.asarray([
    "TRUE",
    "TRUE",
    "FALSE",
])

model = FrequencyEstimator(
    length=7,
    normalization=True,
    estimators={"gaussian", "naive_bayes"}
)

model.fit(X, y)

print("Classes:", model._classes)
print("Predictions:", model.predict(X_test))
print("Probabilities:", model.predict_proba(X_test))


# CIFAR-10 example
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = y_train.ravel()
y_test = y_test.ravel()

X_train, _, y_train, _ = train_test_split(
    X_train,
    y_train,
    train_size=49000,
    stratify=y_train,
    random_state=42,
)


X_test, _, y_test, _ = train_test_split(
    X_test,
    y_test,
    train_size=1000,
    stratify=y_test,
    random_state=42,
)

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

X_train = {
    "red": X_train[..., 0],
    "green": X_train[..., 1],
    "blue": X_train[..., 2],
}

X_test = {
    "red": X_test[..., 0],
    "green": X_test[..., 1],
    "blue": X_test[..., 2],
}

model = FrequencyEstimator(
    length=32,
    n_components=34,
    normalization=True,
    estimators={"gaussian"},
    reg=0.04
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(f"Accuracy (%): {accuracy * 100:.2f}%")
