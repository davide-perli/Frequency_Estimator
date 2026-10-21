import numpy as np, pandas as pd
from skimage.feature import hog
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
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
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar10.load_data()

y_train_raw = y_train_raw.ravel()
y_test_raw = y_test_raw.ravel()

X_train = X_train_raw.astype(np.float32) / 255.0
X_test = X_test_raw.astype(np.float32) / 255.0

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
    n_components=50,
    frequency_selection="low",
    normalization=False,
    estimators={"gaussian"},
    reg=1e-2
)

model.fit(X_train, y_train_raw)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_raw, y_pred)

print(f"Accuracy (%): {accuracy * 100:.2f}%")
print("Balanced accuracy:", balanced_accuracy_score(y_test_raw, y_pred))

# Accuracy (%): 44.26%
# Balanced accuracy: 0.4426

# CIFAR-10 WITH HISTOGRAM REPRESENTATION

def extract_hog(images):
    features = []

    for image in images:
        image = image.astype(np.float32) / 255.0

        feat = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=-1,
        )
        features.append(feat)

    return {"hog": np.asarray(features, dtype=np.float32)}

X_train_hog = extract_hog(X_train_raw)
X_test_hog = extract_hog(X_test_raw)

model = FrequencyEstimator(
    length=None,
    n_components=None,
    normalization=True,
    estimators={"gaussian"},
    reg=1e-3,
)

model.fit(X_train_hog, y_train_raw)

pred = model.predict(X_test_hog)
print(f"HOG Accuracy: {accuracy_score(y_test_raw, pred) * 100:.2f}")
print("HOG Balanced accuracy:", balanced_accuracy_score(y_test_raw, pred))

# HOG Accuracy: 55.02
# HOG Balanced accuracy: 0.5502

# ECG Heartbeat Categorization Dataset

train = pd.read_csv("mitbih_train.csv", header=None).to_numpy()
test = pd.read_csv("mitbih_test.csv", header=None).to_numpy()

X_train = train[:, :-1].astype("float32")
y_train = train[:, -1].astype(int)

X_test = test[:, :-1].astype("float32")
y_test = test[:, -1].astype(int)

X_train = {"ecg": X_train[:, None, :]}
X_test = {"ecg": X_test[:, None, :]}

model = FrequencyEstimator(
    length=187,
    normalization=True,
    estimators={"naive_bayes"},
    reg=0.27

    # Accuracy: 0.5032888726475425
    # Balanced accuracy: 0.48380102404681813
)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, zero_division=0))