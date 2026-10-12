import numpy as np
from scipy.fftpack import dct

_ESTIMATORS = {"gaussian", "naive_bayes"}

class FrequencyEstimator():

    def __init__(self, length=None, n_components=None, normalization=False, estimators = None, reg=1e-2):

        """
            INFO ON PARAMETERS
            length is the length of the DCT (important if input is of varying length. Set based on the length of you input)
            if n < len(input) the input is truncated if n > len(input) input is zero-padded. If you don't want to lose information we recommend a higher value for length
            if a float value is passed it will be converted to an integer
            !!!Length is not the same with reduciton_rate. Length may shorten/pad input, while n_components reduces the resulted DCT signal

            n_components will reduce the DCT by a given number of points
            if a float value is passed it will be converted to an integer

            if normalization is set to true DCT will use norm='ortho'

            estimators can be one element or more. If more are passed the decission will be based on the average of each estimator. Defaul gaussian
        """
        
        if length is not None: 
            if not isinstance(length, (int, float, np.integer, np.floating)) or length <= 0 or not np.isfinite(length):
                raise TypeError("length must be a finite positive integer, float None!")
        if n_components is not None:
            if not isinstance(n_components, (int, float)) or n_components <= 0 or not np.isfinite(n_components):
                raise TypeError("n_components must be a finite positive number or None!")
        if not isinstance(normalization, bool):
            raise TypeError("normalization must be a boolean!")
        if estimators is not None and not isinstance(estimators, set):
            raise TypeError("Estimators must be a set!")
        if isinstance(reg, (bool, np.bool_)) or not np.isfinite(reg) or reg <= 0: 
            raise ValueError("reg must be a finite positive number!")
        if estimators is None:
            selected_estimators = {"gaussin"}
        else:
            if not isinstance(estimators, (set, frozenset)):
                raise TypeError("estimators must be a set or frozentset of strings!")
            if not all(isinstance(estimator, str) for estimator in estimators):
                raise TypeError("estimators must contain only strings!")
            selected_estimators = set(estimators)
        unsupported_estimators = selected_estimators - self._ESTIMATORS

        if unsupported_estimators:
            raise TypeError("Unsupported estimators!")


        self._length = int(length) if length is not None else None
        self._n_components = int(n_components) if n_components is not None else None
        self._normalization = normalization
        self._estimators = frozenset(selected_estimators)
        self._reg=reg
        self._means = {}
        self._covariances = {}
        self._classes = None

    def transform_to_dct(self, X):
        signal = dct(X, axis=-1, n=self._length if self._length else None, type=2, norm='ortho' if self._normalization else None)
        return signal

    def _prepare_features(self, X):
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError("X must have shape\n(n_samples, n_features, signal_length)")
        if any(size == 0 for size in X.shape):
            raise ValueError("X must not contain empty dimensions.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        X_dct = self.transform_to_dct(X)

        if self._n_components is not None:
            if self._n_components > X_dct.shape[-1]:
                raise ValueError("n_components exceeds the available coefficient count.")
            X_dct = X_dct[:, :, :self._n_components]

        return X_dct

    def fit(self, X, y):

        X_dct = self._prepare_features(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional label array.")

        if X_dct.shape[0] != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if len(y) == 0:
            raise ValueError("Cannot fit on an empty dataset.")

        self._classes = np.unique(y)

        n_features = X_dct.shape[1]
        self._means = {feature:{} for feature in range(n_features)}
        self._covariances = {feature:{} for feature in range(n_features)}

        for feature in range(n_features):

            feature_data = X_dct[:, feature, :]

            for c in self._classes:

                X_c = feature_data[y == c]
                mean = np.mean(X_c, axis=0)
                centered = X_c - mean

                covariance = (centered.T @ centered) / len(X_c)

                self._means[feature][c] = mean
                self._covariances[feature][c] = covariance

        return self

    def _feature_log_likelihood(self, x, feature, c):
        mean = self._means[feature][c]
        diff = x - mean
        covariance = self._covariances[feature][c] + self._reg * np.eye(len(mean))
        sign, logdet = np.linalg.slogdet(covariance)
        inverse = np.linalg.inv(covariance)
        log_likelihood = -0.5 * (diff.T @ inverse @ diff + logdet + len(mean) * np.log(2 * np.pi))

        # print(f"\nClass: {c}, Feature: {feature}") 
        # print("Mean:", mean) 
        # print("Covariance diagonal:", np.diag(covariance)) 
        # print("Sign:", sign) 
        # print("Log determinant:", logdet) 
        # print("Log likelihood:", log_likelihood)
        return log_likelihood


    def predict(self, X):

        X_dct = self._prepare_features(X)
        predictions = []

        for x in X_dct:

            likelihoods = {}

            for c in self._classes:

                total_log_likelihood = 0

                for feature in range(X_dct.shape[1]):
                
                    feature_value = x[feature]
                    feature_log_likelihood = self._feature_log_likelihood(feature_value, feature, c)
                    total_log_likelihood += feature_log_likelihood
                likelihoods[c] = total_log_likelihood

            print("Classes:", self._classes)
            print("Likelihoods:", likelihoods)
            predictions.append(max(likelihoods, key=likelihoods.get))

        return np.asarray(predictions)

    def evaluate():
        pass


X = np.asarray([
    [
        [1, 5, 8, 0, 13, -5, 4],   # feature 0
        [2, 4, 7, 1, 12, -4, 3],   # feature 1
    ],
    [
        [2, 4, 7, 1, 12, -4, 3],   # feature 0
        [1, 5, 8, 0, 13, -5, 4],   # feature 1
    ],
    [
        [8, 2, 1, 9, -2, 5, 0],    # feature 0
        [3, 7, 2, 8, -1, 4, 6],    # feature 1
    ],
])
y = np.asarray([
    "TRUE",
    "TRUE",
    "FALSE",
])

model = FrequencyEstimator(
    length=7,
    normalization=True
)

model.fit(X, y)

prediction = model.predict(
    np.asarray([[
    [1, 5, 8, 0, 13, -5, 4],
    [2, 4, 7, 1, 12, -4, 3]
]])
)

print(prediction)