import numpy as np
from scipy.fftpack import dct

class FrequencyEstimator():

    """
        INFO ON PARAMETERS
        length is the length of the DCT (important if input is of varying length. Set based on the length of you input)
        if n < len(input) the input is truncated if n > len(input) input is zero-padded. If you don't want to lose information we recommend a higher value for length
        if a float value is passed it will be converted to an integer
        !!!Length is not the same with reduciton_rate. Length may shorten/pad input, while n_components reduces the resulted DCT signal

        n_components will reduce the DCT by a given number of points
        frequency_selection decides which best n_components will be preserved (high or low)
        if a float value is passed it will be converted to an integer

        if normalization is set to true DCT will use norm='ortho'

        estimators can be one element or more. If more are passed the decission will be based on the average of each estimator. Defaul gaussian
    """

    _ESTIMATORS = {"gaussian", "naive_bayes"}

    def __init__(self, length=None, n_components=None, frequency_selection=None, normalization=False, estimators = None, reg=1e-2):
        
        if length is not None: 
            if not isinstance(length, (int, float, np.integer, np.floating)) or length <= 0 or not np.isfinite(length):
                raise TypeError("length must be a finite positive integer, float None.")
        if n_components is not None:
            if not isinstance(n_components, (int, float)) or n_components <= 0 or not np.isfinite(n_components):
                raise TypeError("n_components must be a finite positive number or None.")
        if not isinstance(normalization, (bool, np.bool_)):
            raise TypeError("normalization must be a boolean.")
        if isinstance(reg, (bool, np.bool_)) or not np.isfinite(reg) or reg <= 0: 
            raise ValueError("reg must be a finite positive number.")
        if frequency_selection is None:
            frequency_selection = "low"
        else:
            if not isinstance(frequency_selection, str):
                raise TypeError("frequency_selection must be a string.")
            if frequency_selection not in("low", "high"):
                raise TypeError("Unsupported frequency_selection. Only use None, low or high.")
        if estimators is None:
            selected_estimators = {"gaussian"}
        else:
            if not isinstance(estimators, (set, frozenset)):
                raise TypeError("estimators must be a set or frozentset of strings.")
            if not all(isinstance(estimator, str) for estimator in estimators):
                raise TypeError("estimators must contain only strings.")
            selected_estimators = set(estimators)
        if not selected_estimators:
            raise ValueError("At least one estimator must be selected.")
        unsupported_estimators = selected_estimators - self._ESTIMATORS

        if unsupported_estimators:
            raise TypeError("Unsupported estimators.")

        self._length = int(length) if length is not None else None
        self._n_components = int(n_components) if n_components is not None else None
        self._frequency_selection = str(frequency_selection)
        self._normalization = bool(normalization)
        self._estimators = frozenset(selected_estimators)
        self._reg=float(reg)

        self._means = {}
        self._covariances = {}
        self._classes = None

        self._class_priors = {}
        self._parameters = {}
        self._feature_names = None
        self._n_features = None

        self._n_coefficients = {} # each feature can have a different number of coefficients

    def transform_to_dct(self, X):
        signal = np.asarray(X, dtype=float)
        for axis in range(1, signal.ndim):
            signal = dct(signal, axis=axis, n=self._length if self._length else None, type=2, norm='ortho' if self._normalization else None)
        return signal

    
    def _select_dct_coefficients(self, feature_dct):
        if self._n_components is None:
            return feature_dct.reshape(feature_dct.shape[0], -1)

        spatial_shape = feature_dct.shape[1:]

        if self._n_components > np.prod(spatial_shape):
            raise ValueError("n_components exceeds available coefficients.")

        # Calculate each coefficient's squared frequency distance.
        grids = np.meshgrid(*[np.arange(n) for n in spatial_shape], indexing="ij")
        frequency = sum(grid**2 for grid in grids).ravel()

        # Select the N lowest-frequency coefficients.
        if self._frequency_selection == "low":
            indices = np.argsort(frequency, kind="stable")[:self._n_components]
        # Select the N highest-frequency coefficients.
        elif self._frequency_selection == "high":
            indices = np.argsort(frequency, kind="stable")[-self._n_components:]

        return feature_dct.reshape(feature_dct.shape[0], -1)[:, indices]


    def _prepare_features(self, X, feature_order=None):

        if isinstance(X, dict):
            if not X:
                raise ValueError("X must contain at least one feature.")
            raw_features = dict(X)
        else:
            try:
                X = np.asarray(X, dtype=float)
            except (TypeError, ValueError) as e:
                raise TypeError("X must contain numerical data.") from e
            if X.ndim != 3:
                raise ValueError("Array input must have shape (n_samples, n_features, signal_length).\nFor other representation shapes, use a dictionary.")
            if any(size == 0 for size in X.shape):
                raise ValueError("X must not contain empty dimensions.")
            if not np.all(np.isfinite(X)):
                raise ValueError("X must contain only finite values.")

            raw_features = {feature: X[:, feature, :] for feature in range(X.shape[1])}

        if feature_order is not None:
            if (len(raw_features) != len(feature_order) or set(raw_features) != set(feature_order)):
                raise ValueError("Prediction features must match the training feature names.")

            raw_features = {feature: raw_features[feature] for feature in feature_order}

        X_dct = {}
        n_samples = None

        for feature, values in raw_features.items():
            try:
                values = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Feature {feature} must contain numerical data.") from e
            if values.ndim < 2:
                raise ValueError(f"Feature {feature} must have shape (n_samples, representation dimensions).")
            if any(size == 0 for size in values.shape):
                raise ValueError(f"Feature {feature} must not have empty dimensions.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Feature {feature} must contain only finite values.")
            if n_samples is None:
                n_samples = values.shape[0]
            elif values.shape[0] != n_samples:
                raise ValueError("All features must contain the same number of samples.")

            feature_dct = self.transform_to_dct(values)
            feature_dct = self._select_dct_coefficients(feature_dct)

            X_dct[feature] = feature_dct

        return X_dct

    def fit(self, X, y):

        X_dct = self._prepare_features(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array.")

        n_samples = next(iter(X_dct.values())).shape[0]

        if n_samples != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if len(y) == 0:
            raise ValueError("Cannot fit on an empty dataset.")

        self._classes, counts = np.unique(y, return_counts=True)
        self._class_priors = {c: np.log(count / len(y)) for c, count in zip(self._classes, counts)}

        self._feature_names = tuple(X_dct.keys())
        self._n_features = len(self._feature_names)
        self._n_coefficients = {feature: X_dct[feature].shape[1] for feature in self._feature_names}
        self._parameters = {estimator:{} for estimator in self._estimators}

        for estimator in self._estimators:
            for feature in self._feature_names:
                self._parameters[estimator][feature] = {}
                feature_data = X_dct[feature]

                for c in self._classes:

                    X_c = feature_data[y == c]
                    mean = np.mean(X_c, axis=0)
                    parameters = {"mean": mean}

                    if estimator == "gaussian":
                        centered = X_c - mean
                        covariance = (centered.T @ centered) / len(X_c)
                        parameters["covariance"] = covariance

                    elif estimator == "naive_bayes":
                        variance = np.var(X_c, axis=0, ddof=0)
                        parameters["variance"] = variance

                    self._parameters[estimator][feature][c] = parameters

        return self

    def _gaussian_log_likelihood(self, x, parameters):
        mean = parameters["mean"]
        diff = x - mean
        covariance = parameters["covariance"] + self._reg * np.eye(len(mean))
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("Regularized covariance is not positive.")
        
        solved = np.linalg.solve(covariance, diff)
        quadratic = diff @ solved
        log_likelihood = -0.5 * (quadratic + logdet + len(mean) * np.log(2 * np.pi))
        return log_likelihood

    def _naive_bayes_log_likelihood(self, x, parameters):
        mean = parameters["mean"]
        variance = parameters["variance"] + self._reg
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + (x - mean) ** 2 / variance)
        return log_likelihood

    def _estimator_log_scores(self, X_dct, estimator):
        n_samples = next(iter(X_dct.values())).shape[0]
        n_classes = len(self._classes)
        scores = np.zeros((n_samples, n_classes))

        for sample_index in range(n_samples):
            for class_index, c in enumerate(self._classes):
                total = self._class_priors[c] # begin with class priors

                for feature in self._feature_names:
                    x_feature = X_dct[feature][sample_index]
                    parameters = self._parameters[estimator][feature][c]

                    if estimator == "gaussian":
                        log_likelihood = self._gaussian_log_likelihood(x_feature, parameters)
                    else:
                        log_likelihood = self._naive_bayes_log_likelihood(x_feature, parameters)

                    total += log_likelihood
                scores[sample_index, class_index] = total
        return scores

    @staticmethod
    def _softmax(scores):
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)

        return probabilities / np.sum(probabilities, axis=1, keepdims=True)

    def _predict_probabilities(self, X):
        if self._classes is None:
            raise RuntimeError("Call fit() before prediction!")

        X_dct = self._prepare_features(X, feature_order=self._feature_names)

        for feature in self._feature_names:
            if X_dct[feature].shape[1] != self._n_coefficients[feature]:
                raise ValueError(f"Feature {feature} has a different number DCT coefficients compared to training data.")

        estimator_probabilities = []

        for estimator in self._estimators:
            log_scores = self._estimator_log_scores(X_dct, estimator)
            probabilities = self._softmax(log_scores)
            estimator_probabilities.append(probabilities)

        return np.mean(estimator_probabilities, axis=0)

    def predict_proba(self, X):
        return self._predict_probabilities(X)

    def predict(self, X):
        probabilities = self._predict_probabilities(X)
        return self._classes[np.argmax(probabilities, axis=1)]
