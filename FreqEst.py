import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, roc_curve
from scipy.fftpack import dct

class TransformToFrequency():
    def __init__(self):
        pass

class FrequencyEstimator():

    def __init__(self, input, length=None, reduction_rate=False, normalization=False, estimators: set[str] | None = None):

        """
            INFO ON PARAMETERS
            length is the length of the DCT (important if input is of varying length. Set based on the length of you input)
            if n < len(input) the input is truncated if n > len(input) input is zero-padded. If you don't want to lose information we recommend a higher value for length
            if a float value is passed it will be converted to an integer
            !!!Length is not the same with reduciton_rate. Length may shorten/pad input, while reduction_rate reduces the resulted DCT signal

            reduction_rate will reduce the DCT by a given number of points
            if a float value is passed it will be converted to an integer

            if normalization is set to true DCT will use norm='ortho'

            estimators can be one element or more. If more are passed the decission will be based on the average of each estimator. Defaul gaussian
        """

        # INPUT NOT VALIDATED YET
        # if not isinstance(input, np.ndarray):
        #     raise TypeError("input must be an numpy.ndarray!")
        if length is not None and not isinstance(length, (int, float)):
            raise TypeError("length must be an integer, float None!")
        if reduction_rate is not None and not isinstance(reduction_rate, (int, float)):
            raise TypeError("reduction_rate must be an integer, float, or None!")
        if not isinstance(normalization, bool):
            raise TypeError("normalization must be a boolean!")
        if estimators is not None and not isinstance(estimators, set):
            raise TypeError("Estimators must be a set!")
        if estimators is not None and not all(isinstance(value, str) for value in estimators):
            raise TypeError("Values for estimator must be strings!")

        self._input = input
        self._length = int(length) if length is not None else None
        self._reduction_rate = int(reduction_rate) if reduction_rate is not None else None
        self._normalization = normalization
        self._estimators = frozenset({"gaussian"} if estimators is None else set(estimators))

    def transform_to_dct(self):
        signal = dct(self._input, axis=-1, n=self._length if self._length else None, type=2, norm='ortho' if self._normalization else None)
        plt.plot(signal)
        plt.xlabel("DCT coefficient")
        plt.ylabel("Magnitude")
        plt.show()
        return signal

    def _retude_dct():
        pass

    def fit():
        pass

    def evaluate():
        pass


# input = [1, 5, 8, 0, 13, -5, 4]
# input_array = np.asarray(input)
# print(type(input_array))
# obj = FrequencyEstimator(input_array)
# obj.transform_to_dct()