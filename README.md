# Frequency Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/MIT-License-red.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)

`frequency-classifier` is a small numerical classification package built around
the discrete cosine transform (DCT). It converts each numerical feature
representation into frequency-domain coefficients and classifies samples using
statistical likelihood models.

The project is implemented in [`FreqEst.py`](./FreqEst.py) and exposes one
public class:

```python
from FreqEst import FrequencyEstimator
```

## Table of contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Windows PowerShell](#windows-powershell)
  - [Linux or macOS](#linux-or-macos)
- [Input format](#input-format)
  - [Dictionary input (recommended)](#dictionary-input-recommended)
  - [Three-dimensional array input](#three-dimensional-array-input)
  - [Labels](#labels)
- [`FrequencyEstimator` parameters](#frequencyestimator-parameters)
  - [`length`](#length)
  - [`n_components`](#n_components)
  - [`frequency_selection`](#frequency_selection)
  - [`normalization`](#normalization)
  - [`estimators`](#estimators)
  - [`reg`](#reg)
- [Basic usage](#basic-usage)
  - [Multiple representations](#multiple-representations)
- [Transforming data manually](#transforming-data-manually)
- [Running tests](#running-tests)
- [Examples](#examples)
- [Project layout](#project-layout)
- [License](#license)

## Overview

For each feature, `FrequencyEstimator`:

1. validates the input values and sample dimensions;
2. applies a type-II DCT along every axis after the sample axis;
3. optionally truncates or zero-pads every transformed axis to `length`;
4. optionally keeps only the lowest-frequency `n_components` coefficients;
5. estimates class-conditional parameters during `fit`;
6. calculates class probabilities during `predict_proba`;
7. returns the most probable class during `predict`.

The estimator supports multiple feature representations with different
dimensions in the same model. For example, one sample can contain a 1-D
numeric signal, a 2-D image, and a multi-channel time series.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Pytest (only required for running the test suite)

## Setup

Create and activate a virtual environment, then install the package in
editable mode:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The editable install makes the local `FreqEst` module importable while
developing. If the package is already installed, the runtime dependencies can
also be installed directly with:

```bash
python -m pip install numpy scipy
```

## Input format

### Dictionary input (recommended)

Pass a dictionary whose values are numerical arrays. Each value must have the
shape:

```text
(n_samples, representation_dimension_1, representation_dimension_2, ...)
```

Every feature must have the same first dimension (`n_samples`), but the
remaining dimensions may differ between features. At least one feature is
required.

Examples of valid feature values:

```python
X = {
    "numeric_signal": np.ndarray(shape=(n_samples, signal_length)),
    "image": np.ndarray(shape=(n_samples, height, width)),
    "time_series": np.ndarray(shape=(n_samples, time_steps, channels)),
}
```

The feature keys used during prediction must exactly match the keys used during
training.

### Three-dimensional array input

For compatibility with a conventional multi-feature representation, `X` may
also be a single array with shape:

```text
(n_samples, n_features, signal_length)
```

Each feature axis is treated as a separate feature. Use dictionary input when
representations have different dimensions or when meaningful feature names are
needed.

### Labels

`y` must be a one-dimensional array with exactly one label per sample. Labels
may be strings, integers, or other values supported by NumPy's
`unique`/indexing operations.

Inputs must contain finite numerical values. Raw text, empty arrays, mismatched
sample counts, and incorrectly shaped labels are rejected.

## `FrequencyEstimator` parameters

```python
FrequencyEstimator(
    length=None,
    n_components=None,
    frequency_selection=None,
    normalization=False,
    estimators=None,
    reg=1e-2,
)
```

### `length`

- Type: positive finite integer or float, or `None`
- Default: `None`

The DCT length applied to every non-sample axis. If it is smaller than an
input axis, that axis is truncated. If it is larger, the input is zero-padded.
Floating-point values are converted to integers. `None` preserves each
original axis length.

`length` controls the transform size; it is not the same as
`n_components`, which controls how many transformed coefficients are retained.

### `n_components`

- Type: positive finite integer or float, or `None`
- Default: `None`

The number of DCT coefficients retained for each feature. The coefficients
with the lowest/highest (depending on `frequency_selection` parameter) multidimensional frequency distance are selected, so this
generally retains the low-frequency content. If `None`, all coefficients are
kept. The value cannot exceed the number of available coefficients after the
transform, and floating-point values are converted to integers.

### `frequency_selection`

- Type: string containing `"high"` or `"low"` or `None`
- Default: `low`

When `low`, keep the DCT coefficients with the lowest multidimensional frequency distance.
When `high`, keep the DCT coefficients with the highest multidimensional frequency distance.

### `normalization`

- Type: boolean
- Default: `False`

When `True`, the DCT uses SciPy's orthonormal normalization (`norm="ortho"`).
When `False`, the unnormalized type-II DCT is used.

### `estimators`

- Type: a non-empty `set` or `frozenset` containing `"gaussian"`/`"fast_gaussian"` and/or
  `"naive_bayes"`
- Default: `{"gaussian"}`

`"gaussian"` uses a full covariance matrix for each feature and class.
`"fast_gaussian"` doesn't calculate covaraince matrix and uses only the variance which means the model ignores correlations between DCT coefficients
`"naive_bayes"` uses independent coefficient variances (a diagonal
covariance). If both(`"gaussian"`/`"fast_gaussian"` and `"naive_bayes"`) are selected, the model calculates probabilities with
both estimators and averages the resulting probability matrices. Techically all 3 can be chosen, although having both `"gaussian"` and `"fast_gaussian"` is not recommended since either the DCT coefficients are correlated or not.

### `reg`

- Type: positive finite number, excluding booleans
- Default: `0.01`

Regularization added to covariance matrices and variances during prediction.
It helps keep likelihood calculations numerically stable, especially with
small datasets or correlated coefficients.

## Basic usage

```python
import numpy as np
from FreqEst import FrequencyEstimator

X = {
    "signal": np.array([
        [0.9, 1.0, 1.1, 1.0],
        [1.2, 0.8, 1.0, 1.1],
        [-1.0, -0.9, -1.1, -1.0],
        [-0.8, -1.2, -1.0, -0.9],
    ])
}
y = np.array(["positive", "positive", "negative", "negative"])

model = FrequencyEstimator(
    normalization=True,
    n_components=3,
    estimators={"gaussian", "naive_bayes"},
    reg=0.1,
)

model.fit(X, y)

X_new = {"signal": np.array([[1.0, 0.9, 1.1, 1.0]])}
print(model.predict(X_new))
print(model.predict_proba(X_new))
```

`fit` returns the fitted estimator, so calls can also be chained:

```python
model = FrequencyEstimator(normalization=True).fit(X, y)
```

### Multiple representations

```python
rng = np.random.default_rng(42)
X = {
    "numeric": rng.normal(size=(20, 6)),
    "image": rng.normal(size=(20, 4, 4)),
    "time_series": rng.normal(size=(20, 10, 3)),
}
y = np.array(["A"] * 10 + ["B"] * 10)

model = FrequencyEstimator(
    length=4,
    n_components=8,
    normalization=True,
    estimators={"gaussian", "naive_bayes"},
)
model.fit(X, y)

probabilities = model.predict_proba(X)
predictions = model.predict(X)
```

`predict_proba` returns an array of shape
`(n_samples, n_classes)`. Its columns follow the class order stored by
`np.unique` during fitting. `predict` returns a one-dimensional array of
predicted labels.

## Transforming data manually

The DCT step can be used independently:

```python
model = FrequencyEstimator(normalization=True)
transformed = model.transform_to_dct(
    np.arange(24, dtype=float).reshape(2, 3, 4)
)
```

The first axis is interpreted as the sample axis and is left unchanged. The
DCT is applied sequentially to every remaining axis. With `normalization=True`
this is equivalent to:

```python
from scipy.fftpack import dct

expected = dct(data, axis=1, type=2, norm="ortho")
expected = dct(expected, axis=2, type=2, norm="ortho")
```

## Running tests

From the repository root, with the virtual environment activated, run:

```bash
python -m pytest
```

The test suite is in [`tests/`](./tests/):

- [`test_numerical.py`](./tests/test_numerical.py) covers DCT behavior,
  individual and multi-feature representations, predictions, probabilities,
  and legacy 3-D array input.
- [`test_validation.py`](./tests/test_validation.py) covers invalid
  constructor values, malformed features and labels, mismatched dimensions,
  prediction-before-fit behavior, and feature-name compatibility.

Useful focused commands:

```bash
python -m pytest tests/test_numerical.py
python -m pytest tests/test_validation.py
python -m pytest -q
```

## Examples

Some examples are presented in this Readme file, but there are 3 more in the [`example.py`](./example.py). A basic numerical example, one using the CIFAR-10 dataset and the last one for classifying ECG from Kaggle [`link`](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

## Project layout

```text
.
├── FreqEst.py
├── pyproject.toml
├── README.md
└── tests/
    ├── test_numerical.py
    └── test_validation.py
```

`main.py` and `classified_cookies.csv` are not part of the documented
library workflow. The package is used through `FrequencyEstimator` and the
test suite rather than through a command-line entry point.

## License

This project is licensed under the MIT License. See
[`LICENSE.txt`](./LICENSE.txt) for the full license text.