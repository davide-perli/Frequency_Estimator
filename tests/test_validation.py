import numpy as np
import pytest

from FreqEst import FrequencyEstimator


def make_model(**kwargs):
    return FrequencyEstimator(
        estimators={"naive_bayes"},
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0},
        {"length": -1},
        {"n_components": 0},
        {"n_components": -1},
        {"reg": 0},
        {"reg": -1},
        {"reg": np.nan},
        {"estimators": {"unknown_estimator"}},
    ],
)
def test_invalid_constructor_parameters(kwargs):
    with pytest.raises((TypeError, ValueError)):
        make_model(**kwargs)


def test_rejects_empty_feature_dictionary():
    model = make_model()

    with pytest.raises(ValueError):
        model.fit({}, np.array(["A", "B"]))


def test_rejects_mismatched_sample_counts():
    model = make_model()

    X = {
        "first": np.ones((4, 3)),
        "second": np.ones((3, 2)),
    }

    with pytest.raises(ValueError):
        model.fit(X, np.array(["A", "A", "B", "B"]))


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_non_finite_values(bad_value):
    model = make_model()

    X = {
        "numeric": np.array([
            [1.0, 2.0],
            [3.0, bad_value],
        ])
    }

    with pytest.raises(ValueError):
        model.fit(X, np.array(["A", "B"]))


def test_rejects_wrong_feature_dimensions():
    model = make_model()

    X = {"numeric": np.ones(4)}

    with pytest.raises(ValueError):
        model.fit(X, np.array(["A", "A", "B", "B"]))


def test_rejects_wrong_label_shape():
    model = make_model()

    X = {"numeric": np.ones((4, 3))}
    y = np.array([["A"], ["A"], ["B"], ["B"]])

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_rejects_inconsistent_number_of_labels():
    model = make_model()

    X = {"numeric": np.ones((4, 3))}
    y = np.array(["A", "B"])

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_prediction_before_fit_fails():
    model = make_model()

    with pytest.raises(RuntimeError):
        model.predict({"numeric": np.ones((2, 3))})


def test_prediction_requires_same_feature_names():
    model = make_model()

    X = {"numeric": np.ones((4, 3))}
    y = np.array(["A", "A", "B", "B"])

    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict({"different_feature": np.ones((2, 3))})


def test_prediction_requires_compatible_dimensions():
    model = make_model()
    X = {"numeric": np.ones((4, 3))}
    y = np.array(["A", "A", "B", "B"])

    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict({"numeric": np.ones((2, 5))})


def test_empty_estimator_set_is_rejected():
    with pytest.raises(ValueError):
        FrequencyEstimator(estimators=set())