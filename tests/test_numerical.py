import numpy as np, pytest
from scipy.fftpack import dct

from FreqEst import FrequencyEstimator

@pytest.fixture
def dataset():
    rng = np.random.default_rng(42)

    y = np.array(["TRUE"] * 6 + ["FALSE"] * 6)
    locations = np.where(y == "TRUE", -1.0, 1.0)

    X = {
        "numeric": rng.normal(
            loc=locations[:, None],
            scale=0.5,
            size=(12, 6),
        ),

        # Grayscale image representations
        "image": rng.normal(
            loc=locations[:, None, None],
            scale=0.5,
            size=(12, 4, 4),
        ),

        # Dense numerical representations of text
        "text_vector": rng.normal(
            loc=locations[:, None],
            scale=0.5,
            size=(12, 8),
        ),

        # Multichannel time series
        "time_series": rng.normal(
            loc=locations[:, None, None],
            scale=0.5,
            size=(12, 10, 3),
        ),
    }

    return X, y


def make_model():
    return FrequencyEstimator(
        normalization=True,
        n_components=4,
        estimators={"gaussian", "naive_bayes"},
        reg=0.1,
    )


@pytest.mark.parametrize(
    "feature",
    ["numeric", "image", "text_vector", "time_series"],
)
def test_individual_feature_types(feature, dataset):
    X, y = dataset
    model = make_model()

    model.fit({feature: X[feature]}, y)

    predictions = model.predict({
        feature: X[feature][:3]
    })

    assert predictions.shape == (3,)
    assert np.isin(predictions, model._classes).all()


def test_multiple_features_with_different_dimensions(dataset):
    X, y = dataset
    model = make_model()

    model.fit(X, y)

    X_test = {
        name: values[:3]
        for name, values in X.items()
    }

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    assert probabilities.shape == (3, len(model._classes))
    assert predictions.shape == (3,)
    assert np.isfinite(probabilities).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(3),
        atol=1e-7,
    )


def test_dct_is_applied_sequentially():
    model = FrequencyEstimator(normalization=True)

    X = np.arange(24, dtype=float).reshape(2, 3, 4)

    expected = dct(X, axis=1, type=2, norm="ortho")
    expected = dct(expected, axis=2, type=2, norm="ortho")

    actual = model.transform_to_dct(X)

    np.testing.assert_allclose(actual, expected)


def test_legacy_three_dimensional_array(dataset):
    X, y = dataset
    rng = np.random.default_rng(10)

    array_X = rng.normal(size=(12, 2, 6))

    model = make_model()
    model.fit(array_X, y)

    predictions = model.predict(array_X[:2])

    assert predictions.shape == (2,)


def test_raw_text_is_rejected():
    model = make_model()

    X = {
        "text": np.array([
            ["red cookie"],
            ["blue cookie"],
        ])
    }

    with pytest.raises((TypeError, ValueError)):
        model._prepare_features(X)