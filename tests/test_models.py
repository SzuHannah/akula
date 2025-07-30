import numpy as np
from akula.model_framework import LCAModel, TEAModel


def test_screen_and_simplify_lasso():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    coefs = np.array([1.0, 0.5, 0, 0, 0])
    y = X @ coefs + rng.normal(scale=0.01, size=100)

    model = TEAModel(X)
    idx = model.screen_contributions(np.abs(coefs), threshold=0.1)
    model.parameterize_exchanges(idx)
    lasso_model = model.simplify_with_lasso(y)
    expr = model.expression_from_model(lasso_model)

    assert "x0" in expr
    assert model.discrepancy(lasso_model, y) < 0.1

