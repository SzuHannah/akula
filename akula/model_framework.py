import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

try:
    from gplearn.genetic import SymbolicRegressor
    HAS_GPLEARN = True
except Exception:  # pragma: no cover - optional dependency
    HAS_GPLEARN = False


@dataclass
class BaseModel:
    """Base class for high dimensional models."""

    data: np.ndarray
    model_type: str = "custom"
    sampling_function: Optional[Callable[[int], np.ndarray]] = None
    contributions: Optional[np.ndarray] = None
    params: Dict[str, np.ndarray] = field(default_factory=dict)

    def sample(self, n: int) -> np.ndarray:
        """Sample input data using the provided sampling_function or default."""
        if self.sampling_function:
            return self.sampling_function(n)
        # Default: simple normal sampling
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0) + 1e-9
        return np.random.normal(mean, std, size=(n, self.data.shape[1]))

    def screen_contributions(self, y: np.ndarray, threshold: float = 0.01) -> List[int]:
        """Return indices of exchanges contributing more than threshold."""
        totals = np.abs(y)
        mask = totals / np.max(totals) > threshold
        self.contributions = mask
        return np.where(mask)[0].tolist()

    def parameterize_exchanges(self, indices: List[int]):
        """Keep only selected indices for further analysis."""
        self.params = {"indices": np.array(indices)}
        self.data = self.data[:, indices]

    def global_sensitivity(self, y: np.ndarray) -> np.ndarray:
        """Compute simple variance-based sensitivity indices."""
        X = self.data
        var_total = np.var(y)
        s = []
        for i in range(X.shape[1]):
            s.append(np.var(X[:, i] * y) / (var_total + 1e-9))
        return np.array(s)

    def simplify_with_lasso(self, y: np.ndarray, alpha: float = 0.01) -> Lasso:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(self.data, y)
        return model

    def simplify_with_symbolic(self, y: np.ndarray):
        if not HAS_GPLEARN:
            raise ImportError("gplearn is not installed")
        model = SymbolicRegressor()
        model.fit(self.data, y)
        return model

    def simplify_with_neural_network(self, y: np.ndarray) -> MLPRegressor:
        model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=2000)
        model.fit(self.data, y)
        return model

    def expression_from_model(self, model) -> str:
        """Return a mathematical expression if available."""
        if hasattr(model, "_program"):
            return str(model._program)
        coef = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", 0.0)
        if coef is None:
            return ""
        terms = [f"{coef[i]:+.3f}*x{i}" for i in range(len(coef)) if abs(coef[i]) > 1e-9]
        expr = " ".join(terms)
        return f"{expr} {intercept:+.3f}"

    def discrepancy(self, model, y: np.ndarray) -> float:
        y_pred = model.predict(self.data)
        return float(np.sqrt(np.mean((y - y_pred) ** 2)))


class LCAModel(BaseModel):
    allocation: str = "cutoff"

    def __init__(self, data: np.ndarray, allocation: str = "cutoff", **kwargs):
        super().__init__(data, model_type="lca", **kwargs)
        self.allocation = allocation


class TEAModel(BaseModel):
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, model_type="tea", **kwargs)

