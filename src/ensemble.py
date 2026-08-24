"""Ensemble methods for combining model predictions."""

import numpy as np
from sklearn.metrics import mean_squared_error
from config import ENSEMBLE_VAL_RATIO


class WeightedEnsemble:
    """Weighted ensemble using inverse validation error weighting."""

    def __init__(self, models_dict):
        """Initialize with dictionary of fitted models.

        Args:
            models_dict: Dict of {name: model_instance}
        """
        self.models = models_dict
        self.weights = None

    def fit_weights(self, X_val, y_val):
        """Learn weights from validation set using inverse MSE weighting.

        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        errors = []
        for name, model in self.models.items():
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            errors.append(mse)

        inv_errors = 1.0 / np.array(errors)
        self.weights = inv_errors / inv_errors.sum()

        print(f"Ensemble weights: {dict(zip(self.models.keys(), self.weights))}")
        return self

    def predict(self, X):
        """Generate weighted ensemble prediction."""
        predictions = []
        for name, model in self.models.items():
            preds = model.predict(X)
            predictions.append(preds)

        pred_matrix = np.column_stack(predictions)
        return pred_matrix @ self.weights


class EqualEnsemble:
    """Simple equal-weighted ensemble."""

    def __init__(self, models_dict):
        self.models = models_dict

    def predict(self, X):
        predictions = []
        for model in self.models.values():
            preds = model.predict(X)
            predictions.append(preds)

        pred_matrix = np.column_stack(predictions)
        return pred_matrix.mean(axis=1)
