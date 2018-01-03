# Collection of Data Science Utilities. If it gets out of hand break them off into their on files
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GetSubsetTransform(BaseEstimator, TransformerMixin):
    """
    Selects the columns of a numpy array.
    This allows subsets of X in a pipeline
    """
    def __init__(self, indices):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(X)[:, self.indices]
