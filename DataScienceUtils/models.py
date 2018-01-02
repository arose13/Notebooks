import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted


def _np_dropna(a):
    """Mimics pandas dropna"""
    return a[~np.isnan(a).any(axis=1)]


# noinspection PyPep8Naming
class PreTrainedVoteEnsemble(ClassifierMixin):
    """
    Binary Ensemble Classifier that only accepts pre-trained models as an input.
    """
    def __init__(self, trained_estimators):
        _ = [check_is_fitted(x) for x in trained_estimators]
        self.estimators = trained_estimators
        self.models_ = [model for _, model in trained_estimators]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)

    def predict_proba(self, X):
        collection = np.asarray([model.predict_proba(X) for model in self.models_])
        return np.average(collection, axis=0)


# noinspection PyPep8Naming
class BayesKDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian Classifier that uses Kernel Density Estimations to generate the joint distribution
    Parameters:
        - bandwidth: float
        - kernel: for scikit learn KernelDensity
    """
    def __init__(self, bandwidth=0.2, kernel='gaussian'):
        self.classes_, self.models_, self.priors_logp_ = [None] * 3
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(x_subset)
                        for x_subset in training_sets]

        self.priors_logp_ = [np.log(x_subset.shape[0] / X.shape[0]) for x_subset in training_sets]
        return self

    def predict_proba(self, X):
        logp = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logp + self.priors_logp_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
