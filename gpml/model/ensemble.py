import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from gpml.xgboost_mod.improved_early_stopping import XGBCEarlyStoppingCV


class Ensemble(object):
    """All together now."""

    def __init__(self, name, models, fitting_parameters):
        self.name = name
        self.models = models
        self.fitting_parameters = fitting_parameters

    def fit(self, X, y):
        self.not_implemented_error()

    def predict_proba(self, X):
        self.not_implemented_error()

    def predict(self, X):
        self.not_implemented_error()

    def fit_each_model(self, X, y):
        print('Training individual models:')
        for name, model in self.models.items():
            print(' - training %s' % name)
            model.fit(X, y, **self.fitting_parameters[name])
        return self

    def predict_proba_by_model(self, X):
        ps = []
        for name, model in self.models.items():
            ps.append(model.predict_proba(X))
        ps = np.array(ps)
        # ps = ps.T
        return ps

    @staticmethod
    def not_implemented_error():
        raise NotImplementedError(
            'Model specific method, inheritors must overload this'
        )


class AveragingEnsemble(Ensemble):
    """Average those models."""

    def __init__(self, name, models, fitting_parameters, weights):
        self.weights = weights
        super().__init__(name, models, fitting_parameters)

    def fit(self, X, y):
        self.fit_each_model(X, y)
        return self

    def predict_proba(self, X):
        ps = []
        weights_array = []
        for name, model in self.models.items():
            ps.append(model.predict_proba(X))
            weights_array.append(self.weights[name])
        ps = np.array(ps)
        weights_array = np.array(weights_array)
        p = (ps.T.dot(weights_array) / weights_array.sum()).T
        return p

    def predict(self, X):
        p = np.zeros((X.shape[0], 2))  # Assume binary.
        for name, model in self.models.items():
            p += (model.predict_proba(X) * self.weights[name])
        k = p.argmax(axis=1)
        return k


class SKLearnBasedEnsemble(Ensemble):
    """Use an sklearn model as an ensembler."""

    def __init__(self, ensemble, name, models, fitting_parameters):
        self.ensemble = ensemble
        super().__init__(name, models, fitting_parameters)

    def fit(self, X, y):
        self.fit_each_model(X, y)
        ps = self.predict_proba_by_model(X)
        ps = self.binarise_probabilities(ps)
        self.ensemble.fit(ps, y, **self.fitting_parameters[self.name])
        return self

    def predict_proba(self, X):
        ps = self.predict_proba_by_model(X)
        ps = self.binarise_probabilities(ps)
        p = self.ensemble.predict_proba(ps)
        return p

    def predict(self, X):
        ps = self.predict_proba_by_model(X)
        ps = self.binarise_probabilities(ps)
        k = self.ensemble.predict(ps)
        return k

    def binarise_probabilities(self, ps):
        ps = ps[:, :, 1]
        # if len(self.models) == 1:
        #     ps = ps.T
        return ps.T


class LREnsemble(SKLearnBasedEnsemble):
    """Weight models using Logistic Regression."""

    def __init__(
        self, name, models, fitting_parameters,
        penalty='l1',
        C=0.1,
        random_state=None,
        n_jobs=-1
    ):
        lr = LogisticRegression(
            penalty=penalty,
            C=C,
            # class_weight='balanced',
            # max_iter=100,
            random_state=random_state,
            # solver='lbfgs',
            # tol=0.000001,
            n_jobs=n_jobs
        )
        super().__init__(lr, name, models, fitting_parameters)


class ETCEnsemble(SKLearnBasedEnsemble):
    def __init__(
        self, name, models, fitting_parameters,
        n_estimators=200,
        max_features=5,
        criterion='entropy',
        min_samples_split=4,
        max_depth=35,
        min_samples_leaf=2,

        n_jobs=-1,
        verbose=0,
        random_state=2
    ):
        ensemble = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion=criterion,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,

            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
        )
        super().__init__(ensemble, name, models, fitting_parameters)


class XGBCEnsemble(SKLearnBasedEnsemble):
    def __init__(
        self, name, models, fitting_parameters,

        max_depth=9,
        learning_rate=0.01,
        n_estimators=10000,
        # silent=False,
        objective='binary:logistic',
        nthread=-1,
        gamma=2,
        min_child_weight=10,
        max_delta_step=1,
        subsample=0.96,
        colsample_bytree=0.45,
        colsample_bylevel=1,
        reg_alpha=1,
        reg_lambda=1,
        seed=1,
    ):
        ensemble = XGBCEarlyStoppingCV(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            # silent=silent,
            objective=objective,
            nthread=nthread,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            seed=seed,
        )
        super().__init__(ensemble, name, models, fitting_parameters)

    def fit(self, X, y, eval_metric, early_stopping_rounds, nfold, verbose):
        return super().fit(X, y)
