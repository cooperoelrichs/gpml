from xgboost.sklearn import XGBClassifier
from xgboost.core import DMatrix
from xgboost.training import train, cv
from xgboost.compat import LabelEncoder


class XGBCEarlyStoppingCV(XGBClassifier):
    def fit(self, X, y,
            # sample_weight=None,
            eval_metric=None,
            early_stopping_rounds=None,
            verbose=True,
            nfold=3,
            seed=1):
        xgb_options = self.get_xgb_params()
        self._le = LabelEncoder().fit(y)
        training_labels = self._le.transform(y)
        train_dmatrix = DMatrix(X, label=training_labels, missing=self.missing)

        evaluation_history = cv(
            xgb_options,
            train_dmatrix,
            num_boost_round=self.n_estimators,
            nfold=nfold,
            stratified=True,
            # folds=None,
            metrics=(eval_metric),
            # obj=None,
            # feval=None,
            # maximize=False,
            early_stopping_rounds=early_stopping_rounds,
            # fpreproc=None,
            # as_pandas=True,
            verbose_eval=verbose,
            show_stdv=False,
            seed=seed
        )

        best_iteration = evaluation_history.index[-1]

        self._Booster = train(
            xgb_options,
            train_dmatrix,
            num_boost_round=best_iteration,
            verbose_eval=verbose,
        )

        return self
