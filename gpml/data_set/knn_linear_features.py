import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class NearestNeighbourLinearFeatures:
    """
    KNN Feature transformer.

    Source:
    https://www.kaggle.com/monigame/bnp-paribas-cardif-claims-management/nearest-neighbour-linear-features/code
    """
    def __init__(self, n_neighbours=1, max_elts=None,
                 verbose=True, random_state=None):
        self.rnd = random_state
        self.n = n_neighbours
        self.max_elts = max_elts
        self.verbose = verbose
        self.neighbours = []
        self.clfs = []

        if random_state is not None:
            random.seed(self.rnd)

    def fit(self, train, y):
        clf = LinearRegression(
            fit_intercept=False, normalize=True,
            copy_X=True, n_jobs=-1
        )

        if self.max_elts is None:
            self.max_elts = len(train.columns)

        list_vars = list(train.columns)
        random.shuffle(list_vars)

        lastscores = np.zeros(self.n) + 1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars = list_vars[self.n:]

        for elt in list_vars:
            indice = 0
            scores = []
            for elt2 in self.neighbours:
                if len(elt2) < self.max_elts:
                    clf.fit(train[elt2 + [elt]], y)
                    scores.append(
                        metrics.log_loss(
                            y,
                            clf.predict(train[elt2 + [elt]])
                        ))
                    indice += 1
                else:
                    scores.append(lastscores[indice])
                    indice += 1
            gains = lastscores - scores
            if gains.max() > 0:
                temp = gains.argmax()
                lastscores[temp] = scores[temp]
                self.neighbours[temp].append(elt)

        indice = 0
        for elt in self.neighbours:
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice += 1

    def transform(self, train):
        indice = 0
        for elt in self.neighbours:
            col_name = '_'.join(pd.Series(elt).sort_values().values)
            train[col_name] = self.clfs[indice].predict(train[elt])
            indice += 1
        return train

    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)
