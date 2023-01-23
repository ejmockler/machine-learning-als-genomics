from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC

from skopt.space import Categorical, Integer, Real

# models keyed by hyperparameter spaces to optimize
stack = {
    AdaBoostClassifier(): [
        Integer(25, 75, name="n_estimators"),
        Real(0, 1, name="learning_rate"),
    ],
    GradientBoostingClassifier(): [
        Real(0, 1, name="learning_rate"),
        Integer(75, 200, name="n_estimators"),
        Integer(0, 10, name="max_depth"),
    ],
    RandomForestClassifier(): [
        Integer(75, 200, name="n_estimators"),
    ],
    LogisticRegression(penalty="elasticnet"): [
        Real(1e-6, 1e-2, name="tol"),
        Real(1e-6, 1, name="C"),
    ],
    MultinomialNB(): [Real(0, 1, name="alpha")],
    LinearSVC(): [
        Real(1e-6, 1e-2, name="tol"),
        Real(1e-6, 1, name="C"),
    ],
    SVC(probability=True, kernel="rbf"): [
        Real(1e-6, 1e-2, name="tol"),
        Real(1e-6, 1, name="C"),
        Categorical(["scale", "auto"], name="gamma"),
    ],
}
