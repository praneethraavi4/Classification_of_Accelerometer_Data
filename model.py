from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
from sklearn.pipeline import Pipeline


class Model:
    def __init__(self):
        self.cv = StratifiedKFold(n_splits=5)
        self.xgb = XGBClassifier()
        self.rfe_cv = RFECV(estimator=self.xgb, step=1, cv=self.cv, scoring="accuracy")
        self.pipeline = Pipeline(steps=[("s", self.rfe_cv), ("m", self.xgb)])

    def fit_model(self, x_train, y_train):
        self.pipeline.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):
        predictions = self.pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        y_probs = self.pipeline.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_probs)
        return accuracy, auc

    def save_model(self, filename):
        joblib.dump(self.pipeline, filename)
        print(f"Model saved to {filename}")

    def get_selected_features(self):
        return self.rfe_cv.support_
