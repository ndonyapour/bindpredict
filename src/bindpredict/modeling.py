from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
import pickle as pkl
from sklearn.metrics import fbeta_score, make_scorer
from imblearn.combine import SMOTEENN
from sklearn import metrics
import pandas as pd


class TrainBalancedRF(object):

    def __init__(self, num_folds=3, n_jobs=-1):
        self.cv = StratifiedKFold(num_folds)
        self.n_jobs = n_jobs
        self.fbeta_scorer = make_scorer(fbeta_score, beta=1.5)

    def train(self, X_train, y_train, model_save_path):

        param_grid = {'base_estimator__n_estimators': range(5, 101, 5),
                      "base_estimator__max_depth": [1, 3, 5, None],
                      "base_estimator__max_features": ["auto", "log2",
                                                       None, 0.5, 0.2],
                      "base_estimator__min_samples_leaf": [0.5, 0.3, 0.1],
                      "base_estimator__criterion": ["gini", "entropy"]}

        estimator = RandomForestClassifier()
        bbc = BalancedBaggingClassifier(base_estimator=estimator(),
                                        sampling_strategy=SMOTEENN())

        bbc_gscv = GridSearchCV(bbc, param_grid,
                                scoring=self.fbeta_scorer,
                                n_jobs=self.n_jobs,
                                cv=self.cv)

        bbc_gscv.fit(X_train, y_train)

        pkl.dump(bbc_gscv.best_estimator_,
                 open(model_save_path, 'wb'))

        return (bbc_gscv.best_estimator_,
                bbc_gscv.best_params_,
                bbc_gscv.best_score_)


class TestModel(object):

    def __init__(self, trained_model_path):
        self.model = pkl.load(open(trained_model_path, 'rb'))

    def test(self,  X_test, y_test):

        y_pred_prob = self.model.predict_proba(X_test)
        y_pred_ = self.model.predict(X_test)
        results = []

        cm = metrics.confusion_matrix(y_test, y_pred_)
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]
        results.extend([TP, FN, TN, FP])

        prec = metrics.precision_score(y_test, y_pred_)
        results.append(prec)
        recall = metrics.recall_score(y_test, y_pred_)
        results.append(recall)
        F1 = metrics.f1_score(y_test, y_pred_)
        results.append(F1)
        MCC = metrics.matthews_corrcoef(y_test, y_pred_)
        results.append(MCC)

        auroc = metrics.roc_auc_score(y_test, y_pred_prob[:, 1])
        results.append(auroc)
        auprc = metrics.average_precision_score(y_test, y_pred_prob[:, 1])
        results.append(auprc)
        brier = metrics.brier_score_loss(y_test, y_pred_prob[:, 1])
        results.append(brier)

        resultsDF = pd.DataFrame(results,
                                 columns=['Target', 'TP', 'FN',
                                          'TN', 'FP', 'Precision',
                                          'Recall', 'F1-Score', 'MCC',
                                          'AUROC', 'AUPRC', 'brier'])

        return resultsDF
