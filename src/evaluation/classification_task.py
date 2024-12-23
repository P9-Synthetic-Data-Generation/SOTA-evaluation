import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def evaluate_model(clf, X_train, y_train, X_test, y_test):
    """
    Train a classifier and evaluate it using AUROC, precision, recall, and F1 score.
    """
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    auroc = roc_auc_score(y_test, y_pred_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    return {"AUROC": auroc, "Precision": precision, "Recall": recall, "F1 Score": f1}


def tstr_and_trtr_experiment(
    real_train_path, real_test_path, real_train_labels_path, real_test_labels_path, synthetic_data_path, classifiers
):
    X_train_real = pd.read_pickle(real_train_path)
    X_test_real = pd.read_pickle(real_test_path)
    y_train_real = pd.read_pickle(real_train_labels_path)
    y_test_real = pd.read_pickle(real_test_labels_path)

    synthetic_data = pd.read_csv(synthetic_data_path, header=None)

    X_synth = synthetic_data.iloc[:, :-1]
    y_synth = synthetic_data.iloc[:, -1]

    results = {"TSTR": {}, "TRTR": {}}

    for clf_name, clf in classifiers.items():
        # TSTR
        tstr_results = evaluate_model(clf, X_synth, y_synth, X_test_real, y_test_real)
        results["TSTR"][clf_name] = tstr_results

        # TRTR
        trtr_results = evaluate_model(clf, X_train_real, y_train_real, X_test_real, y_test_real)
        results["TRTR"][clf_name] = trtr_results

    return results


if __name__ == "__main__":
    real_train_path = "data/mimic-iii_preprocessed/pickle_data/train_data.pkl"
    real_test_path = "data/mimic-iii_preprocessed/pickle_data/test_data.pkl"
    real_train_labels_path = "data/mimic-iii_preprocessed/pickle_data/train_labels.pkl"
    real_test_labels_path = "data/mimic-iii_preprocessed/pickle_data/test_labels.pkl"
    synthetic_data_path = "data/synthetic_data/synthcity_pategan.csv"

    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    results = tstr_and_trtr_experiment(
        real_train_path, real_test_path, real_train_labels_path, real_test_labels_path, synthetic_data_path, classifiers
    )

    for method, method_results in results.items():
        print(f"\nResults for {method}:")
        for clf_name, metrics in method_results.items():
            print(f"\nClassifier: {clf_name}")
            print(f"AUROC: {metrics['AUROC']:.2f}")
            print(f"Precision: {metrics['Precision']:.2f}")
            print(f"Recall: {metrics['Recall']:.2f}")
            print(f"F1 Score: {metrics['F1 Score']:.2f}")
