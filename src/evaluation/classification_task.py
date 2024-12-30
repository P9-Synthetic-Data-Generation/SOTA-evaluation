import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    return {"AUROC": auroc, "Precision": precision, "Recall": recall, "F1 Score": f1}


def tstr_multiple_synthetic(
    real_train_path, real_test_path, real_train_labels_path, real_test_labels_path, synthetic_data_paths, classifiers
):
    X_train_real = pd.read_pickle(real_train_path)
    X_test_real = pd.read_pickle(real_test_path)
    y_train_real = pd.read_pickle(real_train_labels_path)
    y_test_real = pd.read_pickle(real_test_labels_path)

    results = {"TRTR": {}, "TSTR": {}}

    for clf_name, clf in classifiers.items():
        trtr_results = evaluate_model(clf, X_train_real, y_train_real, X_test_real, y_test_real)
        results["TRTR"][clf_name] = trtr_results

    for synth_path in synthetic_data_paths:
        synthetic_data = pd.read_csv(synth_path, header=None)
        X_synth = synthetic_data.iloc[:, :-1]
        y_synth = synthetic_data.iloc[:, -1]

        for clf_name, clf in classifiers.items():
            tstr_results = evaluate_model(clf, X_synth, y_synth, X_test_real, y_test_real)
            results["TSTR"][(synth_path, clf_name)] = tstr_results

    return results


if __name__ == "__main__":
    real_train_path = "data/mimic-iii_preprocessed/pickle_data/training_data.pkl"
    real_train_labels_path = "data/mimic-iii_preprocessed/pickle_data/training_labels.pkl"

    real_test_path = "data/mimic-iii_preprocessed/pickle_data/test_data.pkl"
    real_test_labels_path = "data/mimic-iii_preprocessed/pickle_data/test_labels.pkl"

    synthetic_data_paths = [
        "data_synthetic/synthcity_pategan_1eps.csv",
        "data_synthetic/synthcity_pategan_5eps.csv",
        "data_synthetic/synthcity_pategan_10eps.csv",
        "data_synthetic/synthcity_dpgan_1eps.csv",
        "data_synthetic/synthcity_dpgan_5eps.csv",
        "data_synthetic/synthcity_dpgan_10eps.csv",
        "data_synthetic/smartnoise_patectgan_1eps.csv",
        "data_synthetic/smartnoise_patectgan_5eps.csv",
        "data_synthetic/smartnoise_patectgan_10eps.csv",
        "data_synthetic/smartnoise_dpctgan_1eps.csv",
        "data_synthetic/smartnoise_dpctgan_5eps.csv",
        "data_synthetic/smartnoise_dpctgan_10eps.csv",
    ]

    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boost": GradientBoostingClassifier(),
    }

    results = tstr_multiple_synthetic(
        real_train_path,
        real_test_path,
        real_train_labels_path,
        real_test_labels_path,
        synthetic_data_paths,
        classifiers,
    )

    print("\nResults for TRTR:")
    trtr_table = []
    for clf_name, metrics in results["TRTR"].items():
        trtr_table.append(
            [clf_name, "Real Data", metrics["AUROC"], metrics["Precision"], metrics["Recall"], metrics["F1 Score"]]
        )
    trtr_df = pd.DataFrame(trtr_table, columns=["Classifier", "Dataset", "AUROC", "Precision", "Recall", "F1 Score"])
    print(trtr_df)

    print("\nResults for TSTR:")
    tstr_table = []
    for (synth_path, clf_name), metrics in results["TSTR"].items():
        tstr_table.append(
            [clf_name, synth_path, metrics["AUROC"], metrics["Precision"], metrics["Recall"], metrics["F1 Score"]]
        )
    tstr_df = pd.DataFrame(tstr_table, columns=["Classifier", "Dataset", "AUROC", "Precision", "Recall", "F1 Score"])

    avg_tstr = tstr_df.groupby("Dataset")[["AUROC", "Precision", "Recall", "F1 Score"]].mean().reset_index()
    avg_tstr["Classifier"] = "Average"
    tstr_df = pd.concat([tstr_df, avg_tstr], ignore_index=True)

    print(tstr_df)
