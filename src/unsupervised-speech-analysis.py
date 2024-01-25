MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
LABELS = ["normal", "abnormal"]
DROPPED_COLUMNS = [
    "filename","id_machine","machine_type","label"
]

import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import json
import os
from sklearn.svm import OneClassSVM
import sklearn.metrics as metrics

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def prepare_data(
    normal_filepath: str,
    abnormal_filepath: str,
):
    df_normal = pd.read_csv(normal_filepath)
    df_abnormal = pd.read_csv(abnormal_filepath)

    df_normal_dropped = df_normal.drop(columns=DROPPED_COLUMNS)
    df_abnormal_dropped = df_abnormal.drop(columns=DROPPED_COLUMNS)

    df_normal_dropped["label"] = False
    df_abnormal_dropped["label"] = True

    # get columns with all NaN
    cols_nan_normal = df_normal_dropped.columns[df_normal_dropped.isnull().all()].tolist()
    cols_nan_abnormal = df_abnormal_dropped.columns[df_abnormal_dropped.isnull().all()].tolist()

    # concat
    cols_nan = list(set(cols_nan_normal + cols_nan_abnormal))

    # drop columns with all NaN
    df_normal_dropped = df_normal_dropped.drop(columns=cols_nan)
    df_abnormal_dropped = df_abnormal_dropped.drop(columns=cols_nan)


    imputer_normal = SimpleImputer(strategy="mean")
    ndarray_normal_imputed = imputer_normal.fit_transform(df_normal_dropped)
    df_normal_imputed = pd.DataFrame(ndarray_normal_imputed, columns=df_normal_dropped.columns)

    imputer_abnormal = SimpleImputer(strategy="mean")
    ndarray_abnormal_imputed = imputer_abnormal.fit_transform(df_abnormal_dropped)
    df_abnormal_imputed = pd.DataFrame(ndarray_abnormal_imputed, columns=df_abnormal_dropped.columns)

    # unsupervised learning
    # split normal dataset into 80% training and 20% testing
    df_normal_train = df_normal_imputed.sample(frac=0.8, random_state=0)    
    df_normal_test = df_normal_imputed.drop(df_normal_train.index)

    # combine normal test dataset and abnormal dataset
    df_test = pd.concat([df_normal_test, df_abnormal_imputed], ignore_index=True)
    df_train = df_normal_train

    return df_train, df_test

def isolation_forest(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
):
    x_train = df_train.drop(columns=["label"])
    x_test = df_test.drop(columns=["label"])

    clf = IsolationForest(random_state=0)
    clf.fit(x_train)

    y_pred = clf.predict(x_test)
    score_samples = clf.score_samples(x_test)

    return y_pred, score_samples

def one_class_svm(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
):
    x_train = df_train.drop(columns=["label"])
    x_test = df_test.drop(columns=["label"])

    clf = OneClassSVM(gamma='auto', nu=0.1)
    clf.fit(x_train)

    y_pred = clf.predict(x_test)
    score_samples = clf.score_samples(x_test)

    return y_pred, score_samples

def local_outlier_factor(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
):
    from sklearn.neighbors import LocalOutlierFactor

    x_train = df_train.drop(columns=["label"])
    x_test = df_test.drop(columns=["label"])

    clf = LocalOutlierFactor(n_neighbors=20, novelty=True)
    clf.fit(x_train.values)

    y_pred = clf.predict(x_test)
    score_samples = clf.score_samples(x_test)

    return y_pred, score_samples

def find_outlier_bound(
    data: np.ndarray
):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound

def gaussian_mixture(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
):
    from sklearn.mixture import GaussianMixture

    x_train = df_train.drop(columns=["label"])
    x_test = df_test.drop(columns=["label"])

    clf = GaussianMixture(n_components=1, random_state=0)
    clf.fit(x_train)

    score_samples = clf.score_samples(x_test)

    # lower_bound, upper_bound = find_outlier_bound(score_samples)

    # y_pred = []
    # for score in score_samples:
    #     if score < lower_bound or score > upper_bound:
    #         y_pred.append(-1)
    #     else:
    #         y_pred.append(1)

    return None, score_samples

def calculate_derived_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    f1_score = metrics.f1_score(y_true, y_pred)
    precision_score = metrics.precision_score(y_true, y_pred)
    recall_score = metrics.recall_score(y_true, y_pred)
    accuracy_score = metrics.accuracy_score(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    classification_report = metrics.classification_report(y_true, y_pred)

    return {
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "classification_report": classification_report,
    }

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_samples: np.ndarray,
    machine_type: str,
    machine_id: str,
    imgdirpath: str,
    model_name: str,
):
    # calculate roc auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, score_samples)
    roc_auc = metrics.auc(fpr, tpr)

    best_threshold = None
    best_f1_score = None
    for threshold in thresholds:
        y_pred = [1 if x >= threshold else -1 for x in score_samples]
        cur_f1_score = metrics.f1_score(y_true, y_pred)
        if best_f1_score is None or cur_f1_score > best_f1_score:
            best_f1_score = cur_f1_score
            best_threshold = threshold

    y_pred_iterative = [1 if x >= best_threshold else -1 for x in score_samples]

    iterative_metrics = calculate_derived_metrics(
        y_true=y_true,
        y_pred=y_pred_iterative,
    )
    iterative_metrics["best_threshold"] = best_threshold
    
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]
    y_pred_index = [1 if x >= best_threshold else -1 for x in score_samples]
    index_metrics = calculate_derived_metrics(
        y_true=y_true,
        y_pred=y_pred_index,
    )
    index_metrics["best_threshold"] = best_threshold


    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = f'AUC = {roc_auc:.2f}')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f"{imgdirpath}/{machine_type}-{machine_id}-{model_name}-roc.png")
    plt.clf()

    # calculate precision-recall curve
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, score_samples)
    pr_auc = metrics.auc(recall, precision)

    plt.title('Precision-Recall Curve')
    plt.plot(recall, precision, 'b', label = f'PR AUC = {pr_auc:.2f}')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0.5, 0.5],'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(f"{imgdirpath}/{machine_type}-{machine_id}-{model_name}-pr.png")
    plt.clf()

    # save metrics
    metricsres = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "model_name": model_name,
        "iterative_threshold_method": iterative_metrics,
        "derive_threshold_method": index_metrics,
        # "pred_method": pred_metrics,
    }

    return metricsres


def process(
    machine_type: str,
    machine_id: str,
):
    dirpath = f"../out/raw/speech-analysis"
    normal_filename = f"w-{machine_type}-{machine_id}-normal-(-1).csv"
    abnormal_filename = f"w-{machine_type}-{machine_id}-abnormal-(-1).csv"
    
    outdir = f"../out/results/speech-analysis"
    imgdirpath = f"{outdir}/images"
    resultdirpath = f"{outdir}/results"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(imgdirpath, exist_ok=True)
    os.makedirs(resultdirpath, exist_ok=True)

    normal_filepath = f"{dirpath}/{normal_filename}"
    abnormal_filepath = f"{dirpath}/{abnormal_filename}"

    df_train, df_test = prepare_data(
        normal_filepath=normal_filepath,
        abnormal_filepath=abnormal_filepath,
    )

    y_test = df_test["label"]
    y_true =[1 if x == False else -1 for x in y_test.to_numpy()]

    models = {
        "isolation_forest": isolation_forest,
        "one_class_svm": one_class_svm,
        "local_outlier_factor": local_outlier_factor,
        "gaussian_mixture": gaussian_mixture,
    }

    results = []
    for model_name, model in models.items():
        print(f"Processing {model_name}")
        y_pred, score_samples = model(
            df_train=df_train,
            df_test=df_test,
        )

        metricsres = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            score_samples=score_samples,
            machine_type=machine_type,
            machine_id=machine_id,
            imgdirpath=imgdirpath,
            model_name=model_name,
        )
        print(metricsres)
        results.append(metricsres)
    
    with open(f"{resultdirpath}/{machine_type}-{machine_id}-metrics.json", "w") as outfile:
        json.dump(results, outfile, cls=NpEncoder, indent=4)

    return results


def main():
    all_results = []
    for machine_type in MACHINE_TYPES:
        for machine_id in MACHINE_IDS:
            print(f"Processing {machine_type} {machine_id}")

            res = process(machine_type, machine_id)
            append_res = {
                "machine_type": machine_type,
                "machine_id": machine_id,
                "metrics": res,
            }

            all_results.append(append_res)

    with open(f"../out/results/speech-analysis/all-metrics.json", "w") as outfile:
        json.dump(all_results, outfile, cls=NpEncoder, indent=4)

    brief_results = []
    for result in all_results:
        # get roc auc per model
        machine_type = result["machine_type"]
        machine_id = result["machine_id"]
        roc_aucs = {}
        for metric in result["metrics"]:
            model_name = metric["model_name"]
            roc_auc = metric["roc_auc"]
            roc_aucs[model_name] = roc_auc

        roc_aucs["machine_type"] = machine_type
        roc_aucs["machine_id"] = machine_id

        brief_results.append(roc_aucs)
    
    with open(f"../out/results/speech-analysis/brief-metrics.json", "w") as outfile:
        json.dump(brief_results, outfile, cls=NpEncoder, indent=4)


if __name__ == "__main__":
    main()
