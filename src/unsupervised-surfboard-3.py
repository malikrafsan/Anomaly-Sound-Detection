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

    # drop columns with a lot of NaN
    DROPPED_COLUMNS_NAN = ["apq5Shimmer", "apq11Shimmer", "ppq5Jitter"]
    df_normal_dropped = df_normal_dropped.drop(columns=DROPPED_COLUMNS_NAN)
    df_abnormal_dropped = df_abnormal_dropped.drop(columns=DROPPED_COLUMNS_NAN)

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

    return df_normal_imputed, df_abnormal_imputed

def split_data(
    df_normal_imputed: pd.DataFrame,
    df_abnormal_imputed: pd.DataFrame,
):
    # unsupervised learning
    # split normal dataset into 80% training and 20% testing
    df_normal_train = df_normal_imputed.sample(frac=0.8, random_state=0)    
    df_normal_test = df_normal_imputed.drop(df_normal_train.index)

    # combine normal test dataset and abnormal dataset
    df_test = pd.concat([df_normal_test, df_abnormal_imputed], ignore_index=True)
    df_train = df_normal_train

    return df_train, df_test

def split_data_kfold(
    df_normal_imputed_train: pd.DataFrame,
    df_normal_imputed_test: pd.DataFrame,
    df_abnormal_imputed: pd.DataFrame,
):
    df_test = pd.concat([df_normal_imputed_test, df_abnormal_imputed], ignore_index=True)
    df_train = df_normal_imputed_train

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

def visualize(
    machine_type: str,
    machine_id: str,
    imgdirpath: str,
    model_name: str,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    recall: np.ndarray,
    precision: np.ndarray,
    pr_auc: float,
):
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


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_samples: np.ndarray,
    machine_type: str,
    machine_id: str,
    imgdirpath: str,
    model_name: str,
    visualize_flag: bool = True,
):
    print(f"Calculating metrics for {machine_type} {machine_id} {model_name}")
    print(f"y_true: {len(y_true)}")
    print(f"score_samples: {len(score_samples)}")
    # calculate roc auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, score_samples)
    roc_auc = metrics.auc(fpr, tpr)

    best_threshold = None
    best_f1_score = None
    for threshold in thresholds:
        cur_y_pred = [1 if x >= threshold else -1 for x in score_samples]
        cur_f1_score = metrics.f1_score(y_true, cur_y_pred)
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

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, score_samples)
    pr_auc = metrics.auc(recall, precision)

    if visualize_flag:
        visualize(
            machine_type=machine_type,
            machine_id=machine_id,
            imgdirpath=imgdirpath,
            model_name=model_name,
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            recall=recall,
            precision=precision,
            pr_auc=pr_auc,
        )

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
    dirpath: str,
    outdir: str,
):
    models = {
        "isolation_forest": isolation_forest,
        "one_class_svm": one_class_svm,
        "local_outlier_factor": local_outlier_factor,
        "gaussian_mixture": gaussian_mixture,
    }    

    # dirpath = f"../out/raw/praat"
    normal_filename = f"w-{machine_type}-{machine_id}-normal-(-1).csv"
    abnormal_filename = f"w-{machine_type}-{machine_id}-abnormal-(-1).csv"
    
    # outdir = f"../out/results/praat"
    imgdirpath = f"{outdir}/images-3"
    resultdirpath = f"{outdir}/results-3"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(imgdirpath, exist_ok=True)
    os.makedirs(resultdirpath, exist_ok=True)

    normal_filepath = f"{dirpath}/{normal_filename}"
    abnormal_filepath = f"{dirpath}/{abnormal_filename}"

    df_normal_imputed, df_abnormal_imputed = prepare_data(
        normal_filepath=normal_filepath,
        abnormal_filepath=abnormal_filepath,
    )

    df_train, df_test = split_data(
        df_normal_imputed=df_normal_imputed,
        df_abnormal_imputed=df_abnormal_imputed,
    )

    y_test = df_test["label"]
    y_true =[1 if x == False else -1 for x in y_test.to_numpy()]

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    results = []
    for model_name, model in models.items():
        print(f"Processing {model_name}")
        kfold_results = []
        for train_index, test_index in kf.split(df_normal_imputed):
            df_train_normal_kfold = df_normal_imputed.iloc[train_index]
            df_test_normal_kfold = df_normal_imputed.iloc[test_index]

            df_test_kfold = pd.concat([df_test_normal_kfold, df_abnormal_imputed], ignore_index=True)
            df_train_kfold = df_train_normal_kfold

            y_test_kfold = df_test_kfold["label"]
            y_true_kfold =[1 if x == False else -1 for x in y_test_kfold.to_numpy()]

            y_pred_kfold, score_samples_kfold = model(
                df_train=df_train_kfold,
                df_test=df_test_kfold,
            )

            metricsres_kfold = calculate_metrics(
                y_true=y_true_kfold,
                y_pred=y_pred_kfold,
                score_samples=score_samples_kfold,
                machine_type=machine_type,
                machine_id=machine_id,
                imgdirpath=imgdirpath,
                model_name=model_name,
                visualize_flag=False,
            )
            kfold_results.append(metricsres_kfold)
        
        roc_aucs = [x["roc_auc"] for x in kfold_results]
        pr_aucs = [x["pr_auc"] for x in kfold_results]

        mean_roc_auc = np.mean(roc_aucs)
        mean_pr_auc = np.mean(pr_aucs)
        
        std_roc_auc = np.std(roc_aucs)
        std_pr_auc = np.std(pr_aucs)

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
            visualize_flag=True,
        )
        # print(metricsres)
        # results.append(metricsres)

        res = {
            "model_name": model_name,
            "mean_roc_auc": mean_roc_auc,
            "mean_pr_auc": mean_pr_auc,
            "std_roc_auc": std_roc_auc,
            "std_pr_auc": std_pr_auc,
            **metricsres,
        }
        results.append(res)

    with open(f"{resultdirpath}/{machine_type}-{machine_id}-metrics.json", "w") as outfile:
        json.dump(results, outfile, cls=NpEncoder, indent=4)

    return results

def main():
    dirpath = f"../out/raw/surfboard"
    outdir = f"../out/results/surfboard"

    all_results = []
    for machine_type in MACHINE_TYPES:
        for machine_id in MACHINE_IDS:
            print(f"Processing {machine_type} {machine_id}")

            res = process(machine_type, machine_id, dirpath, outdir)
            append_res = {
                "machine_type": machine_type,
                "machine_id": machine_id,
                "metrics": res,
            }

            all_results.append(append_res)

    with open(f"{outdir}/all-metrics-3.json", "w") as outfile:
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
            roc_aucs[model_name+"_mean"] = metric["mean_roc_auc"]
            roc_aucs[model_name+"_std"] = metric["std_roc_auc"]

        roc_aucs["machine_type"] = machine_type
        roc_aucs["machine_id"] = machine_id

        brief_results.append(roc_aucs)
    
    with open(f"{outdir}/brief-metrics-3.json", "w") as outfile:
        json.dump(brief_results, outfile, cls=NpEncoder, indent=4)


if __name__ == "__main__":
    main()