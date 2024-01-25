from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT_CSV_PATH = "../out/processed/final-timbral"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
MAPPING_FEATURES = {
    "fan": [
        "boominess","brightness","depth","roughness","sharpness",
    ],
    "slider": [
        "brightness","depth","roughness","sharpness",
    ],
    "valve": [
        "boominess","brightness","depth","roughness","sharpness",
    ],
    "pump": [
        "boominess","brightness","roughness","sharpness",
    ]
}

import sys
import logging
import os

def setup_logger(
    filename: str,
    loggername: str = "my_logger",
):
    if os.path.exists(filename):
        os.remove(filename)

    class LogStream(object):
        def __init__(self, logger: logging.Logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level

        def write(self, msg):
            if msg.rstrip() != "":
                self.logger.log(self.log_level, msg.rstrip())

        def flush(self):
            for handler in self.logger.handlers:
                handler.flush()

    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    sys.stdout = LogStream(logger, logging.INFO)
    sys.stderr = LogStream(logger, logging.ERROR)


def scoring(y_true, y_pred, name):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"accuracy: {accuracy:.2f}")

    ps = precision_score(y_true, y_pred)
    print(f"precision_score: {ps:.2f}")

    rs = recall_score(y_true, y_pred)
    print(f"recall_score: {rs:.2f}")

    f1s = f1_score(y_true, y_pred)
    print(f"f1_score: {f1s:.2f}")

    return {
        "name": name,
        "accuracy": accuracy,
        "precision_score": ps,
        "recall_score": rs,
        "f1_score": f1s,
    }

def process(pathnormal:str, pathabnormal: str, machine_type: str):
    df_normal = pd.read_csv(pathnormal)
    df_abnormal = pd.read_csv(pathabnormal)

    df_normal['label'] = False
    df_abnormal['label'] = True

    df = pd.concat([df_normal, df_abnormal], ignore_index=True)

    columns = MAPPING_FEATURES[machine_type]

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(df[columns], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    # perform cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lst_result = []
    idx = 0

    X = df[columns]
    y = df['label']
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        clf = svm.SVC(kernel='linear')  # Linear Kernel
        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_test_fold)

        print(f"cross_validate-{idx}")
        result = scoring(y_test_fold, y_pred, f"cross_validate-{idx}")
        lst_result.append(result)

        idx += 1

    # get mean and std of cross validation
    accuracies = [x['accuracy'] for x in lst_result]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    precisions = [x['precision_score'] for x in lst_result]
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)

    recalls = [x['recall_score'] for x in lst_result]
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)

    f1s = [x['f1_score'] for x in lst_result]
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    # use grid search to find best hyperparameter
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.1, 0.01, 0.001, 'scale', 'auto'],
        'kernel': ['linear', 'rbf', 'sigmoid'],
    }

    svm_classifier = svm.SVC()
    
    grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1')

    grid_search.fit(X_train, y_train)

    print("grid_search.best_params_",grid_search.best_params_)
    print("grid_search.best_score_",grid_search.best_score_)

    # use SVM classifier to predict test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # scoring
    result = scoring(y_test, y_pred, "SVM-no-cross-validate")

    return {
        "cross_validate": lst_result,
        "no_cross_validate": result,
        "mean" : {
            "accuracy": mean_accuracy,
            "precision_score": mean_precision,
            "recall_score": mean_recall,
            "f1_score": mean_f1,
        },
        "std" : {
            "accuracy": std_accuracy,
            "precision_score": std_precision,
            "recall_score": std_recall,
            "f1_score": std_f1,
        },
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
    }


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def main():
    # all_result = []

    # for machine_type in MACHINE_TYPES:
    #     for machine_id in MACHINE_IDS:
    #         print(f"machine_type: {machine_type}, machine_id: {machine_id}")
    #         filepath_normal = f"{ROOT_CSV_PATH}/{machine_type}-{machine_id}-normal-timbral.csv"
    #         filepath_abnormal = f"{ROOT_CSV_PATH}/{machine_type}-{machine_id}-abnormal-timbral.csv"

    #         result = process(filepath_normal, filepath_abnormal, machine_type)
    #         result["machine_type"] = machine_type
    #         result["machine_id"] = machine_id

    #         all_result.append(result)

    machine_type = sys.argv[1]
    machine_id = sys.argv[2]

    dirpath = f"../out/logs/supervised-timbral-cross-validate-hyper"
    os.makedirs(dirpath, exist_ok=True)

    logfilename = f"{machine_type}-{machine_id}.log"
    setup_logger(f"{dirpath}/{logfilename}", "supervised-timbral-cross-validate-hyper")

    print(f"machine_type: {machine_type}, machine_id: {machine_id}")
    filepath_normal = f"{ROOT_CSV_PATH}/{machine_type}-{machine_id}-normal-timbral.csv"
    filepath_abnormal = f"{ROOT_CSV_PATH}/{machine_type}-{machine_id}-abnormal-timbral.csv"

    result = process(filepath_normal, filepath_abnormal, machine_type)
    result["machine_type"] = machine_type
    result["machine_id"] = machine_id

    # store result to json
    os.makedirs("../out/results/supervised-timbral-cross-validate-hyper", exist_ok=True)
    with open(f'../out/results/supervised-timbral-cross-validate-hyper/{machine_type}-{machine_id}.json', 'w') as outfile:
        json.dump(result, outfile, cls=NpEncoder, indent=4)

if __name__ == '__main__':
    main()
