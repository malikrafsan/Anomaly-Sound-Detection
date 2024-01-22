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
columns = [
    "hardness",	"depth", "brightness", "roughness",	"warmth",	"sharpness",	"boominess",	"reverb"
]


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

def process(pathnormal:str, pathabnormal: str):
    df_normal = pd.read_csv(pathnormal)
    df_abnormal = pd.read_csv(pathabnormal)

    df_normal['label'] = False
    df_abnormal['label'] = True

    df = pd.concat([df_normal, df_abnormal], ignore_index=True)

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(df[columns], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    # perform cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lst_result = []
    idx = 0
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        clf = svm.SVC(kernel='linear')  # Linear Kernel
        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_test_fold)

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

    # df_result = pd.DataFrame(lst_result)
    # mean_result = df_result.mean().to_dict()
    # std_result = df_result.std().to_dict()


    # use SVM classifier to predict test data
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # scoring
    result = scoring(y_test, y_pred, "SVM-no-cross-validate")
    # lst_result.append(result)

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
    all_result = []

    for machine_type in MACHINE_TYPES:
        for machine_id in MACHINE_IDS:
            print(f"machine_type: {machine_type}, machine_id: {machine_id}")
            filepath_normal = f"{ROOT_CSV_PATH}/{machine_type}-{machine_id}-normal-timbral.csv"
            filepath_abnormal = f"{ROOT_CSV_PATH}/{machine_type}-{machine_id}-abnormal-timbral.csv"

            result = process(filepath_normal, filepath_abnormal)
            result["machine_type"] = machine_type
            result["machine_id"] = machine_id

            all_result.append(result)

    # store result to json
    os.makedirs("../out/results", exist_ok=True)
    with open('../out/results/supervised_timbral-cross-validate.json', 'w') as outfile:
        json.dump(all_result, outfile, cls=NpEncoder, indent=4)

if __name__ == '__main__':
    main()
