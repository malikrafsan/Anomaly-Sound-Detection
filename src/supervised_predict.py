from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

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

    # split into train and test
    df_normal_train = df_normal.sample(frac=0.8, random_state=0)
    df_normal_test = df_normal.drop(df_normal_train.index)

    df_abnormal_train = df_abnormal.sample(frac=0.8, random_state=0)
    df_abnormal_test = df_abnormal.drop(df_abnormal_train.index)

    df_train = pd.concat([df_normal_train, df_abnormal_train])
    df_test = pd.concat([df_normal_test, df_abnormal_test])

    lst_label = list(df_train['label'])
    train_label = [True if x == "abnormal" else False for x in lst_label]

    # use SVM classifier to predict test data
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(df_train[columns], train_label)
    y_pred = clf.predict(df_test[columns])

    test_label = list(df_test['label'])
    y_true = [True if x == "abnormal" else False for x in test_label]

    # scoring
    return scoring(y_true, y_pred, "SVM")


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
    with open('../out/results/supervised_learning.json', 'w') as outfile:
        json.dump(all_result, outfile, cls=NpEncoder, indent=4)

if __name__ == '__main__':
    main()
