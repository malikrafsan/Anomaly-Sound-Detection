import sys
import os
import pandas as pd
from sklearn.svm import SVC

ROOT_DATASET_PATH = "../../../dataset/mimii"
ROOT_CSV_PATH = "../out/raw/surfboard"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
LABELS = ["abnormal", "normal"]
DROPPED_FEATURES = [
    "filename","id_machine","machine_type","label"
]

def score(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def main():
    mt = sys.argv[1]
    mi = sys.argv[2]

    abnormal = f"w-{mt}-{mi}-abnormal-(-1).csv"
    normal = f"w-{mt}-{mi}-normal-(-1).csv"

    abnormal = os.path.join(ROOT_CSV_PATH, abnormal)
    normal = os.path.join(ROOT_CSV_PATH, normal)

    df_abnormal = pd.read_csv(abnormal)
    df_normal = pd.read_csv(normal)

    df_normal_train = df_normal.sample(frac=0.8, random_state=0)
    df_normal_test = df_normal.drop(df_normal_train.index)

    df_abnormal_train = df_abnormal.sample(frac=0.8, random_state=0)
    df_abnormal_test = df_abnormal.drop(df_abnormal_train.index)

    df_train = pd.concat([df_normal_train, df_abnormal_train])
    df_test = pd.concat([df_normal_test, df_abnormal_test])

    df_train = df_train.sample(frac=1, random_state=0)
    df_test = df_test.sample(frac=1, random_state=0)

    train_label = df_train["label"]
    test_label = df_test["label"]

    y_train = [1 if l == "abnormal" else 0 for l in train_label]
    y_test = [1 if l == "abnormal" else 0 for l in test_label]

    clf = SVC(kernel='linear')
    clf.fit(df_train.drop(columns=DROPPED_FEATURES), y_train)

    y_pred = clf.predict(df_test.drop(columns=DROPPED_FEATURES))
    y_true = y_test

    print(score(y_true, y_pred))


if __name__ == '__main__':
    main()