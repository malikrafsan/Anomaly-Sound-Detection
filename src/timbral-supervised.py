import sys
import os
import pandas as pd
from sklearn.svm import SVC

# ROOT_DATASET_PATH = "../../../dataset/mimii"
ROOT_CSV_PATH = "../out/processed/final-timbral"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
LABELS = ["abnormal", "normal"]
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

def score(y_true, y_pred):
    # use lib to calculate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def process(machine_type: str, machine_id: str):
    files = os.listdir(f"{ROOT_CSV_PATH}")
    files = [f for f in files if f.endswith(".csv")]
    files = [f for f in files if machine_type in f]
    files = [f for f in files if machine_id in f]

    abnormal = [f for f in files if "abnormal" in f][0]
    normal = [f for f in files if "normal" in f][0]

    print(abnormal)
    print(normal)

    abnormal = os.path.join(ROOT_CSV_PATH, abnormal)
    normal = os.path.join(ROOT_CSV_PATH, normal)

    df_abnormal = pd.read_csv(abnormal)
    df_normal = pd.read_csv(normal)

    df_abnormal = df_abnormal[MAPPING_FEATURES[machine_type]]
    df_normal = df_normal[MAPPING_FEATURES[machine_type]]

    df_abnormal["label"] = 1
    df_normal["label"] = 0

    # split into train and test per label
    df_normal_train = df_normal.sample(frac=0.8, random_state=0)
    df_normal_test = df_normal.drop(df_normal_train.index)

    df_abnormal_train = df_abnormal.sample(frac=0.8, random_state=0)
    df_abnormal_test = df_abnormal.drop(df_abnormal_train.index)

    # concat train and test
    df_train = pd.concat([df_normal_train, df_abnormal_train])
    df_test = pd.concat([df_normal_test, df_abnormal_test])

    # shuffle
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    # split into X and y
    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]

    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # train
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(score(y_test, y_pred))


    # df = pd.concat([df_abnormal, df_normal], ignore_index=True)
    # df = df.sample(frac=1).reset_index(drop=True)

    # X = df.drop(columns=["label"])
    # y = df["label"]

    # clf = SVC()
    # clf.fit(X, y)

    # y_pred = clf.predict(X)

    # print(score(y, y_pred))

def main():
    # mt = sys.argv[1]
    # mi = sys.argv[2]

    # process(mt, mi)
    for mt in MACHINE_TYPES:
        for mi in MACHINE_IDS:
            print(mt, mi)
            process(mt, mi)
            print("--------------------")

if __name__ == '__main__':
    main()