import os
import pandas as pd

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_04", "id_06"]
LABELS = ["abnormal", "normal"]

def check_csv_len():
    dirpath = "../out"

    files = os.listdir(dirpath)
    wfiles = [f"{dirpath}/{x}" for x in files if x.startswith("w-")]
    wfiles.sort()

    def check_one_csv(filepath: str):
        df = pd.read_csv(filepath)
        return len(df)
    
    ln = [(x,check_one_csv(x)) for x in wfiles]
    for l in ln:
       print(l)

def check_dataset_len():
   dirpath = "../../../dataset/mimii"
   
   for mt in MACHINE_TYPES:
      for id in MACHINE_IDS:
         for lb in LABELS:
            files = os.listdir(f"{dirpath}/{mt}/{id}/{lb}")
            print(mt,id,lb,len(files))

def main():
   check_csv_len()
#    check_dataset_len()


if __name__ == '__main__':
  main()
