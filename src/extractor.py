from lib.timbral_models.timbral_models import timbral_extractor
import pandas as pd
import sys
import os

def mapping(mp: list, keys: list):
  data = []
  for elmt in mp:
    datum = [elmt[key] for key in keys]
    data.append(datum)

  return data


def label(mp: dict, filename: str, data_type: str, mt: str, id_machine: str):
  mp["filename"] = filename
  mp["id_machine"] = id_machine
  mp['machine_type'] = mt
  mp['data_type'] = data_type
  mp["label"] = "normal" if "normal" in filename else "anomaly"

  return mp

def extract(filepath: str):
  timbre = timbral_extractor(filepath)
  return timbre


def process(
    ROOT_DATASET_PATH,
    mt_path,
    mt,
    id_machine,
    data_type,
    max_data,
):
    dir_path = f"{ROOT_DATASET_PATH}/{mt_path}/{id_machine}/{data_type}"
    files = os.listdir(dir_path)

    lst_mp = []

    rng = len(files) if max_data == -1 else min(max_data, len(files))
    for i in range(rng):
        file = files[i]
        print(f"{i}/{rng}", file)

        timbre = extract(f"{dir_path}/{file}")
        mp = label(timbre, file, data_type, mt, id_machine)
        lst_mp.append(mp)

    keys = list(lst_mp[0].keys())
    mapped = mapping(lst_mp, keys)

    df = pd.DataFrame(data=mapped, columns=keys)

    outname = f"{mt}-{id_machine}-{data_type}-timbral {max_data}.csv"
    df.to_csv(outname, index=False)

    return outname


ROOT_DATASET_PATH = "../"
mt_path = "6_dB_fan/fan"
mt = "fan"
id_machines = ["id_02"] #["id_00", "id_02", "id_04", "id_06"]
data_types = ['normal'] #["abnormal", "normal"]
max_data = -1

def main():
  result = []
  for id_machine in id_machines:
      for dt in data_types:
          outname = process(ROOT_DATASET_PATH, mt_path,
                            mt, id_machine, dt, max_data)
          print(outname)
          result.append(outname)

if __name__ == '__main__':
  main()
