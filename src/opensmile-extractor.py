import os
import opensmile
import pandas as pd
import sys

DATASET_DIR_PATH = "../../../dataset/mimii"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
LABELS = ["abnormal", "normal"]

def process(
  machine_type,
  machine_id,
  label,
):
  dirpath = os.path.join(
    DATASET_DIR_PATH,
    machine_type,
    machine_id,
    label,
  )

  outdir = f"../out/raw/opensmile"
  os.makedirs(outdir, exist_ok=True)

  files = os.listdir(dirpath)
  files = [f for f in files if f.endswith('.wav')]
  files.sort()

  smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
  )

  results = []
  for wavfile in files:
    wavfile = os.path.join(dirpath, wavfile)
    print(wavfile)
    res = smile.process_file(wavfile)
    res['filename'] = wavfile
    results.append(res)

  df = pd.concat(results)
  outname = f"{outdir}/opensmile-{machine_type}-{machine_id}-{label}.csv"

  df.to_csv(outname, index=False)

def main():
  # machine_type = sys.argv[1]
  # machine_id = sys.argv[2]
  # label = sys.argv[3]

  # process(machine_type, machine_id, label)

  for machine_type in MACHINE_TYPES:
    for machine_id in MACHINE_IDS:
      for label in LABELS:
        print(machine_type, machine_id, label)
        process(machine_type, machine_id, label)

if __name__ == "__main__":
  main()
