import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot(
  dirpath: str,
  machine_type: str,
  machine_id: str,
  filepath1: str,
  filepath2: str,
):
  df1 = pd.read_csv(filepath1)
  df2 = pd.read_csv(filepath2)
  
  df = pd.concat([df1, df2])
  
  DROPPED_COLUMNS = ["filename", "machine_type","machine_id"]
  df["label"] = df["label"].astype("category").cat.codes
  df = df.drop(DROPPED_COLUMNS, axis=1)
  
  corr = df.corr()
  plt.figure(figsize=(12, 12))

  heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
  heatmap.set_title(f'{machine_type.upper()} {machine_id.upper()} Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
  
  # make annotation only show 2 decimal places
  for text in heatmap.texts:
    text.set_text(f"{float(text.get_text()):.2f}")

  outdir = f"{dirpath}/corrheatmap"
  os.makedirs(outdir, exist_ok=True)
  
  plt.savefig(f"{outdir}/{machine_type}-{machine_id}.png")
  plt.clf()
   
def calculate_mean_correlation(
  dirpath: str,
  machine_type: str,
  machine_id: str,
  filepath1: str,
  filepath2: str,
):
  df1 = pd.read_csv(filepath1)
  df2 = pd.read_csv(filepath2)
  
  df = pd.concat([df1, df2])
  
  DROPPED_COLUMNS = ["filename", "machine_type","machine_id"]
  df["label"] = df["label"].astype("category").cat.codes
  df = df.drop(DROPPED_COLUMNS, axis=1)
  
  corr = df.corr()

  # calculate mean correlation for label column
  corr_label = corr["label"]
  corr_label = corr_label.drop("label")
  corr_label_mean = corr_label.max()

  return corr_label_mean

def main():
  dirpath = sys.argv[1]
  
  MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
  MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
  LABELS = ["normal", "abnormal"]
  
  # for machine_type in MACHINE_TYPES:
  #   for machine_id in MACHINE_IDS:
  #     filepath1 = f"{dirpath}/{machine_type}-{machine_id}-{LABELS[0]}.csv"
  #     filepath2 = f"{dirpath}/{machine_type}-{machine_id}-{LABELS[1]}.csv"
  #     plot(dirpath, machine_type, machine_id, filepath1, filepath2)

  # calculate mean correlation for each machine type and machine id
  for machine_type in MACHINE_TYPES:
    for machine_id in MACHINE_IDS:
      filepath1 = f"{dirpath}/{machine_type}-{machine_id}-{LABELS[0]}.csv"
      filepath2 = f"{dirpath}/{machine_type}-{machine_id}-{LABELS[1]}.csv"
      corr_label_mean = calculate_mean_correlation(dirpath, machine_type, machine_id, filepath1, filepath2)
      print(f"{machine_type}-{machine_id}: {corr_label_mean}")

if __name__ == '__main__':
  main()
  