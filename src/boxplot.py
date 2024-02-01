import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def stats():
  dirpath = sys.argv[2]
  files = os.listdir(dirpath)
  files.sort()
  csvs = [f for f in files if f.endswith(".csv")]
  
  keys = ["fan", "pump", "slider", "valve"]
  files = [
    [f for f in csvs if f.startswith(key)]
    for key in keys
  ]
    
  for i in range(len(files)):  
    csv_files = files[i]
    
    df_lst = [pd.read_csv(f"{dirpath}/{f}") for f in csv_files]
    
    df_combined = pd.concat(df_lst)
    identifier = f"{keys[i]}-identifier"
    df_combined[identifier] = df_combined["machine_id"] + "-" + df_combined["label"]
    
    # mean, std, min, max of all identifiers
    print("Mean, Std, Min, Max of all identifiers")
    aggall = df_combined["harmonic_test_fft.fundamental_freq"].agg(["mean", "std", "min", "max"])
    print(aggall)
    print()
    
    # create table of mean, std, min, and max
    print("Mean, Std, Min, Max")
    agg = df_combined.groupby(identifier)["harmonic_test_fft.fundamental_freq"].agg(["mean", "std", "min", "max"])
    # append dataframe aggall to dataframe agg
    agg.loc["all"] = aggall
    print(agg)
    print()
    

    outdir = f"{dirpath}/stats"
    os.makedirs(outdir, exist_ok=True)    
    agg.to_csv(f"{outdir}/{keys[i]}.csv")

def plot():
  dirpath = sys.argv[2]
  files = os.listdir(dirpath)
  files.sort()
  csvs = [f for f in files if f.endswith(".csv")]
  
  keys = ["fan", "pump", "slider", "valve"]
  files = [
    [f for f in csvs if f.startswith(key)]
    for key in keys
  ]
    
  for i in range(len(files)):  
    csv_files = files[i]
    
    df_lst = [pd.read_csv(f"{dirpath}/{f}") for f in csv_files]
    
    df_combined = pd.concat(df_lst)
    df_combined["identifier"] = df_combined["machine_id"] + "-" + df_combined["label"]
    
    # calculate statistics of fundamental frequency
    print("Statistics of Fundamental Frequency")
    print(df_combined.groupby("identifier")["harmonic_test_fft.fundamental_freq"].describe())
    print()
    
def main():
  import sys
  
  if len(sys.argv) < 2:
    print("Usage: python3 boxplot.py <path_to_csv>")
    return
  
  cmd = sys.argv[1]
  if cmd == "stats":
    stats()
  elif cmd == "plot":
    plot()
  else:
    print("Unknown command")

if __name__ == "__main__":
  main()
