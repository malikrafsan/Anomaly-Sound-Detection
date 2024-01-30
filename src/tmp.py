import os
import pandas as pd

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00","id_02", "id_04", "id_06"]
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

def check_freq_range():
   import librosa
   import librosa.display
   import matplotlib.pyplot as plt
   import numpy as np

   def get_frequency_range(wav_file):
      # Load the audio file
      y, sr = librosa.load(wav_file)

      # Compute the short-time Fourier transform (STFT)
      D = librosa.stft(y)

      # Compute the magnitude spectrogram
      magnitude = librosa.amplitude_to_db(librosa.magphase(D)[0], ref=np.max)

      # Display the spectrogram
      librosa.display.specshow(magnitude, y_axis='log', x_axis='time')
      plt.colorbar(format='%+2.0f dB')
      plt.title('Spectrogram')
      plt.show()

      # Get the frequency range
      frequency_range = librosa.fft_frequencies(sr=sr)

      return frequency_range

   # Replace 'your_audio_file.wav' with the path to your WAV file
   dirpath = "../../../dataset/mimii/valve/id_06/abnormal"
   files = os.listdir(dirpath)
   files = [f for f in files if f.endswith('.wav')]
   for wavfile in files:
      wavfile = os.path.join(dirpath, wavfile)
      print(wavfile)
      rng = get_frequency_range(wavfile)
      print("range:",rng, len(rng))

def check_nan():
   import numpy as np
   import pandas as pd

   dirpath = "../out/raw/surfboard"
   files = os.listdir(dirpath)
   files = [f for f in files if f.endswith('.csv')]
   files.sort()

   for f in files:
      filepath = os.path.join(dirpath, f)
      df = pd.read_csv(filepath)
      nan_count_rows = df.isna().any(axis=1).sum()
      nan_count_columns = df.isna().any(axis=0).sum()
      total_nan_count = df.isna().sum().sum()
      print(f, "       ", f"row-nan:({nan_count_rows}/{len(df)})", f"column-nan:({nan_count_columns}/{len(df.columns)})", total_nan_count)

def sort_csv():
   dirpath = "../out/processed/timbral"
   files = os.listdir(dirpath)
   files = [f for f in files if f.endswith('.csv')]

   # creaye processed/sorted dir
   sorted_dirpath = "../out/processed/sorted"
   if not os.path.exists(sorted_dirpath):
      os.mkdir(sorted_dirpath)

   for f in files:
      filepath = os.path.join(dirpath, f)
      df = pd.read_csv(filepath)
      df.sort_values(by=['filename'], inplace=True)
      df.to_csv(f"{sorted_dirpath}/{f}", index=False)

def remove_dirpath():
   dirpath = "../out/processed/sorted"
   files = os.listdir(dirpath)
   files = [f for f in files if f.endswith('.csv')]

   remove_dirpath = "../out/processed/remove"
   if not os.path.exists(remove_dirpath):
      os.mkdir(remove_dirpath)

   for f in files:
      filepath = os.path.join(dirpath, f)
      df = pd.read_csv(filepath)

      # change column filename values from "../../../dataset/mimii/valve/id_04/normal/00000000.wav" to "00000000.wav"
      df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1])

      # save to remove dir
      df.to_csv(f"{remove_dirpath}/{f}", index=False)      

def calculate_nan():
   import pandas as pd
   import sys

   dirpath = sys.argv[1]
   files = os.listdir(dirpath)

   results = []
   for file in files:
      filepath = os.path.join(dirpath, file)
      df = pd.read_csv(filepath)

      rows_with_nan = df[df.isnull().any(axis=1)].shape[0]
      columns_with_nan = df.columns[df.isnull().any()].size
      total_nan_cells = df.isnull().sum().sum()

      total_rows = df.shape[0]
      total_columns = df.shape[1]
      total_cells = df.size

      res = {
         "filename": file,
         "(rows_with_nan / total_rows)": f"{rows_with_nan} / {total_rows}",
         "(columns_with_nan / total_columns)": f"{columns_with_nan} / {total_columns}",
         "(total_nan_cells / total_cells)": f"{total_nan_cells} / {total_cells}",
      }
      results.append(res)

   results.sort(key=lambda x: x["filename"])
   df = pd.DataFrame(results)
   df.to_csv(dirpath.split("/")[-1]+".csv", index=False)

   # print(f"Rows with NaN values: {rows_with_nan}")
   # print(f"Columns with NaN values: {columns_with_nan}")
   # print(f"Total cells with NaN values: {total_nan_cells}")

   # print(f"Total rows: {total_rows}")
   # print(f"Total columns: {total_columns}")
   # print(f"Total cells: {total_cells}")

   return {

   }

def shifted_sequence(sequence, num_sequences):
    """Given a sequence (say a list) and an integer, returns a zipped iterator
    of sequence[:-num_sequences + 1], sequence[1:-num_sequences + 2], etc.

    Args:
        sequence (list or other iteratable): the sequence over which to iterate
            in various orders
        num_sequences (int): the number of sequences over which we iterate.
            Also the number of elements which come out of the output at each call.

    Returns:
        iterator: zipped shifted sequences.
    """
    return zip(
        *(
            [list(sequence[i: -num_sequences + 1 + i]) for i in range(
                num_sequences - 1)] + [sequence[num_sequences - 1:]]
        )
    )

def list_dataframe_row_with_nan():
   def getfilename(df: pd.DataFrame):
      rows_with_nan = df[df.isnull().any(axis=1)]

      # get column "filename" values from rows with nan
      nan_filenames = rows_with_nan["filename"].values.tolist()

      return nan_filenames
   
   import sys
   dirpath = sys.argv[1]

   files = os.listdir(dirpath)
   files = [f for f in files if f.endswith('.csv')]

   all_nan_filenames = []
   for f in files:
      filepath = os.path.join(dirpath, f)
      df = pd.read_csv(filepath)

      nan_filenames = getfilename(df)
      # print(f, nan_filenames)
      nan_filenames.sort()

      splitted = f.split("-")
      machine_type = splitted[1]
      machine_id = splitted[2]
      label = splitted[3]

      res = {
         "filename": f,
         "machine_type": machine_type,
         "machine_id": machine_id,
         "label": label,
         "nan_filenames": nan_filenames
      }

      all_nan_filenames.append(res)

   print(all_nan_filenames)


def calc_nan_columns():
   import sys
   import json

   dirpath = sys.argv[1]
   files = os.listdir(dirpath)
   files = [f for f in files if f.endswith('.csv')]
   files.sort()

   EXCLUDED_COLUMNS = ["filename", "label", "id_machine", "machine_type"]

   results = []
   total_normal_rows = 0
   total_abnormal_rows = 0
   total_all_rows = 0
   for file in files:
      filepath = os.path.join(dirpath, file)
      df = pd.read_csv(filepath)

      id_machine = df["id_machine"].values[0]
      machine_type = df["machine_type"].values[0]
      label = df["label"].values[0]

      df = df.drop(EXCLUDED_COLUMNS, axis=1)

      # find columns with nan and how many nan values per column into dict
      nan_cnt = df.isnull().sum().to_dict()

      # calculate percentage of nan values per column into dict
      total_rows = df.shape[0]
      nan_percentage = {}
      for k,v in nan_cnt.items():
         nan_percentage[k] = (v/total_rows)*100

      results.append({
         "filename": file,
         "nan_cnt": nan_cnt,
         "nan_percentage": nan_percentage,
         "id_machine": id_machine,
         "machine_type": machine_type,
         "label": label
      })

      total_all_rows += total_rows
      if label == "normal":
         total_normal_rows += total_rows
      else:
         total_abnormal_rows += total_rows

   # split by label
   abnormal = [r for r in results if r["label"] == "abnormal"]
   normal = [r for r in results if r["label"] == "normal"]

   # summarize data how many nan values per column
   abnormal_summary = {}
   for res in abnormal:
      for k,v in res["nan_cnt"].items():
         if k not in abnormal_summary:
            abnormal_summary[k] = 0
         abnormal_summary[k] += v
   
   # calculate percentage of nan values per column
   abnormal_percentage = {}
   for k,v in abnormal_summary.items():
      abnormal_percentage[k] = (v/total_abnormal_rows)*100
   
   normal_summary = {}
   for res in normal:
      for k,v in res["nan_cnt"].items():
         if k not in normal_summary:
            normal_summary[k] = 0
         normal_summary[k] += v

   normal_percentage = {}
   for k,v in normal_summary.items():
      normal_percentage[k] = (v/total_normal_rows)*100

   all_summary = {}
   for res in results:
      for k,v in res["nan_cnt"].items():
         if k not in all_summary:
            all_summary[k] = 0
         all_summary[k] += v

   all_percentage = {}
   for k,v in all_summary.items():
      all_percentage[k] = (v/total_all_rows)*100

   summary = {
      "abnormal": abnormal_summary,
      "abnormal_percentage": abnormal_percentage,
      "normal": normal_summary,
      "normal_percentage": normal_percentage,
      "all": all_summary
   }

   outdir = sys.argv[2]

   # save summary to json file
   with open(os.path.join(outdir, "nan-summary.json"), "w") as f:
      json.dump(summary, f, indent=4)

   # append summary to results
   results.append({
      "filename": "normal-total",
      "nan_cnt": normal_summary,
      "nan_percentage": normal_percentage,
      "id_machine": "normal-total",
      "machine_type": "normal-total",
      "label": "normal-total"
   })

   results.append({
      "filename": "abnormal-total",
      "nan_cnt": abnormal_summary,
      "nan_percentage": abnormal_percentage,
      "id_machine": "abnormal-total",
      "machine_type": "abnormal-total",
      "label": "abnormal-total"
   })

   results.append({
      "filename": "total",
      "nan_cnt": all_summary,
      "nan_percentage": all_percentage,
      "id_machine": "total",
      "machine_type": "total",
      "label": "total"
   })

   with open(os.path.join(outdir, "nan-results.json"), "w") as f:
      json.dump(results, f, indent=4)

def heatmap():
   import matplotlib.pyplot as plt
   import seaborn as sns
   import pandas as pd
   import numpy as np
   import sys

   filepath = sys.argv[1]
   df = pd.read_csv(filepath)

   # create combined columns
   df["machine-identifier"] = df["machine_type"].astype(str) + "-" + df["id_machine"].astype(str) + "-" + df["label"]

   # drop column 
   df = df.drop(["filename", "id_machine", "machine_type", "label"], axis=1)
   df = df[~df["machine-identifier"].str.contains("total")]
   
   # drop column that startswith "nan_cnt"
   df = df.loc[:, ~df.columns.str.startswith('nan_cnt')]

   heatmap_data = df.set_index("machine-identifier").T


   plt.figure(figsize=(20, 10))

   # fmt float with 2 decimal places
   sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='viridis')

   plt.xticks(rotation=45)

   # add padding to figure
   plt.tight_layout()

   # save figure
   outdir = sys.argv[2]
   plt.savefig(os.path.join(outdir, "heatmap-nan-values.png"))


def main():
   # check_csv_len()
   # check_dataset_len()
   # check_freq_range()
   # check_nan()
   # sort_csv()
   # remove_dirpath()
   # calculate_nan()

   # res = (shifted_sequence([1,2,3,4,5], 2))
   # for r in res:
   #    print(r)

   # list_dataframe_row_with_nan()
   # calc_nan_columns()
   heatmap()


if __name__ == '__main__':
  main()
