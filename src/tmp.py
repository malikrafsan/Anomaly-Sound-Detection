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



def main():
   # check_csv_len()
   check_dataset_len()
   # check_freq_range()
   # check_nan()
   # sort_csv()
   # remove_dirpath()

if __name__ == '__main__':
  main()
