import Signal_Analysis.features.signal as sig
import sys
import os
import pandas as pd
import librosa
from Signal_Analysis.features.signal import get_F_0, get_HNR, get_Jitter, get_Pulses
import logging
import traceback
import sys

def setup_logger(
    filename: str,
    loggername: str = "my_logger",
):
    if os.path.exists(filename):
        os.remove(filename)

    class LogStream(object):
        def __init__(self, logger: logging.Logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level

        def write(self, msg):
            if msg.rstrip() != "":
                self.logger.log(self.log_level, msg.rstrip())

        def flush(self):
            for handler in self.logger.handlers:
                handler.flush()

    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    sys.stdout = LogStream(logger, logging.INFO)
    sys.stderr = LogStream(logger, logging.ERROR)

def process_wavfile(filepath: str):
  y, sr = librosa.core.load(filepath, sr=16000)
  duration = len(y)/sr
  print(duration)
  
  f0=get_F_0(y,sr)[0]
  hnr=get_HNR(y,sr)
  
  pulses=get_Pulses(y,sr)
  pulses=len(pulses) / duration

  jitter=get_Jitter(y,sr)
  jitter_dict = {}
  
  for key in jitter.keys():
    tf_key = key.replace(" ","")
    tf_key = tf_key.replace(",","-")
    jitter_dict["jitter_"+tf_key] = jitter[key]  

  return {
    "FundamentalFrequency": f0,
    "HarmonicstoNoiseRatio": hnr,
    "PulsesPerSec": pulses,
    "jitter_local": jitter_dict["jitter_local"],
    "jitter_local-absolute": jitter_dict["jitter_local-absolute"],
    "jitter_rap": jitter_dict["jitter_rap"],
    "jitter_ppq5": jitter_dict["jitter_ppq5"],
    "jitter_ddp": jitter_dict["jitter_ddp"],
  }


ROOT_DATASET_PATH = "../../../dataset/mimii"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
LABELS = ["abnormal", "normal"]

def extract(filepath: str):
  features = process_wavfile(filepath)
  return features


def main():
    loggername = "my_logger"
    filename = "output1.log"
    setup_logger(filename, loggername)

    filepath = sys.argv[1]
    df = pd.read_csv(filepath)
    df["traceback"] = [None] * len(df)
    df["errorid"] = [None] * len(df)

    # iterate over rows and extract features
    for index, row in df.iterrows():
        try:
            filepath = row["nan_filenames"]
            features = extract(filepath)
            print(features)
        except Exception:
            print("Error")
            tb =traceback.format_exc()
            print(tb)
            df.loc[index, "traceback"] = tb

            hashed = hash(tb)
            df.loc[index, "errorid"] = hashed

            print("EndError")
        print("----------------------------------------")

    df.to_csv(sys.argv[2])

if __name__ == '__main__':
  main()
