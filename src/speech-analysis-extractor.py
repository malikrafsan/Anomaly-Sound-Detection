import Signal_Analysis.features.signal as sig
import sys
import os
import pandas as pd
import librosa
from Signal_Analysis.features.signal import get_F_0, get_HNR, get_Jitter, get_Pulses

def process_wavfile(filepath: str):
  y, sr = librosa.core.load(filepath, sr=16000)
  duration = len(y)/sr
  print(duration)

  try:  
    f0=get_F_0(y,sr)[0]
  except:
    f0=None

  try:
    hnr=get_HNR(y,sr)
  except:
    hnr=None

  try:
    jitter=get_Jitter(y,sr)
    # add prefix to jitter keys
    jitter_dict = {}
    for key in jitter.keys():
      tf_key = key.replace(" ","")
      tf_key = tf_key.replace(",","-")
      jitter_dict["jitter_"+tf_key] = jitter[key]
  except:
    jitter_dict={
      "jitter_local":None,
      "jitter_local-absolute":None,
      "jitter_rap":None,
      "jitter_ppq5":None,
      "jitter_ddp":None
    }
  # jitter_features=list(jitter.values())
  # jitter_labels=list(jitter)
  try:
    pulses=get_Pulses(y,sr)
    pulses=len(pulses) / duration
  except:
    pulses=None

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

  # features=[f0,hnr,pulses]+jitter_features
  # labels=['FundamentalFrequency','HarmonicstoNoiseRatio','PulsesPerSec']+jitter_labels

  # res = dict(zip(labels,features))
  # return res

    # sound = Waveform(path=filepath, sample_rate=16000)

    # shimmers = sound.shimmers()
    # print("shimmers", shimmers)
    # jitters = sound.jitters()
    # print("jitters", jitters)
    # formants = sound.formants()
    # print("formants", formants)
    # f0_statistics = sound.f0_statistics()
    # print("f0_statistics", f0_statistics)

    # return {
    #     **shimmers,
    #     **jitters,
    #     **formants,
    #     **f0_statistics,
    # }

    # spectral_centroid = sound.spectral_centroid()
    # print("spectral_centroid", spectral_centroid)
    # spectral_entropy = sound.spectral_entropy()
    # print("spectral_entropy", spectral_entropy)
    # spectral_flatness = sound.spectral_flatness()
    # print("spectral_flatness", spectral_flatness)
    # spectral_flux = sound.spectral_flux()
    # print("spectral_flux", spectral_flux)
    # spectral_rolloff = sound.spectral_rolloff()
    # print("spectral_rolloff", spectral_rolloff)
    # spectral_spread = sound.spectral_spread()
    # print("spectral_spread", spectral_spread)
    # spectral_kurtosis = sound.spectral_kurtosis()
    # print("spectral_kurtosis", spectral_kurtosis)
    # spectral_skewness = sound.spectral_skewness()
    # print("spectral_skewness", spectral_skewness)
    # spectral_slope = sound.spectral_slope()
    # print("spectral_slope", spectral_slope)
    
    # f0_contour = sound.f0_contour()
    # print("f0_contour", f0_contour)

ROOT_DATASET_PATH = "../../../dataset/mimii"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
LABELS = ["abnormal", "normal"]

# def main():
#     filepath = sys.argv[1]
#     res = process_wavfile(filepath)
#     print(res)


def extract(filepath: str):
  features = process_wavfile(filepath)
  return features

def mapping(mp: list, keys: list):
  data = []
  for elmt in mp:
    datum = [elmt[key] for key in keys]
    data.append(datum)

  return data

def labeling(mp: dict, filename: str, mt: str, id_machine: str, label: str):
  mp["filename"] = filename
  mp["id_machine"] = id_machine
  mp['machine_type'] = mt
  mp["label"] = label

  return mp

def format_set(s: set):
  return ",".join([str(x) for x in list(s)])

def process(
    rootdir: str,
    mt: str,
    id: str,
    label: str,
    maxdata: int = -1,
):
  filenameappend = f"../out/raw/speech-analysis/a-{mt}-{id}-{label}-({maxdata}).csv"

  if os.path.exists(filenameappend):
    os.remove(filenameappend)

  with open(filenameappend, "a") as appendfile:
    dirpath = f"{rootdir}/{mt}/{id}/{label}"
    files = os.listdir(dirpath)

    lstmp = []

    rng = len(files) if maxdata == -1 else min(maxdata, len(files))
    for i in range(rng):
      file = files[i]
      filepath = f"{dirpath}/{file}"
      print("filepath",filepath)

      timbre = extract(filepath)
      mp = labeling(timbre, filepath, mt, id, label)

      if i == 0:
        string = format_set(mp.keys()) + "\n"
        appendfile.write(string)
        print(string)
      
      string = format_set(mp.values()) + "\n"
      appendfile.write(string)
      print(string)
      appendfile.flush()

      lstmp.append(mp)

    keys = list(lstmp[0].keys())
    mapped = mapping(lstmp, keys)

    df = pd.DataFrame(data=mapped, columns=keys)
    outname = f"../out/raw/speech-analysis/w-{mt}-{id}-{label}-({maxdata}).csv"
    df.to_csv(outname, index=False)

    return outname

def main():
  mt = sys.argv[1]
  id = sys.argv[2]
  maxdata = int(sys.argv[len(sys.argv)-1])

  for label in LABELS:
    res = process(ROOT_DATASET_PATH, mt, id, label, maxdata)
    print("res",res)

  # for mt in MACHINE_TYPES:
  #   for id in MACHINE_IDS:
  #     for label in LABELS:
  #       res = process(ROOT_DATASET_PATH, mt, id, label, 2)
  #       print("res",res)

  # res = extract("../../../dataset/mimii/fan/id_00/abnormal/00000071.wav")
  # print("res",res)

if __name__ == '__main__':
  main()
