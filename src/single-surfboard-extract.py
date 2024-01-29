from surfboard.sound import Waveform
import sys
import os
import pandas as pd

def process_wavfile(filepath: str):
    sound = Waveform(path=filepath, sample_rate=16000)

    shimmers = sound.shimmers()
    print("shimmers", shimmers)
    jitters = sound.jitters()
    print("jitters", jitters)
    formants = sound.formants()
    print("formants", formants)
    f0_statistics = sound.f0_statistics()
    print("f0_statistics", f0_statistics)

    return {
        **shimmers,
        **jitters,
        **formants,
        **f0_statistics,
    }

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
  filenameappend = f"../out/raw/surfboard/a-{mt}-{id}-{label}-({maxdata}).csv"

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
    outname = f"../out/raw/surfboard/w-{mt}-{id}-{label}-({maxdata}).csv"
    df.to_csv(outname, index=False)

    return outname

def main():
  # mt = MACHINE_TYPES[0]
  # id = MACHINE_IDS[0]
  # label = LABELS[0]

  filepath = "/home/s2316079/dataset/mimii/fan/id_02/abnormal/00000292.wav"
  result = extract(filepath)
  print("result",result)

if __name__ == '__main__':
  main()
