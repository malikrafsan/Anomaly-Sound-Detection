#Measure pitch of all wav files in directory
import glob
import numpy as np
import pandas as pd
import parselmouth
import os
import sys

from parselmouth.praat import call

ROOT_DATASET_PATH = "../../../dataset/mimii"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
LABELS = ["abnormal", "normal"]


# This is the function to measure voice pitch
def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    print("sound", sound)
    pitch = call(sound, "To Pitch", 0, f0min, f0max) #create a praat pitch object
    print("pitch", pitch)
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    print("meanF0", meanF0)
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    print("stdevF0", stdevF0)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    print("harmonicity", harmonicity)
    hnr = call(harmonicity, "Get mean", 0, 0)
    print("hnr", hnr)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    print("pointProcess", pointProcess)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    print("localJitter", localJitter)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    print("localabsoluteJitter", localabsoluteJitter)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    print("rapJitter", rapJitter)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    print("ppq5Jitter", ppq5Jitter)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    print("ddpJitter", ddpJitter)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    print("localShimmer", localShimmer)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    print("localdbShimmer", localdbShimmer)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    print("apq3Shimmer", apq3Shimmer)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    print("aqpq5Shimmer", aqpq5Shimmer)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    print("apq11Shimmer", apq11Shimmer)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    print("ddaShimmer", ddaShimmer)

    # return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
    return {
        "meanF0": meanF0,
        "stdevF0": stdevF0,
        "hnr": hnr,
        "localJitter": localJitter,
        "localabsoluteJitter": localabsoluteJitter,
        "rapJitter": rapJitter,
        "ppq5Jitter": ppq5Jitter,
        "ddpJitter": ddpJitter,
        "localShimmer": localShimmer,
        "localdbShimmer": localdbShimmer,
        "apq3Shimmer": apq3Shimmer,
        "aqpq5Shimmer": aqpq5Shimmer,
        "apq11Shimmer": apq11Shimmer,
        "ddaShimmer": ddaShimmer
    }

def format_set(s: set):
  return ",".join([str(x) for x in list(s)])

def add_label(mp: dict, filename: str, mt: str, id_machine: str, label: str):
  mp["filename"] = filename
  mp["id_machine"] = id_machine
  mp['machine_type'] = mt
  mp["label"] = label

  return mp

def map_lstdict(mp: list, keys: list):
  data = []
  for elmt in mp:
    datum = [elmt[key] for key in keys]
    data.append(datum)

  return data

def process(
    rootdir: str,
    mt: str,
    id: str,
    label: str,
    maxdata: int = -1,
):
    MIN_FREQ = 75
    MAX_FREQ = 5000

    filenameappend = f"../out/raw/acoustic-5000/a-{mt}-{id}-{label}-({maxdata})-({MIN_FREQ}-{MAX_FREQ}).csv"
    if os.path.exists(filenameappend):
        os.remove(filenameappend)

    with open(filenameappend, "a") as appendfile:
        dirpath = f"{rootdir}/{mt}/{id}/{label}"
        files = os.listdir(dirpath)
        files = [f for f in files if f.endswith('.wav')]

        results = []
        keys = None

        rng = len(files) if maxdata == -1 else min(maxdata, len(files))
        for i in range(rng):
            file = files[i]
            filepath = f"{dirpath}/{file}"
            print(filepath)
            res = measurePitch(filepath, MIN_FREQ, MAX_FREQ, "Hertz")
            res = add_label(res, filepath, mt, id, label)

            if i == 0:
                keys = [str(i) for i in res.keys()]
                string = format_set(res.keys()) + "\n"
                appendfile.write(string)

            values = [res[key] for key in keys]
            string = ",".join ([str(i) for i in values]) + "\n"
            # string = format_set() + "\n"
            appendfile.write(string)
            appendfile.flush()

            results.append(res)

        mapped = map_lstdict(results, keys)
        df = pd.DataFrame(mapped, columns=keys)
        outname = f"../out/raw/acoustic-5000/w-{mt}-{id}-{label}-({maxdata})-({MIN_FREQ}-{MAX_FREQ}).csv"
        df.to_csv(outname, index=False)

    return filenameappend

    # dirpath = "../../../dataset/mimii/valve/id_06/abnormal"

    # results = []
    # files = os.listdir(dirpath)
    # files = [f for f in files if f.endswith('.wav')]
    # for wavfile in files:
    #     wavfile = os.path.join(dirpath, wavfile)
    #     sound = parselmouth.Sound(wavfile)
    #     res = measurePitch(sound, 75,4500, "Hertz")
    #     print(wavfile,res)
    #     results.append(res)
    
    # keys = results[0].keys() 
    # strkeys = [str(i) for i in keys]
    # df = pd.DataFrame(results, columns=strkeys)

    # df.to_csv("../tmp/praat-features.csv", index=False)
        

def main():
    mt = sys.argv[1]
    id = sys.argv[2]
    maxdata = int(sys.argv[len(sys.argv)-1])

    for label in LABELS:
        res = process(ROOT_DATASET_PATH, mt, id, label, maxdata)
        print(res)
   
    # for mt in MACHINE_TYPES:
    #     for id in MACHINE_IDS:
    #         for label in LABELS:
    #             process(ROOT_DATASET_PATH, mt, id, label, 5)

if __name__ == "__main__":
    main()

