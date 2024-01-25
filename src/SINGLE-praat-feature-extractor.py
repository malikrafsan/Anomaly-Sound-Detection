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
        

def main():
    filepath = sys.argv[1]

    res = measurePitch(filepath, 75, 600, "Hertz")
    print("res",res)
    


if __name__ == "__main__":
    main()

