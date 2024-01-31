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

def extract(filename):
    sound = parselmouth.Sound(filename)

    # Extract jitter and shimmer features
    # Note: These functions are typically used for voice analysis, so results may vary for machine sounds
    jitter_local = sound.to_jitter("local", 0.0001, 0.02, 1.3)
    jitter_rap = sound.to_jitter("rap", 0.0001, 0.02, 1.3)
    jitter_ppq5 = sound.to_jitter("ppq5", 0.0001, 0.02, 1.3)
    shimmer_local = sound.to_shimmer("local", 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = sound.to_shimmer("apq3", 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = sound.to_shimmer("apq5", 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11 = sound.to_shimmer("apq11", 0.0001, 0.02, 1.3, 1.6)

    return {
        "jitter_local": jitter_local,
        "jitter_rap": jitter_rap,
        "jitter_ppq5": jitter_ppq5,
        "shimmer_local": shimmer_local,
        "shimmer_apq3": shimmer_apq3,
        "shimmer_apq5": shimmer_apq5,
        "shimmer_apq11": shimmer_apq11,
    }

def main():
    import sys

    filename = sys.argv[1]
    print(extract(filename))

if __name__ == "__main__":
    main()
