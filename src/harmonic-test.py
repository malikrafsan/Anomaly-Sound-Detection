import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
import os
import json
import pandas as pd

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if np.iscomplexobj(obj):
            return abs(obj)
        return super(NpEncoder, self).default(obj)


def harmonic_test_rfft(audio_file, tolerance=0.1):
    y, sr = librosa.load(audio_file, sr=None)

    fft_result = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1/sr)

    fundamental_freq = freqs[np.argmax(np.abs(fft_result))]
    harmonic_frequencies = [fundamental_freq * (n + 1) for n in range(1, 6)]

    for harmonic in harmonic_frequencies:
        if not any(abs(freqs - harmonic) < tolerance):
            return False, fundamental_freq, harmonic_frequencies

    return True, fundamental_freq, harmonic_frequencies

def harmonic_test_fft(filename:str):
    signal, sampling_rate = librosa.load(filename, sr=None)
    
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)

    fundamental_freq = np.abs(frequencies[np.argmax(np.abs(fft_result))])
    harmonics = [fundamental_freq * i for i in range(2, 6)]
    
    is_harmonic = all(np.any(np.isclose(frequencies, harmonic, atol=5)) 
                        for harmonic in harmonics)

    return is_harmonic, fundamental_freq, harmonics


ROOT_DATASET_PATH = "../../../dataset/mimii"
machine_types = [
    "fan",
    "pump",
    "slider",
    "valve"
]
machine_ids = ["id_00", "id_02", "id_04", "id_06", ]
labels = ["abnormal","normal"]


# Load a sound file (replace 'path_to_file' with your file path)
# path_to_file = sys.argv[1]
# process(path_to_file)

def process(filename: str):
    print(filename, end=",")
    res = {}

    is_harmonic, fundamental_freq, harmonics = harmonic_test_rfft(filename)
    res["harmonic_test_rfft.is_harmonic"] = is_harmonic
    res["harmonic_test_rfft.fundamental_freq"] = fundamental_freq
    
    if not is_harmonic:
        print(f"\n\n{filename} is not harmonic (rfft)\n\n")

    is_harmonic, fundamental_freq, harmonics = harmonic_test_fft(filename)
    res["harmonic_test_fft.is_harmonic"] = is_harmonic
    res["harmonic_test_fft.fundamental_freq"] = fundamental_freq
    
    if not is_harmonic:
        print(f"\n\n{filename} is not harmonic (fft)\n\n")

    return res    



def handle(
    outdir: str,
    machine_type: str,
    machine_id: str,
    label: str,
):
    dirpath = f"{ROOT_DATASET_PATH}/{machine_type}/{machine_id}/{label}"
    files = os.listdir(dirpath)
    files.sort()

    results = []
    # for file in files:
    for i in range(len(files)):
        file = files[i]

        path_to_file = f"{dirpath}/{file}"
        res = process(path_to_file)
        res["filename"] = file
        res["machine_type"] = machine_type
        res["machine_id"] = machine_id
        res["label"] = label
    
        results.append(res)
        
    df = pd.DataFrame(results)
    
    outname = f"{outdir}/{machine_type}-{machine_id}-{label}.csv"
    df.to_csv(outname, index=False)

def main():
    outdir = "../out/results/harmonic-test"
    os.makedirs(outdir, exist_ok=True)

    # for machine_type in machine_types:
    #     for machine_id in machine_ids:
    #         for label in labels:
    #             handle(outdir, machine_type, machine_id, label)
    
    
    machine_type = sys.argv[1]
    machine_id = sys.argv[2]
    
    for label in labels:
        handle(outdir, machine_type, machine_id, label)
    
    print("Done!")

if __name__ == "__main__":
    main()

    



# # Plotting
# plt.figure(figsize=(10, 4))
# librosa.display.waveshow(signal, sr=sr)
# plt.title('Waveform')
# plt.show()
