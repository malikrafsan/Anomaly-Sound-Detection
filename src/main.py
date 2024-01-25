MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
LABELS = ["abnormal", "normal"]


def main():
  # print("python3 feature-extractor.py fan id_00 abnormal")
  with open(f"../scripts/nohup-speech-analysis.sh", "w") as f:
    for mt in MACHINE_TYPES:
      for id in MACHINE_IDS:  
        for label in LABELS:
          f.write(f"nohup python3 speech-analysis-extractor.py {mt} {id} {label} -1 > /dev/null 2>&1 &\n\n")

if __name__ == '__main__':
  main()
