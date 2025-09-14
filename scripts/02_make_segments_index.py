import pandas as pd
from pathlib import Path
import yaml, soundfile as sf

def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    in_dir = Path(cfg["participant_wavs_dir"])
    out_csv = Path(cfg["segments_index_csv"]); out_csv.parent.mkdir(parents=True, exist_ok=True)
    win, hop, sr = cfg["win_sec"], cfg["hop_sec"], cfg["sr"]

    rows = []
    for wav in sorted(in_dir.glob("*_participant.wav")):
        pid = int(wav.name.split("_")[0])
        audio, _sr = sf.read(wav); assert _sr == sr
        dur = len(audio)/sr
        t=0.0
        while t + win <= dur + 1e-9:
            rows.append({"participant": pid, "start": round(t,3), "stop": round(t+win,3)})
            t += hop
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("segments indexed ->", out_csv)

if __name__=="__main__":
    main()































