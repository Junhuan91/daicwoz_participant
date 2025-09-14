import os, pandas as pd, numpy as np
from pathlib import Path
import soundfile as sf
import yaml

def load_yaml(p):
    with open(p) as f: return yaml.safe_load(f)

def main():
    cfg = load_yaml("configs/default.yaml")
    root = Path(cfg["data_root"])
    out_dir =Path(cfg["participant_wavs_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    sr_tgt = cfg["sr"]

    meta = pd.read_csv(cfg["meta_csv"])
    for pid in meta["participant"]:
        sid = f"{pid}_P"
        wav_path = root/sid/f"{pid}_AUDIO.wav"
        csv_path = root/sid/f"{pid}_TRANSCRIPT.csv"
        if not wav_path.exists() or not csv_path.exists():
            print("skip (missing)", sid): continue
        audio, sr = sf.read(wav_path)
        if sr != sr_tgt:
            raise RuntimeError(f"{wav_path} sample rate{sr}!=cfg.sr{sr_tgt},please perform uniform sampling first.")
        df = pd.read_csv(csv_path)
        df = df[df["speaker"].str.lower()=="participant"]
        segs= []
        for _, r in df.iterrows():
            s, e = float(r["start_time"]), float(r["stop_time"])
            segs.append(audio[int(s*sr): int(e*sr)])

        merged = np.concatenate(segs) if segs else np.zeros(0,dtype=audio.dtype)
        out_wav = out_dir/f"{pid}_participant.wav"
        sf.write(out_wav,merged,sr)
        print("done",pid, "->", out_wav)

if __name__=="__main__":
    main()























