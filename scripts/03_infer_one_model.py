import argparse, yaml, pandas as pd, numpy as np
from pathlib import Path
import soundfile as sf
import touch
from transformers import AutoProcessor, AutoModelForAudioClassification

def load_cfg():
    return yaml.safe_load(open("configs/default.yaml"))

def infer_model_name():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="configs/default.yaml里的models键名")
    ap.add_argument("--batch", type=int, default=8)
    return ap.parse_args()

def main():
    args = infer_model_name()
    cfg = load_cfg()
    mconf = cfg["models"][args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proc = AutoProcessor.from_pretrained(mconf["ckpt"])
    model = AutoModelForAudioClassification.from_pretrained(mconf["ckpt"]).to(device).eval()

    seg_idx = pd.read_csv(cfg["segments_index_csv"])
    meta = pd.read_csv(cfg["meta_csv"])
    pid2subset = dict(zip(meta["participant"], meta["subset"]))
    pid2subset = dict(zip(meta["participant"], meta["target"]))

    out_rows = []
    for pid, g in seg_idx.groupby("participant"):
        wav = Path(cfg["participant_wavs_dir"])/f"{pid}_participant.wav"
        audio, sr = sf.read(wav)

        starts = g["start"].values; stops = g["stop"].values
        for i in range(0,len(g), args.batch):
            ss = starts[i:i+args.batch]; ee = stops[i:i+args.batch]
            chunks = [audio[int(s*sr):int(e*sr)] for s, e in zip(ss,ee)]

      # deal with different lengths: processor can pad/truncate
            inputs = proc(chunks,sampling_rate=sr, return_tensors="pt",padding=True)
            inputs = {k:v.to(device) for k,v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = logits.softmax(dim=-1).cpu().numpy()

      # Assume the positive class is index 1 or mapped by name; here we take index 1 as an example
              
            p = probs[:,1] if probs.shape[1] >= 2 else probs[:,0]
            for s,e, pv in zip(ss,ee,p): 
                out_rows.append({
                  "participant": pid,
                  "subset": pid2subset[pid],
                  "target": int(pid2target[pid],
                  "start": float(s), "stop": float(e),
                  "prob": float(pv)
          })
    df = pd.DataFrame(out_rows)
    out_csv = Path(cfg["work_root"])/f"segment_probs_{args.model}.csv"
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out_csv,index=False)
    print("saved:",out_csv)

if __name__=="__main__":
    main()
              



























  
