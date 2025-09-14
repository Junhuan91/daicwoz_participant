# scripts/train_head.py
import argparse, yaml
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

def load_cfg():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)

class HFAudioEmbedding:
    def __init__(self, ckpt, layer=-1, pooling="mean", device=None):
        import torch
        from transformers import AutoProcessor, AutoModel
        self.torch = __import__("torch")
        self.device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt, output_hidden_states=True).to(self.device).eval()
        self.layer = layer
        self.pooling = pooling

    @torch.no_grad()
    def embed_batch(self, waves, sr):
        import torch
        inputs = self.proc(waves, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        out = self.model(**inputs)
        hs = out.hidden_states[self.layer] if out.hidden_states is not None else out.last_hidden_state
        if self.pooling == "mean":
            emb = hs.mean(dim=1)
        else:
            raise NotImplementedError("Only mean pooling implemented.")
        return emb.detach().cpu().numpy()

def batched(idx, bs):
    for i in range(0, len(idx), bs):
        yield idx[i:i+bs]

def main():
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="configs/default.yaml 里的 models 键名（须是 hf_audio_embedding）")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    cfg = load_cfg()
    mconf = cfg["models"][args.model]
    assert mconf["backend"] == "hf_audio_embedding"

    # read index and meta_info
    seg = pd.read_csv(cfg["segments_index_csv"])          # participant, start, stop
    meta = pd.read_csv(cfg["meta_csv"])                   # participant, target, subset
    pid2subset = dict(zip(meta["participant"], meta["subset"]))
    pid2target = dict(zip(meta["participant"], meta["target"]))

    seg["subset"]  = seg["participant"].map(pid2subset)
    seg["target"]  = seg["participant"].map(pid2target)
    seg_train = seg[seg["subset"]=="train"].copy()

    # Extract embeddings from train segments
    emb_extractor = HFAudioEmbedding(
        ckpt=mconf["ckpt"], layer=mconf.get("layer",-1), pooling=mconf.get("pooling","mean")
    )
    sr = cfg["sr"]
    Xs, ys = [], []
    for pid, g in tqdm(seg_train.groupby("participant"), desc=f"[{args.model}] build train set"):
        wav_path = Path(cfg["participant_wavs_dir"]) / f"{pid}_participant.wav"
        if not wav_path.exists(): 
            continue
        wav, _sr = sf.read(wav_path); assert _sr == sr
        starts = g["start"].to_numpy(); stops = g["stop"].to_numpy()
        labs   = g["target"].to_numpy()
        idx = list(range(len(starts)))
        for bi in batched(idx, args.batch):
            waves = [wav[int(starts[i]*sr):int(stops[i]*sr)] for i in bi]
            if not waves: continue
            E = emb_extractor.embed_batch(waves, sr=sr)   # (B, D)
            Xs.append(E); ys.extend(labs[bi])
    if not Xs:
        raise RuntimeError("Training set is empty, check data and segmentation.")

    X = np.vstack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=int)

    # read SVM configuration
    svm_cfg = (cfg.get("svm_head") or {})
    C = float(svm_cfg.get("C", 1.0))
    cls_w = svm_cfg.get("class_weight", "balanced")
    use_calib = bool(svm_cfg.get("use_calibration", False))

    # Linear SVM + standard；two paths offer predict_proba
    if use_calib:
        # LinearSVC don't support probability；use CalibratedClassifierCV 包一层（Platt）
        base = LinearSVC(C=C, class_weight=cls_w)
        clf = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("cal", CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=5)),
        ])
    else:
        # use SVC(kernel='linear', probability=True) directly
        clf = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(kernel="linear", C=C, class_weight=cls_w, probability=True)),
        ])

    clf.fit(X, y)

    out_dir = Path(cfg["work_root"]) / "heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.joblib"
    joblib.dump(clf, out_path)
    print(f"saved SVM head -> {out_path}")

if __name__ == "__main__":
    main()
