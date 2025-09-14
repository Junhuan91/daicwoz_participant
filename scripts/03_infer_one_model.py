# scripts/03_infer.py
import argparse, yaml, os
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

def load_cfg():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)

# ----------- Audio embedding backend -----------
class HFAudioEmbedding:
    def __init__(self, ckpt, layer=-1, pooling="mean", device=None):
        from transformers import AutoProcessor, AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(ckpt)
        # 用 AutoModel 拿 hidden_states（通用表征）
        self.model = AutoModel.from_pretrained(ckpt, output_hidden_states=True).to(self.device).eval()
        self.layer = layer
        self.pooling = pooling

    @torch.no_grad()
    def embed_batch(self, waves, sr):
        # processor 负责对齐 padding
        inputs = self.proc(waves, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # 取 hidden_states 指定层：[B, T, D]
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hs = out.hidden_states[self.layer]
        else:
            # 兜底：部分模型没有 hidden_states 时用 last_hidden_state
            hs = out.last_hidden_state
        # 池化到 [B, D]
        if self.pooling == "mean":
            emb = hs.mean(dim=1)
        else:
            raise NotImplementedError("Only mean pooling implemented.")
        return emb.detach().cpu().numpy()  # (B, D)

# ----------- Text classify backend -----------
class HFTextClassifier:
    def __init__(self, ckpt, target_label=None, device=None, max_length=512):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(self.device).eval()
        self.id2label = self.model.config.id2label
        # label 名到索引
        self.target_ix = None
        if target_label is not None:
            self.target_ix = {v.lower(): k for k, v in self.id2label.items()}.get(target_label.lower(), None)
        self.max_length = max_length

    @torch.no_grad()
    def proba_batch(self, texts):
        import torch
        enc = self.tok(
            texts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        if self.target_ix is None:
            # 没指定就取 index=1 作为“抑郁”占位
            return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        return probs[:, self.target_ix]

# ----------- 小头（LogReg/SVM） -----------
def load_head(head_path):
    import joblib
    return joblib.load(head_path)

def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]

def infer_audio(cfg, mname, mconf, batch=16):
    # 路径
    seg_idx = pd.read_csv(cfg["segments_index_csv"])          # participant, start, stop
    meta = pd.read_csv(cfg["meta_csv"])                       # participant, target, subset
    pid2subset = dict(zip(meta["participant"], meta["subset"]))
    pid2target = dict(zip(meta["participant"], meta["target"]))
    sr = cfg["sr"]

    # 模型与小头
    emb = HFAudioEmbedding(
        ckpt=mconf["ckpt"],
        layer=mconf.get("layer", -1),
        pooling=mconf.get("pooling", "mean"),
    )
    head_path = Path(cfg["work_root"]) / "heads" / f"{mname}.joblib"
    if not head_path.exists():
        raise FileNotFoundError(f"未找到小头 {head_path}，请先运行 train_head.py --model {mname}")
    head = load_head(head_path)

    out_rows = []
    # 逐受试者跑
    for pid, g in tqdm(seg_idx.groupby("participant"), desc=f"[{mname}] audio segments"):
        wav_path = Path(cfg["participant_wavs_dir"]) / f"{pid}_participant.wav"
        if not wav_path.exists():
            continue
        wav, _sr = sf.read(wav_path)
        assert _sr == sr, f"{wav_path} sr={_sr}, cfg.sr={sr}"

        starts = g["start"].tolist()
        stops  = g["stop"].tolist()

        # 分批做 embedding + 小头概率
        for bi in batched(list(range(len(starts))), batch):
            waves = [wav[int(starts[i]*sr):int(stops[i]*sr)] for i in bi]
            if len(waves) == 0:
                continue
            X = emb.embed_batch(waves, sr=sr)        # (B, D)
            probs = head.predict_proba(X)[:, 1]      # 段级“抑郁”概率
            for local_idx, p in zip(bi, probs):
                out_rows.append({
                    "participant": int(pid),
                    "subset": pid2subset[int(pid)],
                    "target": int(pid2target[int(pid)]),
                    "start": float(starts[local_idx]),
                    "stop":  float(stops[local_idx]),
                    "prob":  float(p),
                })

    df = pd.DataFrame(out_rows)
    out_csv = Path(cfg["work_root"]) / f"segment_probs__{mname}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)

def infer_text(cfg, mname, mconf, batch=32):
    data_root = Path(cfg["data_root"])
    meta = pd.read_csv(cfg["meta_csv"])
    clf = HFTextClassifier(
        ckpt=mconf["ckpt"],
        target_label=mconf.get("target_label"),
        max_length=mconf.get("max_length", 512)
    )

    rows = []
    for pid, subset, target in tqdm(meta[["participant","subset","target"]].itertuples(index=False), desc=f"[{mname}] text"):
        sid = f"{pid}_P"
        tpath = data_root / sid / f"{pid}_TRANSCRIPT.csv"
        if not tpath.exists():
            continue
        df = pd.read_csv(tpath)
        df = df[df["speaker"].str.lower()=="participant"]
        texts = df["value"].fillna("").astype(str).tolist()
        if not texts:
            continue

        # 分批跑
        idxs = list(range(len(texts)))
        for bi in batched(idxs, batch):
            probs = clf.proba_batch([texts[i] for i in bi])
            for i, p in zip(bi, probs):
                rows.append({
                    "participant": int(pid),
                    "subset": subset,
                    "target": int(target),
                    "text_id": int(i),
                    "prob": float(p),
                })

    out = pd.DataFrame(rows)
    out_csv = Path(cfg["work_root"]) / f"segment_probs__{mname}.csv"
    out.to_csv(out_csv, index=False)
    print("saved:", out_csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="configs/default.yaml 里的 models 键名")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    cfg = load_cfg()
    mconf = cfg["models"][args.model]
    backend = mconf["backend"]

    if backend == "hf_audio_embedding":
        infer_audio(cfg, args.model, mconf, batch=args.batch)
    elif backend == "hf_text_classify":
        infer_text(cfg, args.model, mconf, batch=args.batch)
    else:
        raise ValueError(f"未知 backend: {backend}")

if __name__ == "__main__":
    main()
