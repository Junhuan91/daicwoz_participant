import argparse, yaml, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,precision_recall_curve,auc

def pick_tau_by_f1(y_true,y_prob):
    ps,rs,th = precision_recall_curve(y_true,y_prob)
    th = np.r_[0.0,th]
    f1s = 2*ps*rs/(ps+rs+1e-12)
    return float(th[np.nanargmax(f1s)])
def eval_split(y_true,y_prob, tau):
    y_hat = (y_prob >= tau).astype(int)
    ps, rs, _ = precision_recall_curve(y_true,y_prob)
    return dict(
        ACC=accuracy_score(y_true,y_hat),
        F1=f1_score(y_true,y_hat),
        BAcc=balanced_accuracy_score(y_true,y_hat),
        PR_AUC=auc(rs,ps)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    seg_csv = Path(cfg["work_root"])/f"segment_probs__{args.model}.csv"
    df = pd.read_csv(seg_csv)
  # subjects probability = segments average
    subj = df.groupby(["participant","subset","target"])["prob"].mean().reset_index()
    subj = subj.rename(columns={"prob":"P"})

    dev = subj[subj["subset"]=="dev"]
    tau = pick_tau_by_f1(dev["target"].values,dev["P"].values)
    out_lines=[]
    for split in ["train","dev","test"]:
        part = subj[subj["subset"]==split]
        m = eval_split(part["target"].values,part["P"].values,tau)
        out_lines.append((split.upper(),m))
    print(f"[{args.model}] tau={tau: .3f}")

    for sp,m in out_lines:
        print(sp,"ACC={ACC:.3f} F1={F1:.3f} BAcc={BAcc:.3f} PR-AUC={PR_AUC:.3f}".format(**m))

    out_csv = Path(cfg["work_root"])/f"subject_probs__{args.model}.csv"
    subj.to_csv(out_csv,index=False)


if __name__=="__main__":
    main()



















  
