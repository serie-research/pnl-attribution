"""Recompute report tables from one isolated set of unit parquets."""
import argparse
from pathlib import Path
import polars as pl
from .figures import _load, shares


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    args = p.parse_args()
    df = _load(args.results)
    sh = shares(df)
    ref = sh.filter(pl.col("method") == "durin").select("unit", "seed", "signal", ref="share")
    dist = (sh.join(ref, on=["unit", "seed", "signal"])
            .group_by("config", "unit", "seed", "method")
            .agg(value=0.5*(pl.col("share")-pl.col("ref")).abs().sum()))
    def avg(frame, config, method, metric=None):
        q = frame.filter((pl.col("config") == config) & (pl.col("method") == method))
        if metric:
            q = q.filter(pl.col("metric") == metric)
        v = q["value"].mean()
        return "—" if v is None else f"{v:.4f}"
    cfgs = sorted(df["config"].unique())
    print("| configuration | NNLS distance | residual variance | converged | zero betas | cost/gross vol |")
    print("|---|---:|---:|---:|---:|---:|")
    for c in cfgs:
        vals = [avg(dist,c,"iterated"), avg(df,c,"iterated","var_share"),
                avg(df,c,"iterated","converged"), avg(df,c,"iterated","zero_beta_count"),
                avg(df,c,"desk","cost_gross_vol_ratio")]
        print("| " + " | ".join([c]+vals) + " |")
    print("\n| configuration | method | converged | distance | residual variance |")
    print("|---|---|---:|---:|---:|")
    for c in cfgs:
        for m in ["iterated:damped", "iterated:ridge", "iterated:ridge_damped", "iterated:decoupled", "shapley_target", "shapley_gamma", "shapley_target:decoupled", "shapley_gamma:decoupled"]:
            optional = m in ("iterated:damped", "iterated:ridge", "iterated:ridge_damped")
            failed = df.filter((pl.col("config") == c) & (pl.col("method") == "iterated") & (pl.col("metric") == "converged") & (pl.col("value") == 0)).height > 0
            if avg(dist,c,m) != "—" and (not optional or failed):
                print("| " + " | ".join([c,m,avg(df,c,m,"converged"),avg(dist,c,m),avg(df,c,m,"var_share")]) + " |")
    print("\nTotal seed runtime seconds:", df.filter(pl.col("method") == "timing")["value"].sum())
    print("Units:", df["unit"].n_unique(), "Seeds:", df.select("unit","seed").unique().height)
    print("\nIdentical signal shares:")
    print(sh.filter((pl.col("config") == "identical") & pl.col("method").is_in(["iterated","regression","durin"])).group_by("method","signal").agg(pl.col("share").mean()).sort("method","signal"))


if __name__ == "__main__":
    main()
