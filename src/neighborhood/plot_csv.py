"""Plot any precalculated results in the csv directory."""

import csv
import logging
from typing import List

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

import config

log = logging.getLogger(__name__)

def plot_auc_loss_eab(set_name: str, df: pd.DataFrame):
    # drop row 0 (entire dataset)
    full_set = df.iloc[0]
    df.drop(0, inplace=True)

    # drop row 1 (0-only)
    zero_only = df.iloc[1]
    df.drop(1, inplace=True)

    # convert n_samples to %
    df["n_samples"] /= (full_set["n_samples"] / 100)

    ## filter out low n_sample outliers (below 2%)
    #df.drop(df.index[df["n_samples"] < 2], inplace=True)

    models = [col for col in df.columns if col not in {"x", "n_samples"}]

    fig, ax = plt.subplots()
    lines = []
    for model in models:
        l = ax.plot(df["x"], df[model], label=model,
                    marker="o", fillstyle="none")
        lines.extend(l)

    ax2 = ax.twinx()
    l = ax2.plot(df["x"], df["n_samples"], label="% of Benign",
                 linestyle="--", color="gray", marker="o", fillstyle="none")
    lines.extend(l)

    #fig.legend(bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    #fig.legend(bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    ax.legend(handles=lines)
    #xmin = df.loc[df["x"] > 0, "x"].min()
    #xmin = 10 ** np.floor(np.log10(xmin))
    #ax.set_xlim(df["x"].min(), df["x"].max())
    #ax.set_xscale("symlog", linthresh=xmin)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Reconstruction Loss")

    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel(f"% of Benign Samples\n(Total = {int(full_set['n_samples'])})")

    out_path = f"{set_name}_auc_loss_eab.pdf"
    fig.savefig(out_path)
    plt.close(fig)

    # return leftmost minus rightmost
    out = {model: df[model].iat[0] - df[model].iat[-1]
           for model in models}
    out["dataset"] = set_name
    return out

def plot_loss_stats(sets: List[str], df: pd.DataFrame):
    # CDF of loss vs both neighborhoods and # samples
    df.sort_values(by="loss")

    stats = pd.DataFrame(df.groupby("loss")["loss"].count().rename("n_nbr"))
    stats["n_proc"] = df.groupby("loss")["n_samples"].sum()
    stats.reset_index(inplace=True)

    stats["pdf_nbr"] = stats["n_nbr"] / stats["n_nbr"].sum()
    stats["pdf_proc"] = stats["n_proc"] / stats["n_proc"].sum()
    stats["cdf_nbr"] = stats["pdf_nbr"].cumsum()
    stats["cdf_proc"] = stats["pdf_proc"].cumsum()

    fig, ax = plt.subplots()
    #ax.plot(stats["loss"], stats["cdf_nbr"], label="By Neighborhood")
    ax.plot(stats["loss"], stats["cdf_proc"], label="By Process")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Reconstruction Loss")
    ax.set_ylim(0, 1)
    ax.set_ylabel("CDF")

    #ax.legend()

    out_path = "loss_cdf.pdf"
    fig.savefig(out_path)
    plt.close(fig)

    stats = pd.DataFrame(df.groupby("dataset").apply(
        lambda df: np.average(df["loss"], weights=df["n_samples"])
    ).rename("weighted_loss"))
    stats["unweighted_loss"] = df.groupby("dataset")["loss"].mean()
    stats.reset_index(inplace=True)

    print("Weighted vs Unweighted Loss")
    print(stats)

    out_path = f"csv/dataset_loss.csv"
    stats.to_csv(out_path)

def loss_quantiles(sets: List[str], df: pd.DataFrame):
    df.sort_values(by="loss")

    stats = pd.DataFrame(df.groupby("loss")["loss"].count().rename("n_nbr"))
    stats["n_proc"] = df.groupby("loss")["n_samples"].sum()
    stats.reset_index(inplace=True)

    stats["csum"] = stats["n_proc"].cumsum() / stats["n_proc"].sum()

    stats.to_csv("csum.csv")

def plot_covert_binary(sets: List[str], df: pd.DataFrame):
    x = np.arange(len(sets))
    models = [col for col in df.columns
              if col not in {"dataset", "atk_type", "n_samples"}]
    # need to fit (n_models + 1) * 2 bars per x; also leave some padding
    width = 1. / (2 * (len(models) + 1) + 1)
    fig, ax = plt.subplots(layout="constrained")
    ax2 = ax.twinx()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {model: color_cycle[i] for i, model in enumerate(models)}
    colors["n_samples"] = "gray"

    for i, model in enumerate(models):
        color = colors[model]
        # plot unseen first, then seen second
        y = df[df["atk_type"] == "unseen"][model]
        rects = ax.bar(x + 2 * i * width, y, width=width,
                       color=color, hatch="xxxx")
        #ax.bar_label(rects)

        y = df[df["atk_type"] == "seen"][model]
        rects = ax.bar(x + (2 * i + 1) * width, y, width=width,
                       color=color, hatch="....")
        #ax.bar_label(rects)

    i += 1
    color = colors["n_samples"]
    y = df[df["atk_type"] == "unseen"]["n_samples"]
    rects = ax2.bar(x + 2 * i * width, y, width=width,
                    color=color, hatch="xxxx")
    #ax2.bar_label(rects)

    y = df[df["atk_type"] == "seen"]["n_samples"]
    rects = ax2.bar(x + (2 * i + 1) * width, y, width=width,
                    color=color, hatch="....")
    #ax2.bar_label(rects)


    # manual legend because we want to separate colors and hatch
    handles = [
        mpl.patches.Rectangle((0, 0), 1, 1, color=colors[model], label=model)
        for model in models
    ]
    handles.append(mpl.patches.Rectangle(
        (0, 0), 1, 1, color=colors["n_samples"], label="% of Samples"
    ))
    handles.append(mpl.patches.Rectangle(
        (0, 0), 1, 1, color="none", hatch="xxxx", label="Unseen During Training"
    ))
    handles.append(mpl.patches.Rectangle(
        (0, 0), 1, 1, color="none", hatch="....", label="Seen During Training"
    ))

    fig.legend(bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes,
               handles=handles)
    ax.set_xticks(x + len(models) * width)
    ax.set_xticklabels(sets, rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax2.set_ylabel("% of Attack Samples")
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    out_path = "covert_binary.pdf"
    fig.savefig(out_path)
    plt.close(fig)

    unseen = df.drop(df.index[df["atk_type"] != "unseen"]).set_index("dataset")
    seen = df.drop(df.index[df["atk_type"] != "seen"]).set_index("dataset")

    stats = unseen.drop(columns="atk_type") - seen.drop(columns="atk_type")
    stats["conspicuous"] = unseen["n_samples"]
    stats["covert"] = seen["n_samples"]
    stats["%_covert"] = stats["covert"] / (stats["covert"] + stats["conspicuous"])
    #stats["%_covert"] = seen["n_samples"] * 100
    stats.loc["mean"] = stats.mean()
    print("Covert difference")
    print(stats)

def fmt_df(df: pd.DataFrame):
    """Small-scale formatting, like renaming columns, etc."""
    df.rename(columns={"RCAID": "R-CAID", "Vanilla GNN": "GNN"}, inplace=True)

def plot_individual(set_names: List[str]):
    diffs = []
    for set_name in set_names:
        path = f"csv/{set_name}_auc_loss_eab.csv"
        df = pd.read_csv(path)
        fmt_df(df)
        diffs.append(plot_auc_loss_eab(set_name, df))

    df = pd.DataFrame.from_records(diffs)
    df.to_csv(f"csv/auc_loss_diffs.csv")
    df.loc["mean"] = df.mean()
    print("AUC vs loss")
    print(df)

def plot_aggregate(set_names: List[str]):
    # collect all datasets into one dataframe
    dfs = []
    for set_name in tqdm.tqdm(set_names):
        path = f"csv/{set_name}_loss_stats.csv"
        df = pd.read_csv(path)
        fmt_df(df)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(f"csv/loss_stats.csv")
    plot_loss_stats(set_names, df)

    dfs = []
    for set_name in tqdm.tqdm(set_names):
        path = f"csv/{set_name}_covert_train.csv"
        df = pd.read_csv(path)
        fmt_df(df)
        # add set name field
        df["dataset"] = set_name
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(f"csv/covert_train.csv")
    plot_covert_binary(set_names, df)

def main(argv):
    pd.options.display.float_format = "{:.4f}".format
    sets = ["cadets", "theia", "fivedirections", "trace",
            "trace_departitioned", "h1", "h2", "host201", "host501"]
    plot_aggregate(sets)
    plot_individual(sets)
