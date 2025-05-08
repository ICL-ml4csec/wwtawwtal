"""Produce baseline results (i.e., AUC figures for each model)."""

import logging
import lzma
import uuid
from typing import Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

import config
import util

log = logging.getLogger(__name__)

def nbr_attack_labels(cfg, set_name, g_node_ids: np.array) -> Tuple[Set, Set]:
    """Return the attack and contaminated nid sets for `g_node_ids`."""
    log.info("fetching attack labels...")
    graph_path = f"{cfg.graphs}/{set_name}_prov_graph_collapsed_sockets.pkl.xz"
    log.info("reading %s...", graph_path)
    g = util.pickle_read(graph_path)

    label_path = f"{cfg.labels}/{set_name}_attack_node_uuids.xz"
    log.info("reading %s...", label_path)
    with lzma.open(label_path, "rt") as f:
        if cfg == config.atlasv2:
            # need to transform to uuid.UUID
            atk_uuids = {uuid.UUID(line.rstrip()) for line in f}
        else:
            atk_uuids = {line.rstrip() for line in f}

    label_path = f"{cfg.labels}/{set_name}_contaminated_node_uuids.xz"
    log.info("reading %s...", label_path)
    with lzma.open(label_path, "rt") as f:
        if cfg == config.atlasv2:
            # need to transform to uuid.UUID
            ctm_uuids = {uuid.UUID(line.rstrip()) for line in f}
        else:
            ctm_uuids = {line.rstrip() for line in f}

    atk_set = {nid for nid in g_node_ids if g.vs[nid]["uuid"] in atk_uuids}
    ctm_set = {nid for nid in g_node_ids if g.vs[nid]["uuid"] in ctm_uuids}

    ### N.B.: for consistent labeling, merge contaminated with attack
    # this results in all post-attack behaviors being attack
    # (seems to be especially important for FLASH on TRACE)
    #atk_set.update(ctm_set)
    #ctm_set = set()

    # special handling for CADETS: remove "bad neighborhoods"
    if set_name == "cadets":
        label_path = f"{cfg.labels}/{set_name}_bad_nbr_node_uuids.xz"
        log.info("reading %s...", label_path)
        with lzma.open(label_path, "rt") as f:
            if cfg == config.atlasv2:
                # need to transform to uuid.UUID
                bad_uuids = {uuid.UUID(line.rstrip()) for line in f}
            else:
                bad_uuids = {line.rstrip() for line in f}
        bad_set = {nid for nid in g_node_ids if g.vs[nid]["uuid"] in bad_uuids}
        ctm_set |= bad_set

    return atk_set, ctm_set

def roc_curve(df: pd.DataFrame):
    """Return the ROC curve for `res`."""
    # check if this is FLASH by seeing if score is absent
    if "score" not in df.columns:
        # prediction + confidence AUC: slide over confidence thresholds
        tp = []
        fp = []
        # optimization: remove anomalies, they are unaffected by threshold
        n_atk = df["attack"].sum()
        n_benign = len(df) - n_atk
        df_anomaly = df.loc[~df["prediction"]]
        anomaly_tp = df_anomaly["attack"].sum()
        anomaly_fp = len(df_anomaly) - anomaly_tp
        df = df.loc[df["prediction"]]
        thresholds = df["confidence"].unique()
        thresholds.sort()
        for th in thresholds:
            benign = df["confidence"] >= th
            tp.append((~benign & df["attack"]).sum() + anomaly_tp)
            fp.append((~benign & ~df["attack"]).sum() + anomaly_fp)
        tp = np.array(tp)
        fp = np.array(fp)

        tpr = tp / n_atk
        fpr = fp / n_benign

        # start at (0, 0), end at (1, 1)
        tpr = np.append(np.insert(tpr, 0, 0), 1)
        fpr = np.append(np.insert(fpr, 0, 0), 1)
    else:
        fpr, tpr, _ = sklearn.metrics.roc_curve(df["attack"], df["score"])

    return fpr, tpr

def plot_set(cfg, set_name: str, n_hops=1) -> None:
    path = f"{cfg.neighborhoods}/{set_name}_labels_{n_hops}hop_collapsed_sockets.pkl.xz"
    log.info("reading %s...", path)
    _, _, g_node_ids, *_ = util.pickle_read(path)

    atk_nids, ctm_nids = nbr_attack_labels(cfg, set_name, g_node_ids)

    # Adjust these paths as needed
    basepaths = [
        ("GNN", f"<PATH_TO_GNN_RESULTS>/{set_name}"),
        ("R-CAID", f"<PATH_TO_RCAID_RESULTS>/{set_name}"),
        ("FLASH", f"<PATH_TO_FLASH_RESULTS>/{set_name}"),
        ("KAIROS", f"<PATH_TO_KAIROS_RESULTS>/{set_name}"),
    ]
    fig, ax = plt.subplots()
    all_proc_nids = set(g_node_ids)
    for model, base in basepaths:
        path = f"{base}/nid_anomaly_scores.csv.xz"
        try:
            df = pd.read_csv(path)
            log.info("loaded anomaly results from %s...", path)
            # only keep rows for process nodes
            df = df.loc[df["nid"].isin(all_proc_nids)]
            # delete rows for contaminated nodes
            df = df.loc[~df["nid"].isin(ctm_nids)]
            # overwrite attack labels
            df["attack"] = df["nid"].isin(atk_nids)

            fpr, tpr = roc_curve(df)
            auc = sklearn.metrics.auc(fpr, tpr)
            ax.plot(fpr, tpr, drawstyle="steps-pre", label=f"{model}: {auc:.3f}")
            log.info("%s: %s: %s", set_name, model, auc)
        except FileNotFoundError:
            # skip this one for now
            log.info("skipping %s...", path)
            continue

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4")
    ax.legend(loc="lower right")
    ax.set_box_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    out_path = f"{set_name}_baseline_roc.pdf"
    fig.savefig(out_path)
    plt.close(fig)

def main(argv):
    for set_name in ["cadets", "theia", "fivedirections", "trace",
                     "trace_departitioned"]:
        plot_set(config.e3, set_name, n_hops=2)
    for set_name in ["h1", "h2"]:
        plot_set(config.atlasv2, set_name, n_hops=2)
    for set_name in ["host201", "host501"]:
        plot_set(config.optc, set_name, n_hops=2)
