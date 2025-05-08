""""Autoencoder" using autocorrelation results and the training data. This
essentially repeats the training data to recreate periods.

Many of the final evaluation results are also here.

Strategy:
- Find the actual data (some neighborhoods start off later than others)
- Filter out any artifacts (e.g., startup behavior)
- Repeat these bins to reconstruct the dataset
"""

import csv
import logging
import lzma
import uuid
from typing import Dict, List, Optional, Set, Tuple

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.linear_model
import tqdm

import config
import gnn.baseline
import gnn.util.benign_day_edges
import neighborhood.autocorrelation
import util

log = logging.getLogger(__name__)

def normalize_rows(y: np.array) -> np.array:
    """Normalize the bins of `y` such that each label's line spans 0 to 1."""
    # Strategy 1: sum of line is 1 (or 0, if the row was already 0)
    s = y.sum(axis=0, keepdims=True)
    # easier to just divide everything and fix the NaNs later
    with np.errstate(invalid="ignore"):
        y /= s
        # replace NaN with 0: sum was 0, so row was all 0
        np.nan_to_num(y, copy=False, nan=0)
    return y

    # Strategy 2: max per line is 1
    #m = y.max(axis=0, keepdims=True)
    #return y / m

def adjust_acfs(acfs: np.array) -> np.array:
    """Return adjusted `acfs`: replace None with 0 (meaning if we don't have
    data, assume no correlation) and take the absolute value.

    On absolute value: negative autocorrelation technically means a low value
    indicates the next value is likely to be high, and vice versa. This seems
    likely in two different places: 1. we accidentally found the half-period,
    but we tuned our objective function to avoid this, or 2. the data itself
    is just not very periodic. If the latter is true, the acf will be a small
    negative, so we just turn it into a small positive."""
    out = acfs.copy()
    out[out == None] = 0
    return np.abs(out)

def reconstruction_loss(y_test: np.array, z_test: np.array) -> np.array:
    """Calculate the reconstruction loss of the forecasted neighborhoods
    `z_test` versus the true neighborhoods `y_test`.
    Return the loss per neighborhood."""
    # old model: NMSE. Suffers from overly rewarding 0s
    #mses = ((y_test - z_test) ** 2).sum(axis=0) / y_test.size
    #nmses = mses / y_test.var()
    #return nmses

    # new model: essentially false negative rate
    with np.errstate(invalid="ignore"):
        # how many missed processes? Note that this allows "overguessing"
        # e.g., if the true frequency is 1 and model predicts 2,
        # model gets full credit for that particular bin
        # Hence analogy to false negative rate
        losses = np.clip(y_test - z_test, 0, None)
        # ... out of how many total expected processes?
        # easier to just divide everything and fix the NaNs later
        losses = losses.sum(axis=0) / y_test.sum(axis=0)
        # replace NaN with 0 (perfect reconstruction)
        np.nan_to_num(losses, copy=False, nan=0)

    return losses

def rhoencoder(t_train: np.array, y_train: np.array,
               t_test: np.array, y_test: np.array,
               set_name: str,
               width: float = 900) -> Tuple[np.array, np.array]:
    """Generate `z_test`, the predicted test neighborhoods (to be compared
    with `y_test`) based on the training data. Also return the best
    autocorrelation coefficient per neighborhood."""
    # trim off startup behavior: remove first bin
    y_train = y_train[1:, :]
    t_train = t_train[1:]
    n_bins, n_labels = y_train.shape

    # trim off zeros from start
    # autocorrelation should start from the first instance of a neighborhood
    shift = []
    for l in range(n_labels):
        y = y_train[:, l]
        nz = np.flatnonzero(y > 0)
        if nz.size == 0:
            # no train neighborhoods
            shift.append(None)
        else:
            i = nz.min()
            # require at least N bins, not exactly sure what N is appropriate
            # arbitrarily choose 30 bins = 7 hours for 900 second bins
            if n_bins - i < 30:
                shift.append(None)
            else:
                shift.append(i)

    # N.B.: unlike analysis.py, period here is in bins, not time
    periods = []
    acfs = []
    for l in range(n_labels):
        if shift[l] is None:
            periods.append(None)
            acfs.append(None)
            continue
        # abuse of API here: as long as len(t_train) as correct, the actual
        # values don't matter. The "correct" input would be
        #   t_train[0:len(t_train) - shift[l]]
        y = y_train[shift[l]:, l]
        t = t_train[shift[l]:]
        if len(y) <= 1:
            periods.append(None)
            acfs.append(None)
        else:
            acf, _, _ = neighborhood.analysis.autocorrelate_label(t, y)

            # use first crossover acf
            crossover = (acf <= 0).argmax() # gets first instance of True
            periods.append(acf[crossover:].argmax() + crossover)
    acfs = np.array(acfs)

    z_test = np.zeros_like(y_test) # generated output
    # working backwards, grab as many complete periods as possible
    # (not using vectorized code for ease of implementation)
    for l in range(n_labels):
        y = y_train[shift[l]:, l]
        p = periods[l]
        if p is None:
            continue # not periodic, just leave as zeros
        start = y.size % p
        # copy the training bins into the output
        for i in range(z_test.shape[0]):
            z_test[i, l] = y[start + (i % p)]

    return z_test, acfs

def plot_rhoencoder(t_test: np.array, y_test: np.array, z_test: np.array,
                    set_name: str, labels: np.array,
                    width: float = 900):
    """Plot the results of the rhoencoder, where y_test is the ground truth
    test neighborhoods and z_test in the generated neighborhoods."""
    fig, (axh, axl) = plt.subplots(2, 1,
        gridspec_kw={"hspace": 0}, layout="constrained",
        sharex=True, sharey=True, figsize=(8, 6)
    )

    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad(cmap(0))

    x_mesh = np.append(t_test, t_test[-1] + width)
    y_mesh = np.arange(y_test.shape[1] + 1)

    vmin = min(y_test[y_test > 0].min(), z_test[z_test > 0].min())
    vmax = max(y_test.max(), z_test.max())
    im = axl.pcolormesh(x_mesh, y_mesh, y_test.transpose(),
        cmap=cmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        rasterized=True # weird visual artifacts when not rasterized
    )
    axh.pcolormesh(x_mesh, y_mesh, z_test.transpose(),
        cmap=cmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        rasterized=True
    )

    step = 86400 # 60 * 60 * 24 = one day, in seconds
    def day_formatter(x: float, pos) -> str:
        return f"{x / step:.0f}"
    axl.set_xticks(np.arange(t_test.min(), t_test.max(), step))
    axl.xaxis.set_major_formatter(day_formatter)
    axl.set_xlim(x_mesh.min(), x_mesh.max())
    axl.set_ylim(y_mesh.min(), y_mesh.max())
    axl.margins(x=0)
    axl.set_xlabel("Time (days)")
    axl.set_ylabel("True Neighborhoods")
    axh.set_ylabel("Predicted Neighborhoods")

    out_path = f"{set_name}_rhoencoded.pdf"
    log.info("writing %s...", out_path)
    fig.colorbar(im, ax=[axl, axh])
    fig.savefig(out_path)
    plt.close(fig)

    losses = reconstruction_loss(y_test, z_test)

    # use same coloring as neighborhoods. This ends up being ordered by first
    # appearance and not neighborhood, and would probably be nice to change,
    # but oh well
    neighborhoodset = {l: None for l in labels}
    n_neighborhoods = len(neighborhoodset)
    colors = {l: cc.glasbey_dark[i % len(cc.glasbey_dark)]
              for i, l in zip(range(n_neighborhoods),
                              neighborhoodset)}
    # dummy y value for labels
    y = np.arange(0, n_neighborhoods)
    fig, ax = plt.subplots()
    for l, loss in enumerate(losses):
        c = colors[l]
        ax.plot(loss, y[l], linestyle="", marker="o", markersize=2,
                markerfacecolor=c, markeredgecolor=c)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Reconstruction Loss")
    ax.set_ylim(y.min(), y.max())
    ax.set_ylabel("Provenance Neighborhood")

    out_path = f"{set_name}_loss.pdf"
    log.info("writing %s...", out_path)
    plt.savefig(out_path)
    plt.close(fig)

def calculate_auc(df: pd.DataFrame,
                  attack_nids: Optional[Set[int]] = None,
                  benign_nids: Optional[Set[int]] = None):
    """Calculate the AUC of the result dataframe.
    Optionally, if either parameter is not None, only include a subset of
    attack nodes or benign nodes."""
    # filter out the attacks/benigns first
    if benign_nids is not None:
        df = df.loc[df["attack"]
                    | (~df["attack"] & df["nid"].isin(benign_nids))]
    if attack_nids is not None:
        df = df.loc[~df["attack"]
                    | (df["attack"] & df["nid"].isin(attack_nids))]

    # make sure there's at least one benign and attack sample,
    # otherwise AUC is not defined
    n_atk = df["attack"].sum()
    if n_atk == 0 or n_atk == len(df):
        return np.nan

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

        auc = sklearn.metrics.auc(fpr, tpr)
    else:
        auc = sklearn.metrics.roc_auc_score(df["attack"], df["score"])

    return auc

def auc_loss_eab(out_name: str, anomaly_res: Dict[str, pd.DataFrame],
                 labels: np.array, g_node_ids: np.array,
                 y_test: np.array, z_test: np.array):
    """Compute AUC vs maximum neighborhood loss, but using equal area bins
    (eab) instead of equal width bins.

    anomaly_res should be a dict of {name: DataFrame} where `name` is the
    name of the model and the dataframe contains at least these three columns:
        nid: nid of the node
        score: anomaly score for the node
        attack: True iff node is an attack node

    An exception is FLASH, which should instead have these columns:
        nid: nid of the node
        attack: True iff node is an attack node
        prediction: whether the model predicts this node is an attack node
        confidence: confidence of model's prediction
    """
    losses = reconstruction_loss(y_test, z_test)
    # we are being lazy and using the ground truth in one result here, rather
    # than fetching it ourselves. all models are using the same labeling, so
    # this should be fine
    # NOTE: the GNN and RCAID did some stuff with contaminated nodes and are
    # missing a few nodes as a result. Rather than retrain them (which is
    # expensive), use KAIROS or FLASH which were retrained to not prune off
    # any contaminated results.
    res0 = next(reversed(anomaly_res.values()))

    # prevent multiple windows from containing the same loss by grouping
    # neighborhoods with equal losses
    order = losses.argsort()
    groups = []
    prev_loss = np.nan
    for i in order:
        loss = losses[i]
        if np.isclose(loss, prev_loss):
            groups[-1].append(i)
        else:
            groups.append([i])
            prev_loss = loss

    # slide over windows of increasing loss, forming windows of equal area
    # l, r are left, right edges of window
    area = 0.2
    l = 0
    r = 1
    benign_nids = set(g_node_ids)
    x = []
    aucs = {}
    ns = []

    # precompute the number of benign samples per group
    n_benign = []
    for group in groups:
        nids = set()
        for label in group:
            nids.update(g_node_ids[labels == label])
        n = (~res0.loc[res0["nid"].isin(nids), "attack"]).sum()
        n_benign.append(n)
    n_benign = np.array(n_benign)
    n_total = n_benign.sum()
    # redefine area to be based on n_total (round up to avoid tiny end bin)
    area = int(np.ceil(area * n_total))

    log.info("doing equal area AUC vs loss analysis...")
    pbar = tqdm.tqdm(total=n_benign.size)

    # precompute windows ahead of time so that we can adjust them later
    windows = []
    while l < n_benign.size:
        n = n_benign[l]
        while r < n_benign.size and n + n_benign[r] < area:
            n += n_benign[r]
            r += 1

        # add edge point if the resulting area is closer to the fixed area
        if r < n_benign.size:
            added = n + n_benign[r]
            if added - area < area - n:
                n = added
                r += 1

        # record window
        # use left side loss, not right side
        loss = losses[groups[l][0]]
        windows.append((l, r))
        ns.append(n)
        x.append(loss)

        # slide window
        l = r
        r += 1

    def merge_small(x: List[float], ns: List[int],
                    windows: List[Tuple[int, int]]) -> Tuple[List, List, List]:
        """Adjust small windows by merging them into the smallest neighbor."""
        # (define small as 10% of the expected area --- currently, 2%)
        threshold = int(0.1 * area)
        def aux(x_old, ns_old, windows_old):
            x = []
            ns = []
            windows = []

            i = 0
            while i < len(ns_old):
                n = ns_old[i]
                if n < threshold:
                    if i == 0: # leftmost, always merge right
                        merge = i + 1
                    elif i == len(ns_old) - 1: # rightmost, always merge left
                        merge = i - 1
                    else:
                        n_left = ns_old[i - 1]
                        n_right = ns_old[i + 1]
                        if n_left < n_right:
                            merge = i - 1
                        else:
                            merge = i + 1

                    # actually perform the merge now
                    if merge == i - 1:
                        # modify latest entry of output
                        # no need to modify x, already correct
                        ns[-1] += n
                        windows[-1] = (windows[-1][0], windows_old[i][1])
                    else:
                        # add new merged entry to output
                        x.append(x_old[i])
                        ns.append(n + ns_old[merge])
                        windows.append((windows_old[i][0], windows_old[merge][1]))
                        # skip the next entry due to the merge
                        i += 1
                else:
                    # copy current entry directly
                    x.append(x_old[i])
                    ns.append(ns_old[i])
                    windows.append(windows_old[i])
                i += 1
            return x, ns, windows

        while True:
            n_windows = len(windows)
            x, ns, windows = aux(x, ns, windows)
            if n_windows == len(windows):
                break # reached the fix point, can stop now

        return x, ns, windows

    x, ns, windows = merge_small(x, ns, windows)

    # prepend all node AUC (use sentinel value -1)
    x.insert(0, -1) # these lists already formed, must prepend
    ns.insert(0, n_total)
    pbar.set_description(f"[loss = -1, n = {n_total}]")
    for model, res in anomaly_res.items():
        auc = calculate_auc(res, benign_nids=set(g_node_ids))
        aucs[model] = [auc]
        log.info("%s: auc0 = %s", model, auc)

    # insert 0 loss only as a special value
    x.insert(1, 0)
    ns.insert(1, n_benign[0])
    if losses[groups[0][0]] == 0:
        benign_nids = set()
        for label in groups[0]:
            benign_nids.update(g_node_ids[labels == label])

        for model, res in anomaly_res.items():
            auc = calculate_auc(res, benign_nids=benign_nids)
            aucs[model].append(auc)
    else:
        for model in anomaly_res.keys():
            aucs[model].append(np.nan)

    # now compute the AUC of our windows
    for i, (l, r) in enumerate(windows):
        # collect benign nids
        benign_nids = set()
        for group in groups[l:r]:
            for label in group:
                benign_nids.update(g_node_ids[labels == label])

        # calculate AUC for this benign subset
        pbar.update(r - l)
        pbar.set_description(f"[loss = {x[i]:.3e}, n = {ns[i]}]")
        for model, res in anomaly_res.items():
            auc = calculate_auc(res, benign_nids=benign_nids)
            aucs[model].append(auc)
    pbar.close()

    x = np.array(x)

    with open(f"{out_name}_auc_loss_eab.csv", "wt") as f:
        fieldnames = ("x", *aucs.keys(), "n_samples")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, x in enumerate(x):
            row = {k: v[i] for k, v in aucs.items()}
            row["x"] = x
            row["n_samples"] = ns[i]
            writer.writerow(row)

def covert_auc(out_name: str, anomaly_res: Dict[str, pd.DataFrame],
               labels: np.array, g_node_ids: np.array):
    """Compute AUC, but only including neighborhoods between certain
    covertness thresholds.

    anomaly_res should be a dict of {name: DataFrame} where `name` is the
    name of the model and the dataframe contains at least these three columns:
        nid: nid of the node
        score: anomaly score for the node
        attack: True iff node is an attack node

    An exception is FLASH, which should instead have these columns:
        nid: nid of the node
        attack: True iff node is an attack node
        prediction: whether the model predicts this node is an attack node
        confidence: confidence of model's prediction
    """
    # calculate overlap; covertness is 1 - overlap
    covertness = []
    atk_labels = [] # map covertness value -> label
    # we are being lazy and using the ground truth in one result here, rather
    # than fetching it ourselves. all models are using the same labeling, so
    # this should be fine
    # NOTE: the GNN and RCAID did some stuff with contaminated nodes and are
    # missing a few nodes as a result. Rather than retrain them (which is
    # expensive), use KAIROS or FLASH which were retrained to not prune off
    # any contaminated results.
    res0 = next(reversed(anomaly_res.values()))
    for label in set(labels):
        nids = g_node_ids[labels == label]
        df_atk = res0.loc[res0["nid"].isin(nids) & res0["attack"]]
        # ignore non-attack neighborhoods
        if len(df_atk) > 0:
            covertness.append(1 - (len(df_atk) / len(nids)))
            atk_labels.append(label)
    covertness = np.array(covertness)
    atk_labels = np.array(atk_labels)

    order = covertness.argsort()
    # slide over windows of increasing covertness
    # (therefore, start with all attacks, then slowly _remove_ "obvious"
    #  attacks)
    # l, r are left, right edges of window
    atk_nids = set(res0.loc[res0["attack"]]["nid"])
    l = 0
    r = 1
    x_covertness = []
    aucs = {}
    n_atks = []

    log.info("doing attack covertness analysis...")
    pbar = tqdm.tqdm(total=order.size)

    # first, add 0 covertness point (all node AUC)
    x_covertness.append(0)
    n_atks.append(res0["nid"].isin(atk_nids).sum())
    pbar.set_description(f"[threshold = 0, n = {n_atks[0]}]")
    for model, res in anomaly_res.items():
        auc = calculate_auc(res, attack_nids=atk_nids)
        aucs[model] = [auc]
        log.info("%s: auc0 = %s", model, auc)

    while l < order.size:
        # scan until the covertness changes
        cvt = covertness[order[l]]
        # never remove all attack (otherwise, cannot find AUC)
        while r < order.size - 1 and np.isclose(cvt, covertness[order[r]]):
            r += 1

        # remove nids in current window
        for label in atk_labels[order[l:r]]:
            nids = g_node_ids[labels == label]
            df = res0.loc[res0["nid"].isin(nids) & res0["attack"]]
            atk_nids.difference_update(df["nid"])

        # slide window
        pbar.update(r - l)
        l = r
        r += 1

        # calculate AUC for this attack subset
        x_covertness.append(cvt)
        n_atks.append(res0["nid"].isin(atk_nids).sum())
        pbar.set_description(f"[threshold = {cvt:.3f}, n = {n_atks[-1]}]")
        for model, res in anomaly_res.items():
            auc = calculate_auc(res, attack_nids=atk_nids)
            aucs[model].append(auc)
    pbar.close()

    fig, ax = plt.subplots()

    for model, y in aucs.items():
        ax.plot(x_covertness, y, label=model)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold Attack Covertness")
    ax.set_ylabel("AUROC")

    ax2 = ax.twinx()
    ax2.plot(x_covertness, n_atks, color="k", linestyle=":")
    ax2.set_ylabel("Number of Attack Samples")
    ax2.set_ylim(0, max(n_atks))

    fig.legend(bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

    out_path = f"{out_name}_covert_auc.pdf"
    log.info("writing %s...", out_path)
    fig.savefig(out_path)
    plt.close(fig)

def covert_binary(out_name: str, anomaly_res: Dict[str, pd.DataFrame],
                  labels: np.array, g_node_ids: np.array,
                  y_train: np.array) -> dict:
    """Compute AUC vs binary covertness, i.e., either a neighborhood is seen
    during training or not seen during training. Return a dict of dicts:
    { "unseen": {model: auc}, "seen": {model: auc} } """
    unseen_nids = []
    seen_nids = []
    # we are being lazy and using the ground truth in one result here, rather
    # than fetching it ourselves. all models are using the same labeling, so
    # this should be fine
    # NOTE: the GNN and RCAID did some stuff with contaminated nodes and are
    # missing a few nodes as a result. Rather than retrain them (which is
    # expensive), use KAIROS or FLASH which were retrained to not prune off
    # any contaminated results.
    res0 = next(reversed(anomaly_res.values()))
    for label in set(labels):
        nids = g_node_ids[labels == label]
        df_atk = res0.loc[res0["nid"].isin(nids) & res0["attack"]]
        # ignore non-attack neighborhoods
        if len(df_atk) > 0:
            if y_train[:, label].sum() > 0:
                seen_nids.extend(df_atk["nid"])
            else:
                unseen_nids.extend(df_atk["nid"])
            #if len(df_atk) == len(nids):
            #    unseen_nids.extend(nids)
            #else:
            #    seen_nids.extend(nids)
    unseen_nids = np.array(unseen_nids)
    seen_nids = np.array(seen_nids)

    aucs = {"unseen": {}, "seen": {}}
    for model, res in anomaly_res.items():
        aucs["seen"][model] = calculate_auc(res, attack_nids=seen_nids)
        aucs["seen"]["n_samples"] = len(seen_nids)
        aucs["unseen"][model] = calculate_auc(res, attack_nids=unseen_nids)
        aucs["unseen"]["n_samples"] = len(unseen_nids)

    with open(f"{out_name}_covert_train.csv", "wt") as f:
        fieldnames = ("atk_type", *anomaly_res.keys(), "n_samples")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k, row in aucs.items():
            row["atk_type"] = k
            writer.writerow(row)

def write_loss_stats(set_name: str, y_train: np.array,
                     y_test: np.array, z_test: np.array,
                     labels: np.array) -> None:
    losses = reconstruction_loss(y_test, z_test)

    with open(f"{set_name}_loss_stats.csv", "wt") as f:
        fieldnames = ("dataset", "nbr_id", "loss",
                      "n_samples", "test_only")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, loss in enumerate(losses):
            n_samples = (labels == label).sum()
            writer.writerow({
                "dataset": set_name,
                "nbr_id": label,
                "loss": loss,
                "n_samples": n_samples,
                "test_only": y_train[:, label].sum() == 0
            })

def write_label_stats(set_name: str, anomaly_res: Dict[str, pd.DataFrame]):
    # use ground truth in KAIROS, since it doesn't filter anything out
    res0 = next(reversed(anomaly_res.values()))
    with open(f"{set_name}_label_stats.csv", "wt") as f:
        fieldnames = ("dataset", "n_procs", "n_atk")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "dataset": set_name,
            "n_procs": len(res0),
            "n_atk": res0["attack"].sum(),
        })

def contaminate_auc(set_name: str, anomaly_res: Dict[str, pd.DataFrame],
                    ctm_nids: Set[int]):
    """Calculate AUC when including or removing contaminated nodes."""
    models = [*anomaly_res.keys()]
    # use ground truth in KAIROS, since it doesn't filter anything out
    res0 = next(reversed(anomaly_res.values()))

    non_ctm_nids = set(res0.loc[~res0["nid"].isin(ctm_nids)]["nid"])

    with open(f"{set_name}_contaminated_auc.csv", "wt") as f:
        fieldnames = ("dataset", "contaminates", "non_contaminates",
                      *models, "n_benign", "n_atk")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for contaminates in (True, False):
            for non_contaminates in (True, False):
                nids = set(res0["nid"])
                if not contaminates:
                    nids -= ctm_nids
                if not non_contaminates:
                    nids -= non_ctm_nids
                attack_nids = set(res0.loc[res0["attack"] & res0["nid"].isin(nids)]["nid"])
                benign_nids = set(res0.loc[~res0["attack"] & res0["nid"].isin(nids)]["nid"])
                row = {"dataset": set_name,
                       "contaminates": contaminates,
                       "non_contaminates": non_contaminates,
                       "n_benign": len(benign_nids),
                       "n_atk": len(attack_nids)
                      }
                for model, res in anomaly_res.items():
                    row[model] = calculate_auc(res,
                                               attack_nids=attack_nids,
                                               benign_nids=benign_nids)
                writer.writerow(row)

def encode_set(cfg, set_name: str, n_hops: int = 1) -> None:
    path = f"{cfg.neighborhoods}/{set_name}_labels_{n_hops}hop_collapsed_sockets.pkl.xz"
    log.info("reading %s...", path)
    labels, times, g_node_ids, *_ = util.pickle_read(path)

    bin_width = 900
    y, _ = neighborhood.autocorrelation.fixed_width_vectors(
        times, labels, width=bin_width
    )
    t = neighborhood.autocorrelation.make_binned_t(times, y, width=bin_width)
    t -= t.min()

    if cfg == config.e3:
        t_end = gnn.util.benign_day_edges.e3_time_train_end(set_name)
    elif cfg == config.atlasv2:
        t_end = gnn.util.benign_day_edges.atlasv2_time_train_end()
    elif cfg == config.optc:
        t_end = gnn.util.benign_day_edges.optc_time_train_end(set_name)
    t_end /= 1e9

    # train:test split based on times
    # awkward: should we split exactly based on times? this will fragment the
    # bins! Otherwise, we can split based on the bins, but this will not
    # exactly match the times. Latter seems better; be sure to floor the bin
    train_delta = t_end - times.min()
    train_bins = int(train_delta / bin_width)

    y_train = y[:train_bins, :]
    t_train = t[:train_bins]
    y_test = y[train_bins:, :]
    t_test = t[train_bins:]

    z_test, acfs = rhoencoder(t_train, y_train, t_test, y_test, set_name)

    acfs = adjust_acfs(acfs)
    plot_rhoencoder(t_test, y_test, z_test, set_name, labels)

    write_loss_stats(set_name, y_train, y_test, z_test, labels)

    # the labelings output by FLASH/KAIROS seem slightly inconsistent
    # relative to the GNN labels (probably generated at different times),
    # and we need the flexibility to update them on the fly,
    # so insert our own labels instead
    atk_nids, ctm_nids = gnn.baseline.nbr_attack_labels(cfg, set_name,
                                                        g_node_ids)
    # Adjust these paths as needed
    basepaths = [
        ("GNN", f"<PATH_TO_GNN_RESULTS>/{set_name}"),
        ("R-CAID", f"<PATH_TO_RCAID_RESULTS>/{set_name}"),
        ("FLASH", f"<PATH_TO_FLASH_RESULTS>/{set_name}"),
        ("KAIROS", f"<PATH_TO_KAIROS_RESULTS>/{set_name}"),
    ]
    anomaly_res = {}
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
            anomaly_res[model] = df
        except FileNotFoundError:
            # skip this one for now
            log.info("skipping %s...", path)
            continue

    #contaminate_auc(set_name, anomaly_res, ctm_nids)

    del all_proc_nids, ctm_nids, atk_nids

    write_label_stats(set_name, anomaly_res)
    auc_loss_eab(set_name, anomaly_res, labels, g_node_ids, y_test, z_test)
    covert_binary(set_name, anomaly_res, labels, g_node_ids, y_train)

def main(argv):
    for set_name in ["cadets", "theia", "fivedirections", "trace",
                     "trace_departitioned"]:
        encode_set(config.e3, set_name, n_hops=2)
    for set_name in ["h1", "h2"]:
        encode_set(config.atlasv2, set_name, n_hops=2)
    for set_name in ["host201", "host501"]:
        encode_set(config.optc, set_name, n_hops=2)
