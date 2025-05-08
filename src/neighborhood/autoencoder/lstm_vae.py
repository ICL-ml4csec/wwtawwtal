"""LSTM autoencoder intended for neighborhood signals.

Since the data is a sequence, we use a bi-LSTM as a sequence-to-sequence
model. We are primarily trying to capture periodicity, which should be
well-suited to RNNs.

In practice, we found that this fixates on the horizontal stripes of system
activity, which makes sense as that dominates the dataset. However, we were
hoping to also see the vertical spikes of periodic activity, but those are
largely ignored, presumably because matching 0 against the mostly-0s in those
neighborhoods already gives a very small loss.
"""

import glob
import logging
import math
import pathlib
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
import torch
from torch import nn

import config
import neighborhood.autocorrelation
import gnn.util.benign_day_edges
from .neighborhood_dataset import NbrDataset
import util

log = logging.getLogger(__name__)

class LSTMVAE(nn.Module):
    seed = 42
    lstm_params = {
        "num_layers": 1,
        "bidirectional": True,
        "batch_first": True,
    }

    def __init__(
        self,
        input_size: int, # number of features (# neighborhoods)
        hidden_size: int = 32, # LSTM hidden size; must be <= input_size / 2
        latent_size: int = 16, # dimensions of Gaussian latent space
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # check hidden size
        # ...actually, window size is probably more important?
        #if hidden_size > int(input_size / 2):
        #    log.warn("hidden size [%s] is too large vs input size [%s]",
        #             hidden_size, input_size)

        torch.manual_seed(self.seed)

        # frequency data => no need for embedding?
        # (provenance neighborhoods already are embeddings)

        # for non-VAE, only need linear layers if bidirectional
        # to compress 2 * H_out into H_out
        # VAE always uses linear layer to create distributions anyways
        num_directions = 2 if self.lstm_params["bidirectional"] else 1
        num_layers = self.lstm_params["num_layers"]
        # input: [batch_size, seq_len, input_size]
        self.enc_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            **self.lstm_params,
        )
        # output: [batch_size, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_size]

        enc_size = hidden_size * num_directions * num_layers
        # permute hidden to: [batch_size, enc_size]
        self.enc_mean = nn.Linear(
            in_features=enc_size,
            out_features=latent_size,
        )
        self.enc_logvar = nn.Linear(
            in_features=enc_size,
            out_features=latent_size,
        )
        # output: [batch_size, latent_size]

        # decoder cannot be bidirectional
        dec_lstm_params = self.lstm_params.copy()
        dec_lstm_params["bidirectional"] = False
        dec_size = hidden_size * num_layers
        # sample some z: [batch_size, latent_size]
        self.dec_input = nn.Linear(
            in_features=latent_size,
            out_features=dec_size,
        )
        # output q(z): [batch_size, dec_size]
        # permute q(z) to decoder hidden state:
        #   [num_layers, batch_size, hidden_size]
        #
        # TODO: what is a reasonable input here for generation?
        # Most common approach seems to feed decoder output back into itself
        # During training, we can just recycle the input...
        self.dec_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            **dec_lstm_params,
        )
        # output: [batch_size, seq_len, hidden_size]
        self.dec_output = nn.Linear(
            in_features=hidden_size,
            out_features=input_size,
        )
        # output: [batch_size, seq_len, input_size]

    def decode(self, z, x, seq_len):
        """Decode a given `z` and initial input state `x` to the desired
        `seq_len`."""
        qz = self.dec_input(z)
        # permute qz (inverse of h_n reshape)
        batch_size = qz.shape[0]
        h_i = qz.reshape(batch_size, -1, self.hidden_size)
        h_i = h_i.permute(1, 0, 2).contiguous()
        c_i = torch.zeros_like(h_i)

        # starting from the initial state, iterate forwards until reaching the
        # desired seq_len
        x_i = x
        y = torch.zeros(batch_size, seq_len, self.input_size).to(self.device)
        for i in range(seq_len):
            y_i, (h_i, c_i) = self.dec_lstm(x_i, (h_i, c_i))
            y_i = self.dec_output(y_i)
            x_i = y_i
            y[:, i:i+1] = y_i

        return y

    def forward(self, x):
        y_e, (h_n, c_n) = self.enc_lstm(x)

        # permute h_n
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n.flatten(start_dim=1)

        mu = self.enc_mean(h_n)
        logvar = self.enc_logvar(h_n)
        sd = torch.exp(0.5 * logvar)

        # reparameterize using N(0, I)
        eps = torch.randn_like(sd)
        z = mu + sd * eps

        # conditional decoder (sees generated output)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x_d = x[:, 0:1]
        y_d = self.decode(z, x_d, seq_len)

        ## recycle input (incorrect: this just leads to learning identity)
        #qz = self.dec_input(z)
        ## permute qz (inverse of h_n reshape)
        #batch_size = qz.shape[0]
        #dec_hidden = torch.reshape(qz, (batch_size, -1, self.hidden_size))
        #dec_hidden = torch.permute(dec_hidden, (1, 0, 2)).contiguous()
        #dec_cell = torch.zeros_like(dec_hidden)

        #x_d = x
        #y_d, (h_n, c_n) = self.dec_lstm(x_d, (dec_hidden, dec_cell))
        #y_d = self.dec_output(y_d)
        return y_d, mu, logvar

    def kld_loss(self, mu, logvar):
        """Note that torch's KLDivLoss is not the right function to use here.
        We have to manually implement KLD loss."""
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        return kld_loss

    def recon_loss(self, y, x):
        weight = torch.ones_like(x)
        ## decrease weight of zeros
        #weight[x != 0] = 0.1
        l2norm = (y - x) ** 2
        l2norm *= weight
        return torch.mean(l2norm)
        #return nn.functional.mse_loss(y, x)

    def loss(self, y, x, mu, logvar, kld_weight: float = 1.0):
        """VAE loss, which is a combination of KLD and reconstruction loss."""
        kld_loss = self.kld_loss(mu, logvar)
        recon_loss = self.recon_loss(y, x)

        # TODO: might need to adjust weighting of losses
        # KLD -> how close is the latent space to Gaussian
        # reconstruction -> how well does autoencoder reconstruct input
        return recon_loss + (kld_loss * kld_weight)

class KLDWeight:
    def __init__(self, length: int = 100):
        """Calculate the current KLD weight after `n` epochs, such that the
        weight reaches 1 after `length` epochs."""
        self.length = length

    def __call__(self, epoch: int):
        """Return the weight for the current `epoch`."""
        return 1. / (1. + math.exp(self.length / 2 - epoch))

def normalized_rmse(mse: float, delta: float) -> float:
    """Given the MSE (i.e., recon loss), return the RMSE normalized using the
    range of the underlying data."""
    return np.sqrt(mse) / delta

def train_model(save_path, model, train_loader, val_loader,
                #delta: int,
                epochs: int = 10000):
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(model.device)
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    opt_val_loss = float("inf")
    opt_train_loss = float("inf")
    opt_model = model.state_dict()
    #delay = 50
    delay = 0
    patience = 10
    stop_n = 0
    kld_weight = KLDWeight(delay)
    with tqdm.tqdm(range(epochs), desc="[? | ?] Epoch") as pbar:
        for epoch in pbar:
            train_loss = 0.0
            train_kld_loss = 0.0
            train_recon_loss = 0.0
            model.train()
            for i, batch in enumerate(train_loader):
                x = batch.to(model.device)
                optimizer.zero_grad()
                y, mu, logvar = model(x)
                loss = model.loss(y, x, mu, logvar)#, kld_weight(epoch))
                train_loss += loss.item()
                train_kld_loss += model.kld_loss(mu, logvar).item()
                train_recon_loss += model.recon_loss(y, x).item()
                loss.backward()
                optimizer.step()
            train_loss /= train_batches
            train_kld_loss /= train_batches
            train_recon_loss /= train_batches

            # validate and check for early stopping
            val_loss = 0.0
            val_kld_loss = 0.0
            val_recon_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    x = batch.to(model.device)
                    y, mu, logvar = model(x)
                    val_loss += model.loss(y, x, mu, logvar).item()
                    val_kld_loss += model.kld_loss(mu, logvar).item()
                    val_recon_loss += model.recon_loss(y, x).item()
            val_loss /= val_batches
            val_kld_loss /= val_batches
            val_recon_loss /= val_batches

            desc = (f"[{train_loss:.3e} ({train_kld_loss:.3e} + {train_recon_loss:.3e})" +
                    f" | {val_loss:.3e} ({val_kld_loss:.3e} + {val_recon_loss:.3e})] Epoch")
            pbar.set_description(desc)
            if val_loss < opt_val_loss:
                opt_val_loss = val_loss
                opt_val_kld_loss = val_kld_loss
                opt_val_recon_loss = val_recon_loss
                opt_train_loss = train_loss
                opt_train_kld_loss = train_kld_loss
                opt_train_recon_loss = train_recon_loss
                stop_n = 0
                opt_model = model.state_dict()
                torch.save(opt_model, save_path)
            elif epoch >= delay:
                stop_n += 1

            if stop_n >= patience:
                break

    log.info("Final train/val loss: [%.5e (%.5e + %.5e) | %.5e (%.5e + %.5e)]",
             opt_train_loss, opt_train_kld_loss, opt_train_recon_loss,
             opt_val_loss, opt_val_kld_loss, opt_val_recon_loss)
    #log.info("Normalized RMSE: [%.5e | %.5e]",
    #         normalized_rmse(opt_train_recon_loss, delta),
    #         normalized_rmse(opt_val_recon_loss, delta))
    torch.save(opt_model, save_path)
    model.load_state_dict(opt_model)

def plot_vecs(y: torch.Tensor, x: torch.Tensor, out_path,
              bin_width: float = 900) -> None:
    """Plot an input `x` its autoencoded `y` in the same axis."""
    fig, (axh, axl) = plt.subplots(2, 1,
        gridspec_kw={"hspace": 0},
        layout="constrained",
        sharex=True, sharey=True,
        figsize=(8,6),
    )

    # force same scale for both plots
    vmax = max(x.max(), y.max())

    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad(cmap(0))

    qm = axl.pcolormesh(x.transpose(0, 1),
        cmap=cmap, norm=mpl.colors.LogNorm(vmin=0.1, vmax=vmax),
        rasterized=True
    )

    axh.pcolormesh(y.transpose(0, 1),
        cmap=cmap, norm=mpl.colors.LogNorm(vmin=0.1, vmax=vmax),
        rasterized=True
    )

    axl.set_xlim(0, x.shape[0])
    axl.set_ylim(0, x.shape[1])
    fig.colorbar(qm, ax=fig.get_axes())

    xmax = x.max()
    step = 96 # tick every 24 h, assuming 900 second bins
    def hour_formatter(x: float, pos) -> str:
        return f"{x / 4:.0f}"
    axl.xaxis.set_major_formatter(hour_formatter)
    axl.set_xticks(np.arange(0, xmax, step))
    #axl.set_xlim(0, xmax)
    axl.set_xlabel("Time (h)")
    axl.margins(x=0)
    axl.set_ylabel("True Neighborhoods")
    axh.set_ylabel("Autoencoded Neighborhoods")

    log.info("writing %s...", out_path)
    fig.savefig(out_path)
    plt.close(fig)

def normalize_mat(mat: torch.Tensor, train_end: Optional[int] = None) -> None:
    """Normalize each row of `mat`, optionally computing the max and min only
    up until `train_end`."""
    if train_end is None:
        train_end = mat.shape[0]
    for nbr in range(mat.shape[1]):
        hi = mat[:train_end, nbr].max()
        if hi != 0:
            mat[:, nbr] /= hi

def load_set_mat(cfg, set_name, n_hops: int = 1, bin_width: float = 900) -> torch.Tensor:
    path = f"{cfg.neighborhoods}/{set_name}_labels_{n_hops}hop.pkl.xz"
    log.info("reading %s...", path)
    labels, times, *_ = util.pickle_read(path)

    log.info("generating input matrix...")
    mat, _ = neighborhood.autocorrelation.fixed_width_vectors(
        times, labels, width=bin_width
    )

    mat = torch.from_numpy(mat).float()
    return mat, times

def train_set(cfg, set_name: str, n_hops: int = 1) -> None:
    bin_width = 900
    mat, times = load_set_mat(cfg, set_name, n_hops, bin_width=bin_width)
    # 100 bin window size for 25-hour windows, which cover 1 period
    window_size = 100

    # train:test split based on times
    # awkward: should we split exactly based on times? this will fragment the
    # bins! Otherwise, we can split based on the bins, but this will not
    # exactly match the times. Latter seems better; be sure to floor the bin
    t_end = gnn.util.benign_day_edges.e3_time_train_end(set_name) / 1e9
    val_delta = t_end - times.min()
    val_bins = int(val_delta / bin_width)
    # val is 10% of train
    train_bins = int(val_bins * 0.9)

    # this can result in having too few val bins to form a full window. Two
    # possible approaches:
    # 1. Use a larger split to ensure at least one full val window
    # 2. Allow val windows to be small

    # find the first all-zero row in training
    normalize_mat(mat, train_bins)
    train_mat = mat[:train_bins]
    train_ignore = (train_mat.sum(dim=0) == 0).nonzero()[0].item()
    mat = mat[:, :train_ignore]
    train_mat = mat[:train_bins]
    val_mat = mat[train_bins:val_bins]
    input_size = mat.shape[1]

    train_set = NbrDataset(train_mat, window_size)
    # strategy 2: just allow window size to be smaller
    if window_size > val_bins - train_bins:
        val_set = NbrDataset(val_mat, window_size=None)
    else:
        val_set = NbrDataset(val_mat, window_size)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16)

    save_path = f"{cfg.nbr_autoencoder}/{set_name}_lstmvae.pt"
    model = LSTMVAE(input_size, hidden_size=128, latent_size=64)
    log.info("training LSTM VAE...")
    train_model(save_path, model, train_loader, val_loader)

def train_set_per_nbr(cfg, set_name: str, n_hops: int = 1) -> None:
    bin_width = 900
    mat, times = load_set_mat(cfg, set_name, n_hops, bin_width=bin_width)
    input_size = 1 # one model per neighborhood
    # 100 bin window size for 25-hour windows, which cover 1 period
    window_size = 100

    # train:test split based on times
    # awkward: should we split exactly based on times? this will fragment the
    # bins! Otherwise, we can split based on the bins, but this will not
    # exactly match the times. Latter seems better; be sure to floor the bin
    t_end = gnn.util.benign_day_edges.e3_time_train_end(set_name) / 1e9
    val_delta = t_end - times.min()
    val_bins = int(val_delta / bin_width)
    # val is 10% of train
    train_bins = int(val_bins * 0.9)

    # this can result in having too few val bins to form a full window. Two
    # possible approaches:
    # 1. Use a larger split to ensure at least one full val window
    # 2. Allow val windows to be small

    train_mat = mat[:train_bins]
    val_mat = mat[train_bins:val_bins]
    # find the first all-zero row in training
    train_ignore = (train_mat.sum(dim=0) == 0).nonzero()[0].item()

    # train a model for every remaining neighborhood
    save_dir = f"{cfg.nbr_autoencoder}/{set_name}"
    pathlib.Path(save_dir).mkdir(exist_ok=True)
    for i in tqdm.tqdm(range(train_ignore), desc="Total models"):
        delta = train_mat[:, i].max() - train_mat[:, i].min()
        train_set = NbrDataset(train_mat[:, i].unsqueeze(1), window_size)
        # strategy 2: just allow window size to be smaller
        if window_size > val_bins - train_bins:
            val_set = NbrDataset(val_mat[:, i].unsqueeze(1), window_size=None)
        else:
            val_set = NbrDataset(val_mat[:, i].unsqueeze(1), window_size)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=16)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=16)

        save_path = f"{save_dir}/nbr{i}.pt"
        # need much smaller parameters for individual models
        model = LSTMVAE(input_size, hidden_size=32, latent_size=8)
        log.info("training LSTM VAE...")
        train_model(save_path, model, train_loader, val_loader, delta=delta)


def test_set(cfg, set_name: str, n_hops: int = 1) -> None:
    bin_width = 900
    mat, times = load_set_mat(cfg, set_name, n_hops)

    # train:test split based on times
    # awkward: should we split exactly based on times? this will fragment the
    # bins! Otherwise, we can split based on the bins, but this will not
    # exactly match the times. Latter seems better; be sure to floor the bin
    t_end = gnn.util.benign_day_edges.e3_time_train_end(set_name) / 1e9
    val_delta = t_end - times.min()
    val_bins = int(val_delta / bin_width)
    train_bins = int(val_bins * 0.9)

    # this can result in having too few val bins to form a full window. Two
    # possible approaches:
    # 1. Use a larger split to ensure at least one full val window
    # 2. Allow val windows to be small

    # find the first all-zero row in training
    normalize_mat(mat, train_bins)
    train_mat = mat[:train_bins]
    train_ignore = (train_mat.sum(dim=0) == 0).nonzero()[0].item()
    mat = mat[:, :train_ignore]
    train_mat = mat[:train_bins]
    input_size = mat.shape[1]

    dataset = NbrDataset(mat, None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    log.info("loading pretrained model...")
    save_path = f"{cfg.nbr_autoencoder}/{set_name}_lstmvae.pt"
    model = LSTMVAE(input_size, hidden_size=128, latent_size=64)
    model.load_state_dict(torch.load(save_path))
    model.to(model.device)
    model.eval()

    log.info("running inference...")
    # should only have one
    assert len(loader) == 1
    x = next(iter(loader)).to(model.device)
    y, mu, logvar = model(x)
    kld_loss = model.kld_loss(mu, logvar).item()
    recon_loss = model.recon_loss(y, x).item()
    loss = model.loss(y, x, mu, logvar).item()

    y = y.detach().cpu().squeeze()
    x = x.detach().cpu().squeeze()

    log.info("loss = %.5e (%.5e + %.5e)", loss, kld_loss, recon_loss)

    plot_vecs(y, x, f"{set_name}_lstmvae.pdf")

def test_set_per_nbr(cfg, set_name: str, n_hops: int = 1) -> None:
    bin_width = 900
    mat, times = load_set_mat(cfg, set_name, n_hops, bin_width=bin_width)
    input_size = 1 # one model per neighborhood
    # 100 bin window size for 25-hour windows, which cover 1 period
    window_size = 100

    models_dir = f"{cfg.nbr_autoencoder}/{set_name}"
    # reconstruct the full plots using the individual
    full_y = {}
    full_x = {}
    nrmses = []
    for path in tqdm.tqdm(glob.glob(f"{models_dir}/nbr*.pt")):
        nbr = int(path.rsplit("r", maxsplit=1)[1].split(".", maxsplit=1)[0])
        dataset = NbrDataset(mat[:, nbr].unsqueeze(1), None)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        model = LSTMVAE(input_size, hidden_size=32, latent_size=8)
        model.load_state_dict(torch.load(path))
        model.to(model.device)
        model.eval()

        # should only have one
        assert len(loader) == 1
        x = next(iter(loader)).to(model.device)
        y, mu, logvar = model(x)
        kld_loss = model.kld_loss(mu, logvar).item()
        recon_loss = model.recon_loss(y, x).item()
        loss = model.loss(y, x, mu, logvar).item()
        nrmse = normalized_rmse(recon_loss, y.max() - y.min()).detach().cpu()
        #nrmse = normalized_rmse(recon_loss, 1)

        y = y.detach().cpu().squeeze()
        x = x.detach().cpu().squeeze()

        log.info("nbr %s: loss = %.5e (%.5e + %.5e); NRMSE = %.5e",
                 nbr, loss, kld_loss, recon_loss, nrmse)
        nrmses.append(nrmse)

        assert nbr not in full_y
        full_y[nbr] = y
        full_x[nbr] = x

    # plot nrmses
    fig, ax = plt.subplots()
    ax.hist(nrmses, bins="auto", histtype="step")
    ax.set_xlabel("Normalized RMSE")
    ax.set_ylabel("Frequency")
    ax.set_xscale("log")
    fig.savefig(f"{set_name}_lstmvae_nrmse.pdf")
    plt.close(fig)

    full_size = max(full_y.keys())
    assert full_size + 1 == len(full_y)
    y = []
    x = []
    for i in range(full_size):
        y.append(full_y[i])
        x.append(full_x[i])
    y = torch.stack(y).transpose(0, 1)
    x = torch.stack(x).transpose(0, 1)

    plot_vecs(y, x, f"{set_name}_lstmvae.pdf")


def main(argv):
    #train_set_per_nbr(config.e3, "cadets")
    #test_set_per_nbr(config.e3, "cadets")
    train_set(config.e3, "cadets")
    test_set(config.e3, "cadets")
