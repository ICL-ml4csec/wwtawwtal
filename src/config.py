"""Configuration, e.g., paths of intermediate files."""

import datetime
import logging
import pathlib
from types import SimpleNamespace

import matplotlib as mpl

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s:%(levelname)s:%(name)s] %(message)s",
                    handlers=[
                        logging.FileHandler("main.log", mode="a"),
                        logging.StreamHandler(),
                    ])

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["savefig.bbox"] = "tight"
# need a tiny bit of padding, else lines can get cut off
mpl.rcParams["savefig.pad_inches"] = 0.02
mpl.rcParams["savefig.dpi"] = 600
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["axes.autolimit_mode"] = "round_numbers"
mpl.rcParams["axes.xmargin"] = 0
mpl.rcParams["axes.ymargin"] = 0
mpl.rcParams["figure.autolayout"] = True

intermediates = "intermediates"

atlasv1 = SimpleNamespace(intermediates=f"{intermediates}/atlasv1")
atlasv1.log_dataframes = f"{atlasv1.intermediates}/log_dataframes"
atlasv1.graphs = f"{atlasv1.intermediates}/graphs"
atlasv1.names = f"{atlasv1.intermediates}/names"
atlasv1.neighborhoods = f"{atlasv1.intermediates}/neighborhoods"
atlasv1.events = f"{atlasv1.intermediates}/events"

for d in atlasv1.__dict__.values(): # make directories now
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

atlasv2 = SimpleNamespace(intermediates=f"{intermediates}/atlasv2")
atlasv2.addresses = f"{atlasv2.intermediates}/addresses"
atlasv2.graphs = f"{atlasv2.intermediates}/graphs"
atlasv2.names = f"{atlasv2.intermediates}/names"
atlasv2.neighborhoods = f"{atlasv2.intermediates}/neighborhoods"
atlasv2.events = f"{atlasv2.intermediates}/events"
atlasv2.labels = f"{atlasv2.intermediates}/labels"
atlasv2.rcaid_graphs_csv = f"{atlasv2.intermediates}/rcaid_graphs_csv"
atlasv2.streaming_gnn_roots = f"{atlasv2.intermediates}/streaming_gnn_roots"
atlasv2.nbr_autoencoder = f"{atlasv2.intermediates}/nbr_autoencoder"
atlasv2.flash = f"{atlasv2.intermediates}/flash"
atlasv2.kairos = f"{atlasv2.intermediates}/kairos"

for d in atlasv2.__dict__.values(): # make directories now
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

e3 = SimpleNamespace(intermediates=f"{intermediates}/darpa_e3")
e3.attack_subgraphs = f"{e3.intermediates}/attack_subgraphs"
e3.attack_event_uuids = f"{e3.intermediates}/attack_event_uuids"
e3.attack_node_uuids = f"{e3.intermediates}/attack_node_uuids"
e3.addresses = f"{e3.intermediates}/addresses"
e3.neighborhoods = f"{e3.intermediates}/neighborhoods"
e3.events = f"{e3.intermediates}/events"
e3.graphs = f"{e3.intermediates}/graphs"
e3.names = f"{e3.intermediates}/names"
e3.rcaid_graphs_csv = f"{e3.intermediates}/rcaid_graphs_csv"
e3.labels = f"{e3.intermediates}/labels"
e3.streaming_gnn_roots = f"{e3.intermediates}/streaming_gnn_roots"
e3.nbr_autoencoder = f"{e3.intermediates}/nbr_autoencoder"
e3.flash = f"{e3.intermediates}/flash"
e3.kairos = f"{e3.intermediates}/kairos"

for d in e3.__dict__.values(): # make directories now
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

streamspot = SimpleNamespace(intermediates=f"{intermediates}/streamspot")
streamspot.addresses = f"{streamspot.intermediates}/addresses"
streamspot.events = f"{streamspot.intermediates}/events"
streamspot.graphs = f"{streamspot.intermediates}/graphs"
streamspot.neighborhoods = f"{streamspot.intermediates}/neighborhoods"
streamspot.labels = f"{streamspot.intermediates}/labels"
streamspot.streaming_gnn_roots = f"{streamspot.intermediates}/streaming_gnn_roots"
streamspot.rcaid_graphs_csv = f"{streamspot.intermediates}/rcaid_graphs_csv"
streamspot.nbr_autoencoder = f"{streamspot.intermediates}/nbr_autoencoder"
streamspot.flash = f"{streamspot.intermediates}/flash"
streamspot.kairos = f"{streamspot.intermediates}/kairos"

for d in streamspot.__dict__.values(): # make directories now
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

graph_transforms = SimpleNamespace(intermediates=f"{intermediates}/graph_transforms")
graph_transforms.ip_whois = f"{graph_transforms.intermediates}/ip_whois"

for d in graph_transforms.__dict__.values(): # make directories now
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

optc = SimpleNamespace(intermediates=f"{intermediates}/optc")
optc.graphs = f"{optc.intermediates}/graphs"
optc.neighborhoods = f"{optc.intermediates}/neighborhoods"
optc.names = f"{optc.intermediates}/names"
optc.flash = f"{optc.intermediates}/flash"
optc.kairos = f"{optc.intermediates}/kairos"
optc.timesliced_graphs = f"{optc.intermediates}/timesliced_graphs"
optc.rcaid_graphs_csv = f"{optc.intermediates}/rcaid_graphs_csv"
optc.streaming_gnn_roots = f"{optc.intermediates}/streaming_gnn_roots"
optc.nbr_autoencoder = f"{optc.intermediates}/nbr_autoencoder"
optc.flash = f"{optc.intermediates}/flash"
optc.kairos = f"{optc.intermediates}/kairos"
optc.labels = f"{optc.intermediates}/labels"

for d in optc.__dict__.values(): # make directories now
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

pickle_protocol = 4 # need to be compatible with 3.6.8
