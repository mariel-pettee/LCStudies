### Generic imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import scipy
import uproot
from tqdm import tqdm
import functools
import pickle

### ML-related
import tensorflow as tf
import atlas_mpl_style as ampl
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import sonnet as snt

### GNN-related
from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
import networkx as nx

### Other setup 
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 20)

params = {'legend.fontsize': 13, 'axes.labelsize': 18}
plt.rcParams.update(params)

SEED = 15
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ### GPU Setup
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" # pick a number between 0 & 3
# gpus = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(gpus[0], True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("total_workers", type=int, help="Total number of Slurm worker nodes")
    parser.add_argument("worker_id", type=int, help="Slurm worker node ID number")
    parser.add_argument("input_dir", help="Input directory of .root files")
    parser.add_argument("save_dir", help="Output directory of graph .pkl files")
    parser.add_argument("is_charged", default=False, action='store_true', help="Use this flag for charged pion samples.")
    return parser.parse_args()

def make_graph(event: pd.Series, geo_df: pd.DataFrame, is_charged=args.is_charged):
    """
    Creates a graph representation of an event
    
    inputs
    event (pd.Series) one event/row from EventTree
    geo_df (pd.DataFrame) the CellGeo DataFrame mapping cell_geo_ID to information about the cell
    is_charged (bool) True for charged pion, False for uncharged pion
    
    returns
    A pair of graph representations of the event for the GNN (train_graph, target_graph)
    returns (None, None) if no cell energies detected
    """
    
    ### No cell energies present
    if len(event["cluster_cell_E"]) == 0:
        return None, None
    
    ### Get cell geometry information for this particular event
    temp_df = geo_df[geo_df["cell_geo_ID"].isin([item for sublist in event["cluster_cell_ID"] for item in sublist])]
    temp_df = temp_df.set_index("cell_geo_ID")
    ### Assign cell energies
    for cell_id, cell_e in zip(
        [item for sublist in event["cluster_cell_ID"] for item in sublist],
        [item for sublist in event["cluster_cell_E"] for item in sublist]
    ):
        temp_df.loc[int(cell_id), "cell_E"] = cell_e
    
    ### Define node features
    n_nodes = temp_df.shape[0]
    node_features = ["cell_E", "cell_geo_eta",
                     "cell_geo_phi", "cell_geo_rPerp",
                     "cell_geo_deta", "cell_geo_dphi",
                     "cell_geo_volume"]
    nodes = temp_df[node_features].to_numpy(dtype=np.float32).reshape(-1, len(node_features))
    
    ### Apply k-NN search to find cell neighbors
    # NOTE FAIR also has a faster algo for KNN search. Might want to try it
    k = 6
    k = min(n_nodes, k)
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(temp_df[["cell_geo_x", "cell_geo_y", "cell_geo_z"]])
    distances, indices = nbrs.kneighbors(temp_df[["cell_geo_x", "cell_geo_y", "cell_geo_z"]])
    
    senders = np.repeat([x[0] for x in indices], k-1)               # k-1 for no self edges
    receivers = np.array([x[1:] for x in indices]).flatten()        # x[1:] for no self edges
    edges = np.array([x[1:] for x in distances], dtype=np.float32).flatten().reshape(-1, 1)
    n_edges = len(senders)
        
    global_features = ["cluster_E", "cluster_Eta", "cluster_Phi"]
    global_values = np.asarray(event[global_features]).astype('float32')
    
    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": global_values            # np.array([n_nodes], dtype=np.float32)
    }
    
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([int(is_charged)], dtype=np.float32)
    }

    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    
    return input_graph, target_graph

def divide_chunks(l, n):
    return [l[i::n] for i in range(n)]

if __name__ == "__main__":
    args = get_args()
    
    print("Parallelizing with {} total workers.".format(args.total_workers))
    print("This is worker #{}.".format(args.worker_id))
    print("Saving graphs to folder: {}".format(args.save_dir))
    
    ### Get the list of files for which this worker is responsible
    files = glob(os.path.join(args.input_dir,'*.root'))
    chunks = list(divide_chunks(files, args.total_workers))
    worker_files = chunks[args.worker_id]
    print("{} files for worker #{}:".format(len(chunks[args.worker_id]),args.worker_id))
    print(worker_files) 
    
    for file in tqdm(worker_files):
        ### Define primary dataframe
        df = file['EventTree'].arrays(["cluster_cell_E", "cluster_cell_ID", "cluster_E", "cluster_Eta", "cluster_Phi"], library="pd")
        df.reset_index(inplace=True) # flatten MultiIndexing

        ### Define cell geometry dataframe
        df_geo = file['CellGeo'].arrays(library="pd")
        df_geo = df_geo.reset_index() # remove redundant multi-indexing
        df_geo.drop(columns = ["entry", "subentry"], inplace=True)

        ### Add x,y,z coordinates
        df_geo["cell_geo_x"] = df_geo["cell_geo_rPerp"] * np.cos(df_geo["cell_geo_phi"])
        df_geo["cell_geo_y"] = df_geo["cell_geo_rPerp"] * np.sin(df_geo["cell_geo_phi"])
        cell_geo_theta = 2*np.arctan(np.exp(-df_geo["cell_geo_eta"]))
        df_geo["cell_geo_z"] = df_geo["cell_geo_rPerp"] / np.tan(cell_geo_theta)

        ### Make the graphs for the specified events
        graph_list = []
        for i in range(len(df)):
            graph_list.append(make_graph(df.iloc[i], geo_df=df_geo, is_charged=args.is_charged))

        ### Save Pickle file, with zero-indexing: 
        filepath = os.path.join(args.save_dir,file.split('.')[-2][1:]+'.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(graph_list, f)