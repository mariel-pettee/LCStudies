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
    parser.add_argument("save_dir", help="Output directory of graph .pkl files")
    return parser.parse_args()

def make_graph(event: pd.Series, geo_df: pd.DataFrame, is_charged=False):
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

def get_worker_events(df, n_workers, worker_id):
    return np.array_split(range(len(df)), n_workers)[worker_id]

if __name__ == "__main__":
    args = get_args()
    
    print("Parallelizing with {} total workers.".format(args.total_workers))
    print("This is worker #{}.".format(args.worker_id))
    print("Saving graphs to folder: {}".format(args.save_dir))
    
    ### Load ROOT file
    file_path = '../data/neutral_pion_sample.root'
    f_pi0 = uproot.open(file_path)
    
    ### Define primary dataframe
    df = f_pi0['EventTree'].arrays(["cluster_cell_E", "cluster_cell_ID", "cluster_E", "cluster_Eta", "cluster_Phi"], library="pd")
    df.reset_index(inplace=True) # flatten MultiIndexing
    df = df[:100] ### TEMPORARY -- limit to first 100 events

    ### Define cell geometry dataframe
    df_geo = f_pi0['CellGeo'].arrays(library="pd")
    df_geo = df_geo.reset_index() # remove redundant multi-indexing
    df_geo.drop(columns = ["entry", "subentry"], inplace=True)

    ### Add x,y,z coordinates
    df_geo["cell_geo_x"] = df_geo["cell_geo_rPerp"] * np.cos(df_geo["cell_geo_phi"])
    df_geo["cell_geo_y"] = df_geo["cell_geo_rPerp"] * np.sin(df_geo["cell_geo_phi"])
    cell_geo_theta = 2*np.arctan(np.exp(-df_geo["cell_geo_eta"]))
    df_geo["cell_geo_z"] = df_geo["cell_geo_rPerp"] / np.tan(cell_geo_theta)
    
    ### Get assigned dataframe rows for this worker
    worker_events = get_worker_events(df, args.total_workers, args.worker_id)
    print("Generating graphs for events {} - {}.".format(worker_events[0], worker_events[-1]))
    
    ### Make the graphs for the specified events
    graph_list = []
    for i in tqdm(worker_events):
        graph_list.append(make_graph(df.iloc[i], geo_df=df_geo, is_charged=False))
    
    ### Save Pickle file, with zero-indexing: 
    filepath = os.path.join(args.save_dir,'pi0_graphs_chunk_'+str(args.worker_id)+'_of_'+str(args.total_workers - 1)+'.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(graph_list, f)