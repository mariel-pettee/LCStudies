### Generic imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import scipy
import uproot
from tqdm import tqdm
import functools
from glob import glob
from multiprocessing import Pool
import time
import pickle
import argparse

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_processes", type=int, help="Number of processes for multiprocessing (maximum = number of CPUs available)")
    parser.add_argument("--is_charged", default=False, action='store_true', help="Use this flag for charged pion samples.")
    return parser.parse_args()

def async_tqdm(func, argument_list, num_processes):
    pool = Pool(processes=num_processes)
    jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in tqdm(jobs, desc="Jobs"):
        result_list_tqdm.append(job.get())
    return result_list_tqdm

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
        "globals": global_values
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

def process_file(file, is_charged: bool = False):
    ### Define primary dataframe
    f = uproot.open(file)
    df = f['EventTree'].arrays(["cluster_cell_E", "cluster_cell_ID", "cluster_E", "cluster_Eta", "cluster_Phi"], library="pd")
    df.reset_index(inplace=True) # flatten MultiIndexing

    ### Define cell geometry dataframe
    df_geo = f['CellGeo'].arrays(library="pd")
    df_geo = df_geo.reset_index() # remove redundant multi-indexing
    df_geo.drop(columns = ["entry", "subentry"], inplace=True)

    ### Add x,y,z coordinates
    df_geo["cell_geo_x"] = df_geo["cell_geo_rPerp"] * np.cos(df_geo["cell_geo_phi"])
    df_geo["cell_geo_y"] = df_geo["cell_geo_rPerp"] * np.sin(df_geo["cell_geo_phi"])
    cell_geo_theta = 2*np.arctan(np.exp(-df_geo["cell_geo_eta"]))
    df_geo["cell_geo_z"] = df_geo["cell_geo_rPerp"] / np.tan(cell_geo_theta)

    ### Make the graphs for the specified events
    graph_list = []
    n_events = 100 # limit dataframe size for testing
    for i in range(len(df[:n_events])):
        graph_list.append(make_graph(df.iloc[i], geo_df=df_geo, is_charged=is_charged))

    ### Save Pickle file, with zero-indexing:
    if is_charged == False:
        save_dir = "/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/graphs/neutral_pion/"
    elif is_charged == True:
        save_dir = "/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/graphs/charged_pion/"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir,file.split('.')[-2][1:]+'.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(graph_list, f)
        
if __name__ == "__main__":
    args = get_args()
            
    print("{} CPUs available.".format(os.cpu_count()))
    print("Using {} processes.".format(args.num_processes))
    
    pi0_files = glob('/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/data/*singlepi0*/*.root')
    pion_files = glob('/global/cfs/cdirs/m3246/mpettee/ml4pions/LCStudies/data/*singlepion*/*.root')
    
    if args.is_charged: 
        print("Processing CHARGED pion samples.")
        file_list = pion_files
        is_charged = [True]*len(file_list)
    else:
        print("Processing NEUTRAL pion samples.")
        file_list = pi0_files
        is_charged = [False]*len(file_list)
        
    ### Generate graphs in parallel via multiprocessing
    async_tqdm(func=process_file, argument_list=zip(file_list, is_charged), num_processes=args.num_processes)
    
