# This python file is used to train the Weighted  Multiplex Network Embedding model
# Original Author: Hongming ZHANG, HKUST KnowComp Group
import networkx as nx
import Random_walk
from WMNE import *
import sys

file_name = './data/Fusion6Net.txt'
edge_data_by_type, all_edges, all_nodes = load_network_data(file_name)
model = train_model(edge_data_by_type)
save_model(model, 'model')