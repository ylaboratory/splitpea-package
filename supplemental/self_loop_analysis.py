""" 
Analyzing the effect of self-loops in Splitpea networks

We will load the networks generated in `/examples/sample_mode_multi.ipynb`.  
In that tutorial, the networks were saved as pickled NetworkX graphs in:

`ucs_rewired_networks/`

We will do two things:
1. Measure how common self-loops are across the saved networks.
2. Compare the node degree distributionwith vs without self-loops 
"""

# INPUT PATHS
NET_DIR = "ucs_rewired_networks"   # path to folder created by /examples/sample_mode_multi.ipynb

import os
import glob
import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


 
#  1) Load saved networks

search_pattern = os.path.join(NET_DIR, "*.pickle")
files = glob.glob(search_pattern)

print(f"Found {len(files)} files in {NET_DIR}")
files[:5]


 
#  2) Self-loop prevalence across networks

self_loop_rates = []

for file_path in files:
    try:
        with open(file_path, "rb") as f:
            G = pickle.load(f)

        if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            print(f"Skipping {file_path}: Not a NetworkX graph object.")
            continue

        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            print(f"Skipping {file_path}: Graph is empty.")
            continue

        n_self_loops = nx.number_of_selfloops(G)
        rate = n_self_loops / n_nodes
        self_loop_rates.append(rate)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

plt.figure(figsize=(10, 6))
plt.hist(self_loop_rates, bins=20)
plt.title("proportion of nodes with self-loops")
plt.xlabel("proportion")
plt.ylabel("number of networks")
plt.tight_layout()
plt.savefig("self_loop_edges_per_node_hist.pdf", format="pdf")
plt.show()
plt.close()


 
#  3) Degree distributions with vs without self-loops 

def aggregate_node_degrees(net_dir, include_self_loops: bool):
    """
    Returns a list of node degrees aggregated across all graphs.
    Each node contributes one degree value per graph it appears in.
    """
    degrees = []

    for f in glob.glob(net_dir + "/*.pickle"):
        g = pickle.load(open(f, "rb"))

        if not include_self_loops:
            g = g.copy()
            g.remove_edges_from([(u, v) for u, v in g.edges() if u == v])

        g = g.copy()
        g.remove_edges_from([(u, v) for u, v in g.edges() if g[u][v].get("chaos", False)])

        degrees.extend([deg for _, deg in g.degree()])

    return np.array(degrees, dtype=float)



deg_with = aggregate_node_degrees(NET_DIR, include_self_loops=True)
deg_without = aggregate_node_degrees(NET_DIR, include_self_loops=False)

d_with = np.log10(deg_with[deg_with > 0])
d_without = np.log10(deg_without[deg_without > 0])

bins = 20
plt.figure(figsize=(10, 6))
plt.hist(d_with, bins=bins, alpha=0.5, label="with self-loops")
plt.hist(d_without, bins=bins, alpha=0.5, label="without self-loops")
plt.title(f"node degree distribution across {len(files)} patient networks")
plt.xlabel("log10(degree)")
plt.ylabel("count")
plt.legend()
plt.tight_layout()
plt.savefig("degree_overlay_log10.pdf")
plt.show()
plt.close()