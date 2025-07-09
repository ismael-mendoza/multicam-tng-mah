#!/usr/bin/env python3
"""Add `IndexInHaloTable` to trees for convenience."""

import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

tng_path = "data/processed/tng"
tree_file = f"{tng_path}/TNG300-1-Dark_cut_trees.p"
new_tree_file = f"{tng_path}/TNG300-1-Dark_cut_trees2.p"
halo_snap_files = (
    "data/processed/tng/TNG300-1-Dark/TNG300-1-Dark_HaloHistory_MBP_snap{}.hdf5"
)

TNG_trees_clean = pickle.load(open(tree_file, "rb"))

N_SNAPS = 100


halos = {}
for i in range(N_SNAPS):
    halos[i] = pd.read_hdf(halo_snap_files.format(i), key="Halos")
    print(i, end="")


for tree in TNG_trees_clean:
    IndexInHaloTable = np.zeros_like(tree["SubhaloGrNr"]) - 1
    for i, grnr in enumerate(tree["SubhaloGrNr"]):
        sn = tree["SnapNum"][i]
        if sn > 2:
            IndexInHaloTable[i] = np.where(halos[sn]["HaloID"] == grnr)[0][0]
    tree["IndexInHaloTable"] = IndexInHaloTable

assert "IndexInHaloTable" in TNG_trees_clean[0].keys()


# save new file
pickle.dump(TNG_trees_clean, open(new_tree_file, "wb"))
