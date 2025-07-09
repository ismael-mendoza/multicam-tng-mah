"""Several utility functions to read data from TNG catalogs."""

import pickle

import h5py
import numpy as np
import pandas as pd
from scipy import spatial
from tqdm import tqdm

SNAPS = np.arange(0, 100, 1)
TNG_H = 0.6774  # from website

MISSING = -999999999


def get_vvir(rvir, mvir):
    """Get vvir from rvir and mvir.

    Args:
        rvir: virial radius in kpc/h
        mvir: virial mass in Msun/h
    """
    k = 6.674e-11 * 1.988435e30 / 3.086e19
    vvir_mks = np.sqrt(k * mvir / rvir)
    vvir = vvir_mks / 1e3
    return vvir


def convert_tng_mass(gmass):
    """Convert TNG mass to log10(Msun)."""
    # TNG units are 1e10 Msun / h; https://www.tng-project.org/data/docs/specifications
    # return in units of log10(Msun)
    # robust to 0 mass
    return np.where(gmass > 0, np.log10(gmass * 1e10 / TNG_H), 0)


def get_mpeak_from_mah(mah: np.ndarray):
    """Compute m_peak from mah."""
    assert mah.ndim == 2
    m_peak1 = np.fmax.accumulate(10**mah, axis=1)
    m_peak = m_peak1 / m_peak1[:, -1][:, None]
    return m_peak


def get_vmax_over_vvir(cat: pd.DataFrame):
    """Compute vmax / vvir from catalog columns."""
    # compute vvir and create new column

    # ensure units of mvir is in units of Msun / h
    mvir = 10 ** cat["Mvir"].values / TNG_H  # og units: log10(msun)
    rvir = cat["Rvir"].values / TNG_H  # og units: kpc
    vvir = get_vvir(rvir, mvir)

    return cat["Vmax_DM"] / vvir


def _reverse_trees(trees):
    """Reverse each entry in trees so that order is from early to late times."""
    for tree in trees:
        for key in tree.keys():
            if key not in ["Number", "ChunkNumber", "TreeID"]:
                tree[key] = tree[key][::-1]
    return trees


def read_trees(trees_file: str):
    """Read in the trees file and convert masses to log10(M/Msun)."""
    with open(trees_file, "rb") as pickle_file:
        _trees = pickle.load(pickle_file)
        trees = _reverse_trees(_trees)
        for tree in trees:
            for k in tree.keys():
                if "Mass" in k or "_M_" in k:
                    tree[k] = convert_tng_mass(tree[k])
    return trees


def get_msmhmr(mstar, mvir, mass_bin=(11.5, 12.0), n_bins=11):
    """Compute mean stellar mass to halo mass relation and deviation."""
    # NOTE: Previously mstar we use `Mstar_30pkpc`
    # both masses are assumed to be in log units

    ratio = np.log10(10**mstar / 10**mvir)

    assert np.all(mvir > mass_bin[0]) and np.all(mvir < mass_bin[1])

    # compute mean ratio in bins of mvir
    bins = np.linspace(mass_bin[0], mass_bin[1], n_bins)
    mean_ratio_per_bin = np.zeros(len(bins) - 1)
    for ii in range(len(bins) - 1):
        idx = np.where((mvir > bins[ii]) & (mvir < bins[ii + 1]))[0]
        mean_ratio_per_bin[ii] = np.mean(ratio[idx])

    middle_point_of_bins = (bins[1:] + bins[:-1]) / 2

    m, b = np.polyfit(middle_point_of_bins, mean_ratio_per_bin, 1)

    # finally, calculate deviation from mean log ratio
    #  want \Delta Log ( M_star )
    m_star_dev = mstar - np.log10(10 ** (m * mvir + b) * 10**mvir)

    return m_star_dev, (m, b)


def get_color(color_file: str, cat: pd.DataFrame):
    """Read in color file and return dataframe with colors (in order of catalog)."""
    df_color = get_color_and_match(color_file, cat["SubhaloID"].values)
    return df_color


def get_color_and_match(color_file: str, subfind_ids: np.ndarray):
    assert np.all(np.sort(subfind_ids) == subfind_ids)  # needs to be sorted
    f = h5py.File(color_file, "r")

    colnames = (
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "wfc_acs_f606w",
        "des_y",
        "jwst_f150w",
    )
    arr = f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:]

    # average over projections
    arr = np.mean(arr, axis=-1)
    df_color = pd.DataFrame(arr, columns=colnames)

    color_ids = f["subhaloIDs"][:]

    f.close()

    df_color = df_color.iloc[np.where(np.isin(color_ids, subfind_ids))[0]]
    assert all(df_color.index.values == subfind_ids)

    return df_color


def match_dm_and_hydro_cat(cat: pd.DataFrame, dcat: pd.DataFrame):
    """Matching using KD tree on halo positions.

    We match based the distance between haloes divided by the virial radius
    of the candidate halo.

    """

    # get positions
    pos = np.array(cat[["pos_x", "pos_y", "pos_z"]])
    dpos = np.array(dcat[["pos_x", "pos_y", "pos_z"]])

    # construct kdtree
    tree = spatial.KDTree(pos)
    dtree = spatial.KDTree(dpos)

    # for each DM halo, indx is the index of the nearest hydro halo
    _, indx = tree.query(dpos)

    # for each hydro halo, dindx is the index of the nearest DM halo
    _, dindx = dtree.query(pos)

    # keep only bijectively matched haloes
    dmo_match = []
    hydro_match = []
    for ii in range(len(cat)):
        if ii == indx[dindx[ii]]:
            dmo_match.append(dindx[ii])
        else:
            dmo_match.append(MISSING)

    for ii in range(len(dcat)):
        if ii == dindx[indx[ii]]:
            hydro_match.append(indx[ii])
        else:
            hydro_match.append(MISSING)

    dmo_match = np.array(dmo_match)
    hydro_match = np.array(hydro_match)

    cat["dmo_match"] = dmo_match.astype(int)
    dcat["hydro_match"] = hydro_match.astype(int)

    cat["hydro_id"] = np.arange(len(cat))
    dcat["dmo_id"] = np.arange(len(dcat))

    return cat, dcat


def setup_mah_from_trees(
    trees_file: str,
    metadata_file: str,
    snaps: np.array,
    mass_bin=(12.8, 13.1),
    scale_cut=0.185,  # corresponds to original paper
):
    """Read in trees and present-day catalog, and return mah and catalog."""

    # get scales and redshifts
    metadata = pd.read_csv(metadata_file)
    zs = metadata["Redshift"].values
    scales = 1 / (1 + zs)
    assert len(snaps) == len(scales)
    n_snaps = len(snaps)

    # read trees and present-day catalog
    trees = read_trees(trees_file)

    # select trees in mass bin that have a snapshot at z=0
    trees = [
        t
        for t in trees
        if 99 in t["SnapNum"]
        and t["Group_M_TopHat200"][-1] > mass_bin[0]
        and t["Group_M_TopHat200"][-1] < mass_bin[1]
    ]

    # get mah from trees, and convert to correct units
    mah = np.zeros((len(trees), n_snaps))
    for ii, t in enumerate(trees):
        mah_t = np.zeros(n_snaps) * np.nan
        t_snaps = t["SnapNum"]
        gmass = t["Group_M_TopHat200"]
        mah_t[t_snaps] = gmass

        # linearly interpolate nan values
        mah_t = pd.Series(mah_t)
        mah_t = mah_t.interpolate(method="linear", limit_direction="forward", axis=0)
        mah[ii] = mah_t.values

    idx = np.where(scales > scale_cut)[0][0]
    snaps = snaps[idx:]
    scales = scales[idx:]
    mah = mah[:, idx:]

    # remove haloes with nans and hope not too many
    assert sum(np.isnan(mah.mean(axis=1))) < 100
    kp_idx = np.where(np.isnan(mah).sum(axis=1) == 0)[0]
    mah = mah[kp_idx]

    halo_idx = np.array([t["IndexInHaloTable"][-1] for t in trees])[kp_idx]

    return {
        "halo_idx": halo_idx,
        "mah": mah,
        "z": zs,
        "snaps": snaps,
        "scales": scales,
    }


def match_mah_and_cat(halo_idx: np.ndarray, cat: pd.DataFrame):
    """Match mah and catalog by halo index."""
    # halo_indx is original cat

    return cat.iloc[halo_idx]


def rematch_dm_and_hydro_cat(
    cat: pd.DataFrame, dcat: pd.DataFrame, mah: np.ndarray, dmah: np.ndarray
):
    """Match after matching by trees' halo index."""
    new_cat = pd.DataFrame(columns=cat.columns)
    new_mah = []
    new_dcat = pd.DataFrame(columns=dcat.columns)
    new_dmah = []

    for ii in tqdm(range(len(cat))):
        dmo_match_ii = cat["dmo_match"].values[ii]
        dmo_ids = dcat["dmo_id"].values
        is_in = np.isin(dmo_ids, dmo_match_ii)
        if sum(is_in) == 1 and dmo_match_ii != MISSING:
            dmo_idx = np.where(is_in)[0][0]
            jj = len(new_cat)
            new_cat.loc[jj] = dict(cat.iloc[ii])
            new_mah.append(mah[ii])
            new_dcat.loc[jj] = dict(dcat.iloc[dmo_idx])
            new_dmah.append(dmah[dmo_idx])

    new_mah = np.array(new_mah)
    new_dmah = np.array(new_dmah)

    # sort by "SubhaloID" for colors later
    subhalo_id = new_cat["SubhaloID"].values
    sort_idx = np.argsort(subhalo_id)
    new_mah = new_mah[sort_idx]
    new_dmah = new_dmah[sort_idx]
    new_cat = new_cat.sort_values(by="SubhaloID")
    new_dcat = new_dcat.iloc[sort_idx]

    # last few checks
    flen = len(new_cat)
    assert flen == new_mah.shape[0] == new_dmah.shape[0] == len(new_dcat)
    assert sum(new_cat["dmo_match"].values == MISSING) == 0
    assert sum(new_dcat["hydro_match"].values == MISSING) == 0
    assert np.all(new_cat["dmo_match"].values == new_dcat["dmo_id"].values)
    assert np.all(new_cat["hydro_id"].values == new_dcat["hydro_match"].values)

    return new_cat, new_dcat, new_mah, new_dmah


def hydro_dm_match_pipeline(
    present_snapfile: str,
    present_dark_snapfile: str,
    trees_file: str,
    dark_trees_file: str,
    metadata_file: str,
    color_file: str,
    mass_bin=(12.8, 13.1),
    mbin_fudge=0.3,
):
    """Full pipeline that matches hydro and DMO catalogs, along with their corresponding MAHs."""
    cat = pd.read_hdf(present_snapfile)
    dcat = pd.read_hdf(present_dark_snapfile)

    cat, dcat = match_dm_and_hydro_cat(cat, dcat)

    assert sum(cat["dmo_match"] != MISSING) == sum(dcat["hydro_match"] != MISSING)

    mah_info = setup_mah_from_trees(trees_file, metadata_file, SNAPS, mass_bin=mass_bin)

    dmass_bin = (
        mass_bin[0] - mbin_fudge,
        mass_bin[1] + mbin_fudge,
    )  # wider, to avoid edge effects
    dmah_info = setup_mah_from_trees(
        dark_trees_file, metadata_file, SNAPS, mass_bin=dmass_bin
    )

    mah, dmah = mah_info["mah"], dmah_info["mah"]

    halo_idx = mah_info["halo_idx"]
    dhalo_idx = dmah_info["halo_idx"]

    snaps, zs, scales = mah_info["snaps"], mah_info["z"], mah_info["scales"]
    dsnaps, dzs, dscales = dmah_info["snaps"], dmah_info["z"], dmah_info["scales"]

    assert np.all(snaps == dsnaps)
    assert np.all(zs == dzs)
    assert np.all(scales == dscales)

    cat = match_mah_and_cat(halo_idx, cat)
    dcat = match_mah_and_cat(dhalo_idx, dcat)

    # finally rematch the hydro and dm catalogs from the columns we created.
    cat, dcat, mah, dmah = rematch_dm_and_hydro_cat(cat, dcat, mah, dmah)

    # get additional properties we need for analysis
    m_peak, dm_peak = get_mpeak_from_mah(mah), get_mpeak_from_mah(dmah)

    cat["vmax/vvir"] = get_vmax_over_vvir(cat)
    dcat["vmax/vvir"] = get_vmax_over_vvir(dcat)

    # get defragmented frames
    cat = cat.copy()
    dcat = dcat.copy()

    # final group mass
    gmass = mah[:, -1]
    dgmass = dmah[:, -1]

    # get other interesting hydro properties
    msmhmr, _ = get_msmhmr(cat, gmass)
    df_color = get_color(color_file, cat)
    gr = (df_color["sdss_g"] - df_color["sdss_r"]).values
    cat["msmhmr"] = msmhmr
    cat["g-r"] = gr

    return {
        "cat": cat,
        "dcat": dcat,
        "mah": mah,
        "dmah": dmah,
        "m_peak": m_peak,
        "dm_peak": dm_peak,
        "gmass": gmass,
        "dgmass": dgmass,
        "snaps": snaps,
        "scales": scales,
    }
