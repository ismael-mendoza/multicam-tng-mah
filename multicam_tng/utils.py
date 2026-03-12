"""Several utility functions to read data from TNG catalogs."""

import pickle

import numpy as np
import pandas as pd
from halotools.mock_observables import return_xyz_formatted_array
from numpy.random import default_rng
from tqdm import tqdm

SNAPS = np.arange(0, 100, 1)
TNG_H = 0.6774  # from website

MISSING = -999999999


def generate_randoms(*, lbox, n_randoms, rng=None):
    """Generate randoms to be used for tpcf_jackknife
    Args:
        Lbox: size of TNG simulation box (usually Mpc/h)
        n_randoms: desired number of randoms
    Return:
        Nran: xyz formatted array of uniform random variables
    """
    if rng is None:
        _rng = default_rng(42)
    elif isinstance(rng, int):
        _rng = default_rng(rng)
    else:
        _rng = rng

    Xran = _rng.uniform(0, lbox, n_randoms)
    Yran = _rng.uniform(0, lbox, n_randoms)
    Zran = _rng.uniform(0, lbox, n_randoms)
    Nran = return_xyz_formatted_array(Xran, Yran, Zran)
    return Nran


def get_mah_interpolated(mvir: np.ndarray):
    assert mvir.shape[1] == 100
    assert mvir.ndim == 2

    n_snaps = 100
    n_haloes = mvir.shape[0]
    mah = np.zeros((n_haloes, n_snaps)) * np.nan

    for ii in tqdm(range(n_haloes)):
        mdm = mvir[ii, :]
        _mask = mdm == 0
        mdm[_mask] = np.nan

        # linearly interpolate nan values
        mdm = pd.Series(mdm)
        mdm = mdm.interpolate(method="linear", limit_direction="both", axis=0)
        mah[ii] = mdm.values

    return mah


def get_mpeak_from_mah(mah: np.ndarray):
    """Compute m_peak from (log10) mah."""
    assert mah.ndim == 2
    _m_peak = np.fmax.accumulate(10**mah, axis=1)
    m_peak = _m_peak / _m_peak[:, -1][:, None]
    return m_peak


def get_an_from_mpeak_mah(mpeak_mah: np.ndarray, *, scales: np.ndarray, ns: np.ndarray):
    # get a_{n} (correctly with interpolation)
    # mpeak_mah is the normalized mpeak, so that it monotonically increases and values all [0, 1]
    assert scales.shape[0] == mpeak_mah.shape[1]
    assert scales.ndim == 1 and mpeak_mah.ndim == 2
    assert ns.ndim == 1
    n_haloes = mpeak_mah.shape[0]
    an = np.zeros((n_haloes, len(ns)))
    for ii in range(n_haloes):
        mah_ii = mpeak_mah[ii]
        an[ii] = np.interp(ns, mah_ii, scales)
    return an


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


def get_vmax_over_vvir(cat: pd.DataFrame):
    """Compute vmax / vvir from catalog columns."""
    # compute vvir and create new column

    # ensure units of mvir is in units of Msun / h
    mvir = 10 ** cat["Mvir"].values / TNG_H  # og units: log10(msun)
    rvir = cat["Rvir"].values / TNG_H  # og units: kpc
    vvir = get_vvir(rvir, mvir)

    return cat["Vmax_DM"] / vvir


def get_msmhmr(
    mstar: np.ndarray,
    mvir: np.ndarray,
    *,
    mass_bin: tuple[float, float],
    n_bins: int = 11,
):
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


def _reverse_trees(trees):
    """Reverse each entry in trees so that order is from early to late times."""
    for tree in trees:
        for key in tree.keys():  # noqa
            if key not in ["Number", "ChunkNumber", "TreeID"]:
                tree[key] = tree[key][::-1]
    return trees


def read_trees(trees_file: str):
    """Read in the trees file and convert masses to log10(M/Msun)."""
    with open(trees_file, "rb") as pickle_file:
        _trees = pickle.load(pickle_file)
        trees = _reverse_trees(_trees)
        for tree in trees:
            for k in tree.keys():  # noqa
                if "Mass" in k or "_M_" in k:
                    tree[k] = convert_tng_mass(tree[k])
    return trees
