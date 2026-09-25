"""Shared fixtures for the pyMacroIO test suite."""

from __future__ import annotations

import numpy as np
import pytest


def _make_data_dict() -> dict:
    """Return a minimal self-consistent 3-sector IO data dictionary."""
    Z0 = np.array(
        [
            [ 0., 20.,  5.],
            [ 3.,  0., 15.],
            [ 8.,  2.,  0.],
        ],
        dtype=np.float64,
    )

    cons  = np.array([[15.], [10.], [14.]], dtype=np.float64)
    gov   = np.array([[ 8.], [ 5.], [ 7.]], dtype=np.float64)
    inv   = np.array([[ 5.], [ 3.], [ 5.]], dtype=np.float64)
    invnt = np.array([[ 4.], [ 2.], [ 2.]], dtype=np.float64)
    exp   = np.array([[ 3.], [ 2.], [ 2.]], dtype=np.float64)
    l0       = np.array([20.,  7.,  6.], dtype=np.float64)
    cap0     = np.array([15.,  4.,  5.], dtype=np.float64)
    tax0     = np.array([ 5.,  2.,  2.], dtype=np.float64)
    imp0     = np.array([ 4.,  3.,  5.], dtype=np.float64)
    profits0 = np.array([5., 2., 2.], dtype=np.float64)

    return {
        "sector_labels":        ["A", "B", "C"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0
        },
    }


@pytest.fixture(scope="module")
def minimal_data_dict():
    """Return the minimal 3-sector IO data dictionary."""
    return _make_data_dict()


@pytest.fixture(scope="module")
def minimal_model(minimal_data_dict):
    """Return an InputOutputModel built from the minimal 3-sector data dictionary."""
    from pyMacroIO.model import InputOutputModel
    from pyMacroIO.config import ModelConfig

    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


def _make_inactive_sector_data_dict() -> dict:
    """Return the minimal dictionary with a fourth sector carrying no flows and no output."""
    d = _make_data_dict()
    Z0 = np.zeros((4, 4), dtype=np.float64)
    Z0[:3, :3] = d["Z0"]
    d["sector_labels"] = d["sector_labels"] + ["D"]
    d["Z0"] = Z0
    for key in ("cons_vec", "gov_vec", "inv_vec", "invnt_vec", "exp_vec"):
        d[key] = np.vstack([d[key], np.zeros((1, d[key].shape[1]), dtype=np.float64)])
    for key in ("l0", "cap0", "tax0", "imp0", "profits0"):
        d[key] = np.append(d[key], 0.0)
    return d


@pytest.fixture(scope="module")
def inactive_sector_data_dict():
    """Return the 4-sector IO data dictionary whose fourth sector has zero gross output."""
    return _make_inactive_sector_data_dict()


def _make_klems_data_dict() -> dict:
    """Return a self-consistent 3-sector IO data dictionary for KLEMS tests."""
    Z0 = np.array(
        [
            [0., 8., 4.],
            [3., 0., 6.],
            [2., 1., 0.],
        ],
        dtype=np.float64,
    )

    cons  = np.array([[10.], [ 5.], [10.]], dtype=np.float64)
    gov   = np.array([[ 4.], [ 3.], [ 4.]], dtype=np.float64)
    inv   = np.array([[ 2.], [ 2.], [ 2.]], dtype=np.float64)
    invnt = np.array([[ 1.], [ 1.], [ 1.]], dtype=np.float64)
    exp   = np.array([[ 1.], [ 0.], [ 0.]], dtype=np.float64)

    l0       = np.array([5., 3., 6.], dtype=np.float64)
    cap0     = np.array([3., 2., 1.], dtype=np.float64)
    tax0     = np.array([1., 1., 1.], dtype=np.float64)
    imp0     = np.array([2., 1., 1.], dtype=np.float64)
    profits0 = np.array([14., 4., 1.], dtype=np.float64)

    # l0_by_skill: (3 skill tiers, 3 sectors); column sums must equal l0.
    l0_by_skill = np.array(
        [
            [2., 1., 2.],
            [2., 1., 3.],
            [1., 1., 1.],
        ],
        dtype=np.float64,
    )

    return {
        "sector_labels":        ["electricity", "steel", "services"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0,
        },
        "l0_by_skill":          l0_by_skill,
    }


@pytest.fixture(scope="module")
def klems_data_dict():
    """Return the KLEMS 3-sector IO data dictionary."""
    return _make_klems_data_dict()


def _make_two_region_data_dict() -> dict:
    """Return a 3-sector IO data dictionary with two regions for multi-region tests."""
    Z0 = np.array(
        [[0., 20.,  5.],
         [3.,  0., 15.],
         [8.,  2.,  0.]],
        dtype=np.float64,
    )

    # row sums = [15, 10, 14]; region 0 consumes A and B, region 1 consumes C.
    cons  = np.array([[15.,  0.], [10.,  0.], [ 0., 14.]], dtype=np.float64)
    gov   = np.array([[ 8.,  0.], [ 5.,  0.], [ 0.,  7.]], dtype=np.float64)
    inv   = np.array([[ 5.,  0.], [ 3.,  0.], [ 0.,  5.]], dtype=np.float64)
    invnt = np.array([[ 4.,  0.], [ 2.,  0.], [ 0.,  2.]], dtype=np.float64)
    exp   = np.array([[ 3.,  0.], [ 2.,  0.], [ 0.,  2.]], dtype=np.float64)

    l0       = np.array([20.,  7.,  6.], dtype=np.float64)
    cap0     = np.array([15.,  4.,  5.], dtype=np.float64)
    tax0     = np.array([ 5.,  2.,  2.], dtype=np.float64)
    imp0     = np.array([ 4.,  3.,  5.], dtype=np.float64)
    profits0 = np.array([ 5.,  2.,  2.], dtype=np.float64)

    return {
        "sector_labels":        ["A", "B", "C"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0
        },
        "region_map":           np.array([0, 0, 1], dtype=np.int32),
    }


@pytest.fixture(scope="module")
def two_region_data_dict():
    """Return the two-region 3-sector IO data dictionary."""
    return _make_two_region_data_dict()


def _make_bilateral_data_dict() -> dict:
    """Return a two-region, four-sector dictionary with cross-border trade in one good."""
    # R0:gas has no government, investment or inventory demand, only intermediate and household.
    Z0 = np.array(
        [[0., 10., 0., 12.],
         [5.,  0., 0.,  3.],
         [0.,  0., 0.,  5.],
         [0.,  4., 3.,  0.]],
        dtype=np.float64,
    )

    cons  = np.array([[12., 18.], [20., 5.], [0., 10.], [ 2., 15.]], dtype=np.float64)
    gov   = np.array([[ 0.,  0.], [ 5., 2.], [0.,  2.], [ 0.,  3.]], dtype=np.float64)
    inv   = np.array([[ 0.,  0.], [ 4., 2.], [0.,  1.], [ 0.,  2.]], dtype=np.float64)
    invnt = np.array([[ 0.,  0.], [ 2., 0.], [0.,  1.], [ 0.,  1.]], dtype=np.float64)
    exp   = np.zeros((4, 2), dtype=np.float64)

    l0       = np.array([20., 10., 6., 4.], dtype=np.float64)
    cap0     = np.array([10.,  5., 3., 2.], dtype=np.float64)
    tax0     = np.array([ 5.,  3., 2., 1.], dtype=np.float64)
    imp0     = np.array([ 2.,  3., 1., 2.], dtype=np.float64)
    profits0 = np.array([10., 13., 4., 1.], dtype=np.float64)

    return {
        "sector_labels":        ["R0:gas", "R0:mfg", "R1:gas", "R1:mfg"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0
        },
        "region_map":           np.array([0, 0, 1, 1], dtype=np.int32),
        "region_labels":        ["R0", "R1"],
    }


@pytest.fixture(scope="module")
def bilateral_data_dict():
    """Return the two-region four-sector dictionary used for bilateral delivery shocks."""
    return _make_bilateral_data_dict()


def _make_skill_data_dict() -> dict:
    """Return the standard 3-sector IO data dictionary augmented with l0_by_skill."""
    Z0 = np.array(
        [
            [ 0., 20.,  5.],
            [ 3.,  0., 15.],
            [ 8.,  2.,  0.],
        ],
        dtype=np.float64,
    )

    cons  = np.array([[15.], [10.], [14.]], dtype=np.float64)
    gov   = np.array([[ 8.], [ 5.], [ 7.]], dtype=np.float64)
    inv   = np.array([[ 5.], [ 3.], [ 5.]], dtype=np.float64)
    invnt = np.array([[ 4.], [ 2.], [ 2.]], dtype=np.float64)
    exp   = np.array([[ 3.], [ 2.], [ 2.]], dtype=np.float64)

    l0       = np.array([20.,  7.,  6.], dtype=np.float64)
    cap0     = np.array([15.,  4.,  5.], dtype=np.float64)
    tax0     = np.array([ 5.,  2.,  2.], dtype=np.float64)
    imp0     = np.array([ 4.,  3.,  5.], dtype=np.float64)
    profits0 = np.array([ 5.,  2.,  2.], dtype=np.float64)

    l0_by_skill = np.array(
        [
            [10., 3., 3.],
            [ 7., 2., 2.],
            [ 3., 2., 1.],
        ],
        dtype=np.float64,
    )

    return {
        "sector_labels":        ["A", "B", "C"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0
        },
        "l0_by_skill":          l0_by_skill,
    }


@pytest.fixture(scope="module")
def skill_data_dict():
    """Return the 3-sector IO data dictionary augmented with l0_by_skill."""
    return _make_skill_data_dict()


def _make_row_data_dict() -> dict:
    """Return a 4-sector, 2-region IO data dict with a RoW region matching domestic sectors."""
    Z0 = np.array(
        [
            [0., 5., 0., 0.],
            [3., 0., 0., 0.],
            [0., 0., 0., 3.],
            [0., 0., 2., 0.],
        ],
        dtype=np.float64,
    )

    l0       = np.array([10., 5., 8., 4.], dtype=np.float64)
    cap0     = np.array([ 4., 2., 4., 2.], dtype=np.float64)
    tax0     = np.array([ 2., 1., 1., 1.], dtype=np.float64)
    imp0     = np.array([ 2., 1., 1., 1.], dtype=np.float64)
    profits0 = np.array([ 2., 1., 1., 0.], dtype=np.float64)

    cons  = np.array([[10., 0.], [7., 0.], [0., 9.], [0., 5.]], dtype=np.float64)
    gov   = np.array([[ 4., 0.], [2., 0.], [0., 2.], [0., 2.]], dtype=np.float64)
    inv   = np.array([[ 2., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float64)
    invnt = np.array([[ 1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float64)
    exp   = np.array([[ 1., 0.], [1., 0.], [0., 1.], [0., 0.]], dtype=np.float64)

    return {
        "sector_labels":        ["A", "B", "RoW:A", "RoW:B"],
        "Z0":                   Z0,
        "l0":                   l0,
        "cap0":                 cap0,
        "tax0":                 tax0,
        "imp0":                 imp0,
        "profits0":             profits0,
        "cons_vec":             cons,
        "gov_vec":              gov,
        "inv_vec":              inv,
        "invnt_vec":            invnt,
        "exp_vec":              exp,
        "consumer_taxes_total": 0.0,
        "fd_imports_totals":    {
            "cons": 0.0, "gov": 0.0, "inv": 0.0, "invnt": 0.0, "exp": 0.0,
        },
        "region_map":           np.array([0, 0, 1, 1], dtype=np.int32),
        "region_labels":        ["domestic", "RoW"],
    }


@pytest.fixture(scope="module")
def row_data_dict():
    """Return the 4-sector, 2-region IO data dictionary with an explicit RoW region."""
    return _make_row_data_dict()
