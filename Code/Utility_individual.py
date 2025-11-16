
import random
import requests
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import entropy, wasserstein_distance, truncnorm
from scipy.spatial.distance import jensenshannon


import geopandas as gpd
from scipy.stats import lognorm
from shapely.geometry import Point

api_key = "0291ed453c49b88aafdb6d6d12bac47801f0e6f5"
state   = "36"
county  = "061"


# Population Marginals Reading
tags_indiv = [
    # Age categories (total)
    *[f"S0101_C01_{str(i).zfill(3)}E" for i in range(2, 20)],
    
    # Gender totals
    "S0101_C03_001E",  # Male total
    "S0101_C05_001E",  # Female total
    
    # Employment status (usual hours worked)
    "S2303_C01_008E",  # Did not work
    "S2303_C01_009E",  # Usually worked 35 or more hours
    "S2303_C01_016E",  # Usually worked 15 to 34 hours
    "S2303_C01_023E",  # Usually worked 1 to 14 hours
    
    # Education attainment (25+)
    *[f"S1501_C01_{str(j).zfill(3)}E" for j in range(7, 15)]
]

def get_section(tag):
    if tag.startswith("B"):
        return ""
    elif tag.startswith("S"):
        return "/subject"
    elif tag.startswith("DP"):
        return "/profile"
    return ""

def fetch_individual_marginals(state, county, api_key):
    dfs = {}
    for tag in tags_indiv:
        sec = get_section(tag)
        url = (
            f"https://api.census.gov/data/2023/acs/acs5{sec}"
            f"?get=NAME,{tag}&for=county:{county}&in=state:{state}&key={api_key}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        dfs[tag] = pd.DataFrame(data[1:], columns=data[0])
    return dfs

def aggregate_individual_marginals(dfs):
    dist = {}
    
    # Age distribution
    age_tags = [f"S0101_C01_{str(i).zfill(3)}E" for i in range(2, 20)]
    age_vals = [int(dfs[t].iloc[0][t]) for t in age_tags]
    dist['age_dist'] = {i+1: age_vals[i] for i in range(len(age_vals))}
    
    # Gender distribution
    male = int(dfs["S0101_C03_001E"].iloc[0]["S0101_C03_001E"])
    female = int(dfs["S0101_C05_001E"].iloc[0]["S0101_C05_001E"])
    dist['gender_dist'] = {'male': male, 'female': female}
    
    # Employment distribution (hours worked categories)
    emp_tags = ["S2303_C01_008E","S2303_C01_009E","S2303_C01_016E","S2303_C01_023E"]
    emp_vals = [int(dfs[t].iloc[0][t]) for t in emp_tags]
    dist['employment_dist'] = {
        'none': emp_vals[0],
        '35+_hrs': emp_vals[1],
        '15-34_hrs': emp_vals[2],
        '1-14_hrs': emp_vals[3]
    }
    
    # Education distribution (25+)
    edu_tags = [f"S1501_C01_{str(j).zfill(3)}E" for j in range(7, 15)]
    edu_vals = [int(dfs[t].iloc[0][t]) for t in edu_tags]
    dist['education_dist'] = {
        'less_9th': edu_vals[0],
        '9-12_no_diploma': edu_vals[1],
        'high_school': edu_vals[2],
        'some_college': edu_vals[3],
        'associate': edu_vals[4],
        'bachelor': edu_vals[5],
        'graduate': edu_vals[6]
    }
    
    return dist

# IPF variant

age_bin_labels = [
    "Under 5 years","5 to 9 years","10 to 14 years","15 to 19 years",
    "20 to 24 years","25 to 29 years","30 to 34 years","35 to 39 years",
    "40 to 44 years","45 to 49 years","50 to 54 years","55 to 59 years",
    "60 to 64 years","65 to 69 years","70 to 74 years","75 to 79 years",
    "80 to 84 years","85 years and over"
]
age_bin_ranges = {
    "Under 5 years":      (0,4),    "5 to 9 years":      (5,9),
    "10 to 14 years":     (10,14),  "15 to 19 years":    (15,19),
    "20 to 24 years":     (20,24),  "25 to 29 years":    (25,29),
    "30 to 34 years":     (30,34),  "35 to 39 years":    (35,39),
    "40 to 44 years":     (40,44),  "45 to 49 years":    (45,49),
    "50 to 54 years":     (50,54),  "55 to 59 years":    (55,59),
    "60 to 64 years":     (60,64),  "65 to 69 years":    (65,69),
    "70 to 74 years":     (70,74),  "75 to 79 years":    (75,79),
    "80 to 84 years":     (80,84),  "85 years and over": (85,100),
}

def ipf(seed, target_row, target_col, max_iter=100, tol=1e-6):
    """Fit seed to match target_row and target_col via RAS/IPF, preserving zeros."""
    X = seed.astype(float)
    for _ in range(max_iter):
        # Row scaling
        row_sums = X.sum(axis=1)
        X *= (target_row / row_sums)[:, None]
        # Column scaling
        col_sums = X.sum(axis=0)
        X *= (target_col / col_sums)[None, :]
        # Check convergence
        if (np.allclose(X.sum(axis=1), target_row, atol=tol) and
            np.allclose(X.sum(axis=0), target_col, atol=tol)):
            break
    return X

def integerize_table(fitted, total_count):
    """Convert real-valued fitted table into integer counts summing to total_count."""
    floors = np.floor(fitted)
    residuals = fitted - floors
    current_total = int(floors.sum())
    diff = int(total_count - current_total)
    # Assign extras to cells with largest residuals
    idxs = np.argsort(residuals.flatten())[::-1][:diff]
    ints = floors.flatten()
    ints[idxs] += 1
    return ints.reshape(fitted.shape).astype(int)

def integerize_by_row(fitted, target_row):
    """
    For each row i, round fitted[i,:] to integers summing exactly to target_row[i]
    via floor + largest‐residuals.
    """
    ints = np.zeros_like(fitted, dtype=int)
    for i, (row, tr) in enumerate(zip(fitted, target_row)):
        floors    = np.floor(row)
        residuals = row - floors
        diff      = int(tr - floors.sum())
        # pick the `diff` cells with largest residuals
        idx_top   = np.argsort(residuals)[::-1][:diff]
        row_int   = floors.astype(int)
        row_int[idx_top] += 1
        ints[i, :] = row_int
    return ints



def generate_ages_ipf_constrained(df_hh, acs_age_dist, age_bin_labels, age_bin_ranges, seed=42):
    """
    Role × age-bin IPF with structural zeros:
    - Only 'Child' role can occupy bins 1-4 (under 18).
    - Only non-child roles can occupy bins 5-18 (18+).
    - In bin '15 to 19 years', children are capped at age 17.
    """
    np.random.seed(seed)
    
    # 1) Expand households into individuals
    records = []
    for hh_id, r in df_hh.iterrows():
        records.append({'household_id': hh_id, 'relationship': 'Head'})
        records += [{'household_id': hh_id, 'relationship': 'Spouse'}] * int(r.spouse_cnt)
        records += [{'household_id': hh_id, 'relationship': 'Child'}] * int(r.child_cnt)
        records += [{'household_id': hh_id, 'relationship': 'OtherRelative'}] * int(r.other_rel_cnt)
        records += [{'household_id': hh_id, 'relationship': 'OtherNonRelative'}] * int(r.other_nonrel_cnt)
    df_ind = pd.DataFrame(records)
    total_ind = len(df_ind)
    
    # 2) Build IPF targets
    roles = ['Head','Spouse','Child','OtherRelative','OtherNonRelative']
    bins = sorted(acs_age_dist.keys())
    
    target_role = np.array([(df_ind.relationship == r).sum() for r in roles], float)
    # Rescale ACS age counts to synthetic total
    raw_counts = np.array([acs_age_dist[b] for b in bins], float)
    props = raw_counts / raw_counts.sum()
    target_bin = props * total_ind
    
    # 3) Seed with structural zeros
    seed_mat = np.zeros((len(roles), len(bins)))
    for i, r in enumerate(roles):
        for j, b in enumerate(bins):
            if r == 'Child' and b <= 4:
                seed_mat[i,j] = 1
            if r != 'Child' and b >= 5:
                seed_mat[i,j] = 1
    

    # 4) IPF + integerize
    fitted = ipf(seed_mat, target_role, target_bin)
    actual_role_counts = np.array([
        (df_ind.relationship == r).sum() for r in roles
    ], float)

    # Compute current row‐sums of the fitted table
    row_sums = fitted.sum(axis=1)

    # Rescale each row so row i sums to actual_role_counts[i]
    # (preserves zeros in seed_mat)
    fitted = fitted * (actual_role_counts / row_sums)[:, None]

    # 5) Integerize by row
    counts = integerize_by_row(fitted, actual_role_counts)

    # 5) Sample ages per role-bin cell
    roles = ['Head','Spouse','Child','OtherRelative','OtherNonRelative']
    bins  = sorted(acs_age_dist.keys())

    # 1) Sample ages per (role, bin) into separate buckets
    role_ages = { role: [] for role in roles }

    for i, role in enumerate(roles):
        for j, b in enumerate(bins):
            n = counts[i, j]
            if n <= 0:
                continue
            lo, hi = age_bin_ranges[age_bin_labels[j]]
            if role == 'Child' and age_bin_labels[j] == "15 to 19 years":
                hi = 17
            # draw `n` ages in [lo, hi]
            samples = np.random.randint(lo, hi+1, size=n)
            role_ages[role].extend(samples)

    # 2) Shuffle _within_ each role
    for role in roles:
        random.shuffle(role_ages[role])

    # 3) Assign back to df_ind by relationship
    df_ind['age'] = None
    for role in roles:
        mask = (df_ind.relationship == role)
        # get the right number of ages
        df_ind.loc[mask, 'age'] = role_ages[role]

    return df_ind


# other individual attributes via IPF

def ipf(seed, target_row, target_col, max_iter=100, tol=1e-6):
    """RAS/IPF fitting."""
    X = seed.astype(float)
    for _ in range(max_iter):
        # scale rows
        X *= (target_row / X.sum(axis=1))[:, None]
        # scale cols
        X *= (target_col / X.sum(axis=0))[None, :]
        if (np.allclose(X.sum(axis=1), target_row, atol=tol) and
            np.allclose(X.sum(axis=0), target_col, atol=tol)):
            break
    return X

def integerize_by_row(fitted, target_row):
    """Floor + largest-residual rounding per row to match target_row sums."""
    ints = np.zeros_like(fitted, dtype=int)
    for i, (row, tr) in enumerate(zip(fitted, target_row)):
        floors = np.floor(row)
        residuals = row - floors
        diff = int(tr - floors.sum())
        idxs = np.argsort(residuals)[::-1][:diff]
        base = floors.astype(int)
        base[idxs] += 1
        ints[i] = base
    return ints

def ipf_assign_by_bin(df, bins, categories, acs_dist, age_bin_labels, age_bin_ranges, seed=0):
    """Assign a categorical attribute via IPF on age-bin × categories."""
    np.random.seed(seed)
    total = len(df)
    # assign each row to a bin index
    bin_idx = []
    for age in df['age']:
        for i, label in enumerate(age_bin_labels, start=1):
            lo, hi = age_bin_ranges[label]
            if lo <= age <= hi:
                bin_idx.append(i)
                break
    df = df.copy()
    df['bin'] = bin_idx
    
    # row margins: individuals per bin
    target_row = np.array([ (df['bin']==b).sum() for b in bins ], float)
    # column margins: ACS dist scaled to total
    raw = np.array([ acs_dist[c] for c in categories ], float)
    props = raw / raw.sum()
    target_col = props * total
    
    # seed matrix (no structural zeros here; can add if needed)
    seed_mat = np.ones((len(bins), len(categories)))
    
    fitted = ipf(seed_mat, target_row, target_col)
    counts = integerize_by_row(fitted, target_row)
    
    # assign labels per bin
    assignment = np.empty(total, dtype=object)
    for i, b in enumerate(bins):
        mask = (df['bin'] == b).values
        labels = []
        for j, cat in enumerate(categories):
            labels += [cat] * counts[i, j]
        random.shuffle(labels)
        assignment[mask] = labels
    
    return assignment

def assign_demographics_ipf(df_ages, indiv_dists, age_bin_labels, age_bin_ranges, seed=42):
    """Assign gender, employment, and education via IPF and age constraints."""
    df = df_ages.copy()
    bins = sorted(indiv_dists['age_dist'].keys())
    
    # Gender
    gender_cats = list(indiv_dists['gender_dist'].keys())
    df['gender'] = ipf_assign_by_bin(df, bins, gender_cats,
                                     indiv_dists['gender_dist'],
                                     age_bin_labels, age_bin_ranges, seed)
    
    # Employment
    emp_cats = list(indiv_dists['employment_dist'].keys())
    df['employment'] = ipf_assign_by_bin(df, bins, emp_cats,
                                         indiv_dists['employment_dist'],
                                         age_bin_labels, age_bin_ranges, seed+1)
    df.loc[df['age'] < 16, 'employment'] = 'none'
    
    # Education
    edu_cats = list(indiv_dists['education_dist'].keys())
    df['education'] = ipf_assign_by_bin(df, bins, edu_cats,
                                        indiv_dists['education_dist'],
                                        age_bin_labels, age_bin_ranges, seed+2)
    df.loc[df['age'] < 25, 'education'] = 'Under25'
    
    return df



#Age Metrics Comparison


def compare_age_metrics(df_ages, acs_age_dist, age_bin_labels, age_bin_ranges):
    """
    Compare synthetic age distribution against ACS marginals.

    Parameters:
    - df_ages: DataFrame with columns ['household_id', 'relationship', 'age'] for all individuals.
    - acs_age_dist: dict mapping bin index (1-18) to ACS counts.
    - age_bin_labels: list of 18 bin labels in order.
    - age_bin_ranges: dict mapping bin label to (lo, hi) tuple.

    Returns:
    - dict with keys ['KL', 'JS', 'EMD', 'RMSE', 'MAPE'].
    """
    # ACS distribution Q
    bins = list(acs_age_dist.keys())  # [1, 2, ..., 18]
    Q_counts = np.array([acs_age_dist[b] for b in bins], dtype=float)
    Q = Q_counts / Q_counts.sum()
    
    # Synthetic distribution P
    P_counts = np.zeros_like(Q_counts)
    for idx, b in enumerate(bins):
        label = age_bin_labels[b-1]
        lo, hi = age_bin_ranges[label]
        P_counts[idx] = ((df_ages['age'] >= lo) & (df_ages['age'] <= hi)).sum()
    P = P_counts / P_counts.sum()
    
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    P_safe = P + eps
    Q_safe = Q + eps
    
    # KL divergence
    KL = np.sum(P_safe * np.log(P_safe / Q_safe))
    
    # Jensen-Shannon divergence
    M = 0.5 * (P + Q)
    M_safe = M + eps
    JS = 0.5 * (np.sum(P_safe * np.log(P_safe / M_safe)) +
                np.sum(Q_safe * np.log(Q_safe / M_safe)))
    
    # Earth Mover's Distance (Wasserstein-1D)
    # Use midpoints of each bin
    midpoints = np.array([(age_bin_ranges[label][0] + age_bin_ranges[label][1]) / 2 
                          for label in age_bin_labels])
    EMD = wasserstein_distance(midpoints, midpoints, u_weights=P, v_weights=Q)
    
    # Root Mean Squared Error
    RMSE = np.sqrt(np.mean((P - Q)**2))
    
    # Mean Absolute Percentage Error
    MAPE = 100.0 * np.mean(np.abs((P - Q) / Q_safe))
    
    return {
        'KL': KL,
        'JS': JS,
        'EMD': EMD,
        'RMSE': RMSE,
        'MAPE': MAPE
    }
