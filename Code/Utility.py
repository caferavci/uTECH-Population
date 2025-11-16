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
######################################################################################################################################
# 1. Read in ACS data for household characteristics

tags = [
    # Household-size & structure
    "DP02_0016E",  # avg household size
    "DP02_0017E",  # avg family size
    "S2501_C01_002E", "S2501_C01_003E", "S2501_C01_004E", "S2501_C01_005E",
    
    # Household-type by householder age
    "S2501_C01_010E", "S2501_C01_015E",
    "S2501_C01_019E", "S2501_C01_024E", "S2501_C01_028E",
    
    # Household-income distributions
    *[f"S1901_C01_{str(i).zfill(3)}E" for i in range(2, 12)],  # all households
    *[f"S1901_C02_{str(i).zfill(3)}E" for i in range(2, 12)],  # families
    *[f"S1901_C03_{str(i).zfill(3)}E" for i in range(2, 12)],  # married-couple
    *[f"S1901_C04_{str(i).zfill(3)}E" for i in range(2, 12)],  # nonfamily
    
    # Households with own children under 18
    "B11005_004E", "B11005_006E", "B11005_007E", "B11005_008E",
    
    # Relationship counts
    "DP02_0019E", "DP02_0020E", "DP02_0021E",
    "DP02_0022E", "DP02_0023E", "DP02_0024E",
    
    # Grandparent-grandchild households
    "B10063_002E", "B10063_003E", "B10063_004E", "B10063_005E",
    
    # Children-count by family type (B17023)
    *[f"B17023_{str(i).zfill(3)}E" for i in range(1, 36)]
]

def get_section(tag):
    if tag.startswith("B"):
        return ""
    elif tag.startswith("S"):
        return "/subject"
    elif tag.startswith("DP"):
        return "/profile"
    return ""

def fetch_census_data(state, county, api_key):
    """Fetch ACS data for specified tags."""
    dfs = {}
    for tag in tags:
        section = get_section(tag)
        url = (
            f"https://api.census.gov/data/2023/acs/acs5{section}"
            f"?get=NAME,{tag}&for=county:{county}&in=state:{state}&key={api_key}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        dfs[tag] = pd.DataFrame(data[1:], columns=data[0])
    return dfs

def aggregate_household_data(dfs):
    """Aggregate and return distributions needed for household IPF."""
    dist = {}
    
    # Averages
    dist['avg_household_size'] = float(dfs['DP02_0016E'].iloc[0]['DP02_0016E'])
    dist['avg_family_size']    = float(dfs['DP02_0017E'].iloc[0]['DP02_0017E'])
    
    # Household size distribution
    size_keys = ['1','2','3','4+']
    size_tags = ["S2501_C01_002E","S2501_C01_003E",
                 "S2501_C01_004E","S2501_C01_005E"]
    dist['household_size_dist'] = {
        size_keys[i]: int(dfs[tag].iloc[0][tag])
        for i, tag in enumerate(size_tags)
    }
    
    # Household type distribution
    type_map = {
        'family_married': "S2501_C01_010E",
        'family_other_male': "S2501_C01_015E",
        'family_other_female': "S2501_C01_019E",
        'nonfam_alone': "S2501_C01_024E",
        'nonfam_not_alone': "S2501_C01_028E",
    }
    dist['household_type_dist'] = {
        k: int(dfs[tag].iloc[0][tag])
        for k, tag in type_map.items()
    }
    
    # Income distributions
    def _agg_income(prefix):
        tags_ = [f"{prefix}_{str(i).zfill(3)}E" for i in range(2,12)]
        return [float(dfs[t].iloc[0][t]) for t in tags_]
    
    dist['income_all'] = _agg_income("S1901_C01")
    dist['income_families'] = _agg_income("S1901_C02")
    dist['income_married'] = _agg_income("S1901_C03")
    dist['income_nonfamily'] = _agg_income("S1901_C04")
    
    # Households with children flags
    dist['hh_with_child'] = {
        'married': int(dfs['B11005_004E'].iloc[0]['B11005_004E']),
        'other_male': int(dfs['B11005_006E'].iloc[0]['B11005_006E']),
        'other_female': int(dfs['B11005_007E'].iloc[0]['B11005_007E']),
        'nonfamily': int(dfs['B11005_008E'].iloc[0]['B11005_008E']),
    }
    
    # Relationship counts
    rel_map = {
        'householder': "DP02_0019E",
        'spouse': "DP02_0020E",
        'partner': "DP02_0021E",
        'child': "DP02_0022E",
        'other_rel': "DP02_0023E",
        'other_nonrel': "DP02_0024E",
    }
    dist['relationship_counts'] = {
        k: int(dfs[tag].iloc[0][tag])
        for k, tag in rel_map.items()
    }
    
    # Grandparent-grandchild households
    gp_map = {
        'with_gp': "B10063_002E",
        'gp_responsible': "B10063_003E",
        'gp_no_parent': "B10063_004E",
        'other_gp': "B10063_005E",
    }
    dist['gp_households'] = {
        k: int(dfs[tag].iloc[0][tag])
        for k, tag in gp_map.items()
    }
    
    # Children-count by family type (use B17023 series)
    # You can map specific B17023 tags to categories as needed here.
    dist['children_count_by_family'] = {
        tag: int(dfs[tag].iloc[0][tag])
        for tag in dfs if tag.startswith("B17023")
    }
    
    return dist


def build_children_distribution(dist):
    """
    From dist['children_count_by_family'] (raw B17023 counts),
    build a nested dict children_dist[family_type][bracket] = count
    """
    # Mapping of B17023 tags to (family_type, bracket)
    mapping = {
      # Below poverty
      'B17023_004E': ('family_married','0'),
      'B17023_005E': ('family_married','1-2'),
      'B17023_006E': ('family_married','3-4'),
      'B17023_007E': ('family_married','5+'),
      'B17023_009E': ('family_other_male','0'),
      'B17023_010E': ('family_other_male','1-2'),
      'B17023_011E': ('family_other_male','3-4'),
      'B17023_012E': ('family_other_male','5+'),
      'B17023_014E': ('family_other_female','0'),
      'B17023_015E': ('family_other_female','1-2'),
      'B17023_016E': ('family_other_female','3-4'),
      'B17023_017E': ('family_other_female','5+'),
      # Above poverty
      'B17023_021E': ('family_married','0'),
      'B17023_022E': ('family_married','1-2'),
      'B17023_023E': ('family_married','3-4'),
      'B17023_024E': ('family_married','5+'),
      'B17023_026E': ('family_other_male','0'),
      'B17023_027E': ('family_other_male','1-2'),
      'B17023_028E': ('family_other_male','3-4'),
      'B17023_029E': ('family_other_male','5+'),
      'B17023_031E': ('family_other_female','0'),
      'B17023_032E': ('family_other_female','1-2'),
      'B17023_033E': ('family_other_female','3-4'),
      'B17023_034E': ('family_other_female','5+'),
    }
    children = {
        'family_married':    {},
        'family_other_male': {},
        'family_other_female': {}
    }
    for tag, ct in dist['children_count_by_family'].items():
        if tag not in mapping: continue
        ftype, bracket = mapping[tag]
        children[ftype][bracket] = children[ftype].get(bracket,0) + ct
    return children

######################################################################################################################################
# 2. Household generation via IPF

# Household Generation
def sample_bracket(bracket):
    """Turn a bracket string into an integer count."""
    if bracket == '0':
        return 0
    if bracket == '1-2':
        return random.choice([1,2])
    if bracket == '3-4':
        return random.choice([3,4])
    # '5+'
    return random.randint(5,6)

def ipf(seed, target_row, target_col, max_iter=100, tol=1e-6):
    """Fit seed to match target_row and target_col via RAS/IPF."""
    X = seed.astype(float)
    for _ in range(max_iter):
        # 1) Row scaling
        row_sums = X.sum(axis=1)
        X *= (target_row / row_sums)[:, None]
        # 2) Col scaling
        col_sums = X.sum(axis=0)
        X *= (target_col / col_sums)[None, :]
        # convergence?
        if (np.allclose(X.sum(axis=1), target_row, atol=tol) and
            np.allclose(X.sum(axis=0), target_col, atol=tol)):
            break
    return X

def integerize_table(fitted, total_count):
    """
    Turn real‐valued fitted table into integers summing to total_count
    via floor + largest‐residuals rounding.
    """
    floors = np.floor(fitted)
    residuals = fitted - floors
    current_total = int(floors.sum())
    # how many to add
    diff = total_count - current_total
    # flatten residuals with indices
    idx_flat = np.argsort(residuals.flat)[::-1]
    to_add = idx_flat[:diff]
    ints = floors.flatten()
    ints[to_add] += 1
    return ints.reshape(fitted.shape).astype(int)

def generate_households_ipf(dist, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # 1) Prepare categories & targets
    size_cats = list(dist['household_size_dist'].keys())      # ['1','2','3','4+']
    type_cats = list(dist['household_type_dist'].keys())      # e.g. ['family_married',...]
    target_size = np.array([dist['household_size_dist'][s] for s in size_cats])
    target_type = np.array([dist['household_type_dist'][t] for t in type_cats])
    N = int(target_size.sum())

    # 2) Seed table & IPF (omitted here for brevity; assume you have `counts[i,j]`)
    # (a) start with all ones
    seed_table = np.ones((len(size_cats), len(type_cats)))

    # (b) find the index for the '1' bracket
    i1 = size_cats.index('1')

    # (c) zero‐out any one-person “family” types
    for j, t in enumerate(type_cats):
        if t.startswith('family_'):
            seed_table[i1, j] = 0

    # (d) also zero‐out one-person 'nonfam_not_alone'
    j_nonfam_na = type_cats.index('nonfam_not_alone')
    seed_table[i1, j_nonfam_na] = 0
    fitted = ipf(seed_table, target_size, target_type)

    counts = integerize_table(fitted, N)

    # 3) Compute Poisson lambda for '4+' bracket
    c1, c2, c3, c4p = [dist['household_size_dist'][b] for b in size_cats]
    avg_all = dist['avg_household_size']
    sum_lt4 = 1*c1 + 2*c2 + 3*c3
    m4plus = (avg_all * N - sum_lt4) / c4p
    lam = max(m4plus - 4, 0)

    # 4) Build children and income params
    child_dist = build_children_distribution(dist)
    nonfam_total = dist['household_type_dist']['nonfam_alone'] + dist['household_type_dist']['nonfam_not_alone']
    p_nonfam_child = dist['hh_with_child']['nonfamily'] / nonfam_total

    incomes = {
        'family_married':    dist['income_married'],
        'family_other_male': dist['income_families'],
        'family_other_female': dist['income_families'],
        'nonfam_alone':      dist['income_nonfamily'],
        'nonfam_not_alone':  dist['income_nonfamily'],
    }

    # 5) Expand cells and sample each household
    cells = []
    for i, s in enumerate(size_cats):
        for j, t in enumerate(type_cats):
            cells += [(s, t)] * counts[i, j]
    random.shuffle(cells)

    households = []
    for bracket, htype in cells:
        # resolve bracket → actual size
        if bracket != '4+':
            size = int(bracket)
        else:
            size = 4 + np.random.poisson(lam)

        # always 1 householder
        hh = 1
        # spouse only for married
        spouse = 1 if htype == 'family_married' else 0

        # sample children
        if htype.startswith('family_'):
            brackets, weights = zip(*child_dist[htype].items())
            cb = random.choices(brackets, weights=weights, k=1)[0]
            child_ct = sample_bracket(cb)
        else:
            child_ct = 1 if random.random() < p_nonfam_child else 0

        # —— clamp children so rem >= 0 —— 
        max_children = size - hh - spouse
        child_ct = max(0, min(child_ct, max_children))

        # remaining roles
        rem = size - (hh + spouse + child_ct)
        other_rel    = rem if htype.startswith('family_') else 0
        other_nonrel = rem if not htype.startswith('family_') else 0

        households.append({
            'size':             size,
            'type':             htype,
            'income_bracket':   random.choices(range(10), weights=incomes[htype], k=1)[0],
            'child_count':      child_ct,
            'householder_cnt':  hh,
            'spouse_cnt':       spouse,
            'child_cnt':        child_ct,
            'other_rel_cnt':    other_rel,
            'other_nonrel_cnt': other_nonrel,
        })

    return pd.DataFrame(households)


######################################################################################################################################
# 3. Household Validation

def compare_distributions(obs, gen):
    if isinstance(obs, dict):
        keys = sorted(obs.keys())
        obs_arr = np.array([obs[k] for k in keys], dtype=float)
    else:
        obs_arr = np.array(obs, dtype=float)
        keys = list(range(len(obs_arr)))
    if isinstance(gen, dict):
        gen_arr = np.array([gen.get(k, 0) for k in keys], dtype=float)
    else:
        gen_arr = np.array(gen, dtype=float)

    obs_prob = obs_arr / obs_arr.sum()
    gen_prob = gen_arr / gen_arr.sum()

    kl = entropy(obs_prob, gen_prob)
    js = jensenshannon(obs_prob, gen_prob, base=2.0)
    positions = np.arange(len(obs_prob))
    emd = wasserstein_distance(positions, positions, u_weights=obs_prob, v_weights=gen_prob)
    rmse = np.sqrt(np.mean((gen_prob - obs_prob)**2))
    mask = obs_prob != 0
    mape = np.mean(np.abs((gen_prob[mask] - obs_prob[mask]) / obs_prob[mask]))

    return {'KL': kl, 'JS': js, 'EMD': emd, 'RMSE': rmse, 'MAPE': mape}

def group_size_counts(size_counts):
    grouped = {}
    for k, v in size_counts.items():
        # Convert key to integer for comparison
        size = int(k)
        key = '4+' if size >= 4 else str(size)
        grouped[key] = grouped.get(key, 0) + v
    return grouped


def evaluate_households(df_hh, dists):
    results = {}

    # 1) Household size
    obs_size = dists['household_size_dist']
    gen_size = df_hh['size'].value_counts().to_dict()
    gen_size_grouped = group_size_counts(gen_size)
    results['household_size'] = compare_distributions(obs_size, gen_size_grouped)

    # 2) Household type
    obs_type = dists['household_type_dist']
    gen_type = df_hh['type'].value_counts().to_dict()
    results['household_type'] = compare_distributions(obs_type, gen_type)

    # 3) Overall income
    obs_inc_all = {i: cnt for i, cnt in enumerate(dists['income_all'])}
    gen_inc_all = df_hh['income_bracket'].value_counts().to_dict()
    results['income_all'] = compare_distributions(obs_inc_all, gen_inc_all)

    # 4) Income by type
    for typ in obs_type:
        if typ == 'family_married':
            obs_list = dists['income_married']
        elif typ.startswith('family_other'):
            obs_list = dists['income_families']
        else:
            obs_list = dists['income_nonfamily']
        obs_inc = {i: obs_list[i] for i in range(len(obs_list))}
        gen_inc = df_hh[df_hh['type'] == typ]['income_bracket'].value_counts().to_dict()
        results[f'income_{typ}'] = compare_distributions(obs_inc, gen_inc)

    # 5) Households-with-child proportion by type
    child_flag_keys = {
        'family_married': 'married',
        'family_other_male': 'other_male',
        'family_other_female': 'other_female',
        'nonfam_alone': 'nonfamily',
        'nonfam_not_alone': 'nonfamily'
    }
    for typ, obs_key in child_flag_keys.items():
        total = dists['household_type_dist'][typ]
        obs_child = dists['hh_with_child'].get(obs_key, 0)
        obs_binary = {0: total - obs_child, 1: obs_child}
        sub = df_hh[df_hh['type'] == typ]
        gen_child = sub['child_count'].apply(lambda x: 1 if x > 0 else 0).value_counts().to_dict()
        results[f'has_child_{typ}'] = compare_distributions(obs_binary, gen_child)

    # 6) Child-count bracket by family type
    mapping = {
        'B17023_004E':('family_married','0'), 'B17023_005E':('family_married','1-2'),
        'B17023_006E':('family_married','3-4'), 'B17023_007E':('family_married','5+'),
        'B17023_009E':('family_other_male','0'), 'B17023_010E':('family_other_male','1-2'),
        'B17023_011E':('family_other_male','3-4'), 'B17023_012E':('family_other_male','5+'),
        'B17023_014E':('family_other_female','0'), 'B17023_015E':('family_other_female','1-2'),
        'B17023_016E':('family_other_female','3-4'), 'B17023_017E':('family_other_female','5+'),
        'B17023_021E':('family_married','0'), 'B17023_022E':('family_married','1-2'),
        'B17023_023E':('family_married','3-4'), 'B17023_024E':('family_married','5+'),
        'B17023_026E':('family_other_male','0'), 'B17023_027E':('family_other_male','1-2'),
        'B17023_028E':('family_other_male','3-4'), 'B17023_029E':('family_other_male','5+'),
        'B17023_031E':('family_other_female','0'), 'B17023_032E':('family_other_female','1-2'),
        'B17023_033E':('family_other_female','3-4'), 'B17023_034E':('family_other_female','5+'),
    }
    obs_brackets = {t:{'0':0,'1-2':0,'3-4':0,'5+':0} for t in obs_type if t.startswith('family')}
    for tag, (ftype, bracket) in mapping.items():
        obs_brackets[ftype][bracket] += dists['children_count_by_family'].get(tag, 0)

    def to_bracket(x):
        return '0' if x == 0 else '1-2' if x <= 2 else '3-4' if x <= 4 else '5+'

    for typ, obs_dict in obs_brackets.items():
        sub = df_hh[df_hh['type'] == typ]
        gen_counts = sub['child_count'].map(to_bracket).value_counts().to_dict()
        results[f'child_bracket_{typ}'] = compare_distributions(obs_dict, gen_counts)

    # 7) Relationship counts
    obs_rels = dists['relationship_counts']
    gen_rels = {
        'householder': df_hh['householder_cnt'].sum(),
        'spouse':      df_hh['spouse_cnt'].sum(),
        'partner':     df_hh.get('partner_cnt', pd.Series()).sum(),
        'child':       df_hh['child_cnt'].sum(),
        'other_rel':   df_hh['other_rel_cnt'].sum(),
        'other_nonrel':df_hh['other_nonrel_cnt'].sum(),
    }
    results['relationship_counts'] = compare_distributions(obs_rels, gen_rels)

    return pd.DataFrame(results).T
