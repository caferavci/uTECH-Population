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

# Define the housing‐share function
def housing_share(income,
                  s_max=0.70,    # share at very low income
                  s_min=0.15,    # share at very high income
                  M=60_000,      # midpoint income of inflection
                  k=1.5,         # slope‐control
                  sigma=0.06     # log‐normal noise
                 ):
    """
    s(income) = s_min + (s_max – s_min) / [1 + (income/M)^k]
    then multiplied by lognormal noise to add micro‐variation
    """
    base = s_min + (s_max - s_min) / (1 + (income / M) ** k)
    noise = np.random.lognormal(mean=0.0, sigma=sigma, size=income.shape)
    return np.clip(base * noise, 0.05, 1.2)  
    # lower‐bound 5%, upper‐bound 120% of income


# -----------------------------
# Config
# -----------------------------
YEAR        = 2023
STATE_FIPS  = "36"   # New York
COUNTY_FIPS = "061"  # New York County (Manhattan)

ACS5_URL        = f"https://api.census.gov/data/{YEAR}/acs/acs5"
ACS5_SUBJECT_URL = f"https://api.census.gov/data/{YEAR}/acs/acs5/subject"

# B25077_001E: Median value (dollars) of owner-occupied housing units
# S1901_C01_012E: Median household income (subject table S1901)
VAR_HOUSING_VALUE = "B25077_001E"
VAR_MEDIAN_INCOME = "S1901_C01_012E"


def fetch_acs_table(url, variables, geo_for, geo_in, api_key):
    """
    小工具：从 Census API 拉一张表，返回 DataFrame
    url:          基础 URL（acs5 或 acs5/subject）
    variables:    list of variable names
    geo_for:      e.g. 'tract:*' or 'county:061'
    geo_in:       e.g. 'state:36 county:061' 或 'state:36'
    """
    params = {
        "get": ",".join(["NAME"] + variables),
        "for": geo_for,
    }
    if geo_in:
        params["in"] = geo_in
    if api_key is not None:
        params["key"] = api_key

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    cols = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=cols)
    return df


def get_manhattan_tracts_with_acs(api_key=None):
    tiger_url = f"https://www2.census.gov/geo/tiger/TIGER{YEAR}/TRACT/tl_{YEAR}_36_tract.zip"
    gdf_tracts = gpd.read_file(tiger_url)

    gdf_tracts = gdf_tracts[gdf_tracts["COUNTYFP"] == COUNTY_FIPS].copy()
    gdf_tracts["geoid"] = gdf_tracts["GEOID"].astype(str)

    df_tract_b25077 = fetch_acs_table(
        url=ACS5_URL,
        variables=[VAR_HOUSING_VALUE],
        geo_for="tract:*",
        geo_in=f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
        api_key=api_key
    )
    df_tract_b25077["geoid"] = (
        df_tract_b25077["state"] + df_tract_b25077["county"] + df_tract_b25077["tract"]
    )

    df_tract_s1901 = fetch_acs_table(
        url=ACS5_SUBJECT_URL,
        variables=[VAR_MEDIAN_INCOME],
        geo_for="tract:*",
        geo_in=f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
        api_key=api_key
    )
    df_tract_s1901["geoid"] = (
        df_tract_s1901["state"] + df_tract_s1901["county"] + df_tract_s1901["tract"]
    )

    tract_attrs = (
        df_tract_b25077[["geoid", VAR_HOUSING_VALUE]]
        .merge(
            df_tract_s1901[["geoid", VAR_MEDIAN_INCOME]],
            on="geoid",
            how="left"
        )
    )

    gdf_tracts_acs = gdf_tracts.merge(tract_attrs, on="geoid", how="left")

    # B25077_001E at county level
    df_county_b25077 = fetch_acs_table(
        url=ACS5_URL,
        variables=[VAR_HOUSING_VALUE],
        geo_for=f"county:{COUNTY_FIPS}",
        geo_in=f"state:{STATE_FIPS}",
        api_key=api_key
    )

    # S1901_C01_012E at county level
    df_county_s1901 = fetch_acs_table(
        url=ACS5_SUBJECT_URL,
        variables=[VAR_MEDIAN_INCOME],
        geo_for=f"county:{COUNTY_FIPS}",
        geo_in=f"state:{STATE_FIPS}",
        api_key=api_key
    )

    df_county = df_county_b25077.merge(
        df_county_s1901,
        on=["NAME", "state", "county", "county"],
        suffixes=("_b25077", "_s1901")
    )

    df_county = df_county[["NAME", VAR_HOUSING_VALUE, VAR_MEDIAN_INCOME, "state", "county"]]

    return gdf_tracts_acs, df_county


import numpy as np
import geopandas as gpd

def assign_building_prices_eq15(
    gdf_buildings,
    gdf_tracts_acs,
    value_col="B25077_001E",    # tract-level median value / rent
    price_col="price_annual",   # new column for building price
    seed=42
):
    """
    Implements Eq. (15) from the paper:
        1. Fit a Lognormal(mu_hat, sigma_hat) to tract medians {m_t}.
        2. For each tract t, sample p_j ~ Lognormal(mu_hat, sigma_hat) for each building j in t.
        3. Rescale p_j so that median({p_j: j in t}) == m_t.
    Result: gdf_buildings with a new column `price_col`.
    """
    np.random.seed(seed)

    # Ensure GeoDataFrame
    if not isinstance(gdf_buildings, gpd.GeoDataFrame):
        gdf_buildings = gpd.GeoDataFrame(gdf_buildings, geometry="geometry", crs="EPSG:4326")

    # Align CRS for spatial join
    if gdf_buildings.crs is None:
        gdf_buildings.set_crs("EPSG:4326", inplace=True)

    if gdf_tracts_acs.crs is None:
        # TIGER is usually EPSG:4269
        gdf_tracts_acs = gdf_tracts_acs.set_crs("EPSG:4269")

    if gdf_buildings.crs != gdf_tracts_acs.crs:
        gdf_buildings = gdf_buildings.to_crs(gdf_tracts_acs.crs)

    # 1) Attach tract + median value to each building via spatial join
    #    (building geometry on left, tract on right)
    tracts_min = gdf_tracts_acs[["GEOID", value_col, "geometry"]].copy()
    joined = gpd.sjoin(
        gdf_buildings,
        tracts_min,
        how="left",
        predicate="within",
    )

    # After sjoin, columns:
    # - 'geometry'          : building geometry
    # - 'GEOID'             : tract id
    # - value_col           : tract median value
    # - 'index_right'       : tract index

    # 2) Fit lognormal(mu_hat, sigma_hat) to tract medians {m_t}
    mt = gdf_tracts_acs[value_col].astype(float).values
    mt = mt[~np.isnan(mt)]
    mt = mt[mt > 0]

    log_mt = np.log(mt)
    mu_hat = log_mt.mean()
    sigma_hat = log_mt.std(ddof=1)

    # 3) For each tract, sample building prices and rescale median to m_t
    joined[price_col] = np.nan

    for geoid, group in joined.groupby("GEOID"):
        if pd.isna(geoid):
            # buildings that didn't get a tract (shouldn't happen if all in county)
            continue

        m_t_vals = group[value_col].dropna().unique()
        if len(m_t_vals) == 0:
            continue
        # All rows in a tract should share the same median; take the first
        m_t = float(m_t_vals[0])

        idx = group.index
        n = len(idx)
        if n == 0 or m_t <= 0:
            continue

        # raw lognormal draws
        raw = np.random.lognormal(mean=mu_hat, sigma=sigma_hat, size=n)

        median_raw = np.median(raw)
        if median_raw <= 0:
            scale = 1.0
        else:
            scale = m_t / median_raw

        prices = raw * scale
        joined.loc[idx, price_col] = prices

    # 4) Return buildings with attached tract ID, ACS value, and sampled price
    # Drop the sjoin helper column
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined

import numpy as np
import pandas as pd

def derive_household_capacity_from_proxy(
    gdf_buildings_priced,
    proxy_col="capacity",   # current column = height * area
    total_households=None,
    out_col="HH_capacity"
):
    """
    Turn a continuous capacity proxy (height * area) into an integer household
    capacity per building, proportional to the proxy and summing to total_households.
    For buildings with missing proxy, use the median proxy value.
    """
    gdf = gdf_buildings_priced.copy()

    if total_households is None:
        raise ValueError("total_households must be provided.")

    # 1) Get proxy as numeric
    proxy_raw = pd.to_numeric(gdf[proxy_col], errors="coerce")

    # 1a) Fill NaN proxy with median proxy
    if proxy_raw.notna().any():
        median_proxy = proxy_raw.median()
        proxy = proxy_raw.fillna(median_proxy)
    else:
        # Degenerate case: all proxies are NaN → give everyone equal share
        n_b = len(gdf)
        raw_caps = np.full(n_b, total_households / n_b)
        floors = np.floor(raw_caps)
        residuals = raw_caps - floors
        current_sum = int(floors.sum())
        diff = int(total_households - current_sum)

        caps = floors.astype(int)
        if diff > 0:
            idx = np.argsort(residuals)[::-1][:diff]
            caps[idx] += 1
        elif diff < 0:
            idx = np.argsort(residuals)[:abs(diff)]
            caps[idx] = np.maximum(0, caps[idx] - 1)

        gdf[out_col] = caps.astype(int)
        return gdf

    total_proxy = proxy.sum()

    if total_proxy <= 0:
        # Degenerate case: nonpositive total → equal share
        n_b = len(gdf)
        raw_caps = np.full(n_b, total_households / n_b)
    else:
        # 2) Raw capacity proportional to proxy
        raw_caps = proxy / total_proxy * total_households  # sum ≈ total_households

    # 3) Integerize so capacities sum exactly to total_households
    floors = np.floor(raw_caps)
    residuals = raw_caps - floors
    current_sum = int(floors.sum())
    diff = int(total_households - current_sum)

    caps = floors.astype(int)

    if diff > 0:
        # Add 1 to the diff largest residuals
        idx = np.argsort(residuals)[::-1][:diff]
        caps[idx] += 1
    elif diff < 0:
        # Remove 1 from the |diff| smallest residuals
        idx = np.argsort(residuals)[:abs(diff)]
        caps[idx] = np.maximum(0, caps[idx] - 1)

    # caps now sum exactly to total_households (some buildings can have 0)
    gdf[out_col] = caps.astype(int)
    return gdf

def assign_household_building_ids_sorted_capacity(
    df_hh,
    gdf_buildings_cap,
    hh_cost_col="housing_cost",
    b_price_col="price_annual",
    building_id_col="id",
    capacity_col="HH_capacity",
    hh_id_col=None,   # if you have a household ID column; otherwise we use the index
):
    """
    Assign each household to a building ID using 1D optimal matching with capacities.

    Returns a DataFrame with:
        - 'hh_key'  : household identifier (index or hh_id_col)
        - 'building_id' : assigned building ID
    """
    hh = df_hh.copy()
    bldg = gdf_buildings_cap.copy()

    # 1) Keep only valid rows and ensure numeric
    hh = hh[hh[hh_cost_col].notna()].copy()
    bldg = bldg[bldg[b_price_col].notna()].copy()

    hh[hh_cost_col] = pd.to_numeric(hh[hh_cost_col], errors="coerce")
    bldg[b_price_col] = pd.to_numeric(bldg[b_price_col], errors="coerce")
    bldg[capacity_col] = pd.to_numeric(bldg[capacity_col], errors="coerce").fillna(0).astype(int)

    hh = hh[hh[hh_cost_col].notna()].copy()
    bldg = bldg[bldg[b_price_col].notna()].copy()

    # 2) Define household key
    if hh_id_col is not None and hh_id_col in hh.columns:
        hh_key = hh[hh_id_col].copy()
    else:
        hh_key = hh.index.to_series()
    hh = hh.reset_index(drop=True)
    hh["hh_key"] = hh_key.values

    # 3) Sort households by cost and buildings by price
    hh = hh.sort_values(hh_cost_col).reset_index(drop=True)
    bldg = bldg.sort_values(b_price_col).reset_index(drop=True)

    n_hh = len(hh)
    total_capacity = int(bldg[capacity_col].sum())

    if n_hh == 0:
        raise ValueError("No valid households to match.")
    if total_capacity < n_hh:
        raise ValueError(
            f"Total building capacity ({total_capacity}) is less than "
            f"number of households ({n_hh})."
        )

    # 4) Greedy sorted-capacity assignment (1D optimal for |h - p|)
    assigned_building_ids = np.empty(n_hh, dtype=object)

    j = 0
    remaining = int(bldg.loc[j, capacity_col])

    for i in range(n_hh):
        # move to next building when capacity exhausted
        while remaining == 0:
            j += 1
            if j >= len(bldg):
                raise RuntimeError("Ran out of building capacity.")
            remaining = int(bldg.loc[j, capacity_col])

        assigned_building_ids[i] = bldg.loc[j, building_id_col]
        remaining -= 1

    # 5) Return a simple mapping: household -> building_id
    df_assign = pd.DataFrame({
        "hh_key": hh["hh_key"].values,
        building_id_col: assigned_building_ids,
    })

    return df_assign
