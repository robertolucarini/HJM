import pandas as pd
import numpy as np
import os
import io
import polars as pl
from src.config import HJM_VOL_METHOD, CALIB_SWAPTION_EXP, CALIB_SWAPTION_TEN, PCA_F, SIM_PATHS, OPT_EXP, SWP_MAT, VOL_DATA_NAME

# -----------------
# Logs
# -----------------
def log_header(title):
    print(f"\n" + "="*65)
    print(f"  {title.upper()}")
    print("="*65)

def log_subheader(title):
    print(f"\n" + "-"*65)
    print(f"  {title}")
    print("-"*65)

def log_info(label, value, unit=""):
    print(f"  > {label:<25} : {value} {unit}")

def log_step(message):
    print(f"\n  > {message}")

def log_table(headers, rows):
    w = [20, 15, 15]
    row_fmt = f"{{:<{w[0]}}} | {{:<{w[1]}}} | {{:<{w[2]}}}"
    
    print("-" * 65)
    print(row_fmt.format(*headers))
    print("-" * 65)
    
    for row in rows:
        row_clean = [str(r) for r in row]
        print(row_fmt.format(*row_clean))
    print("-" * 65)

# -----------------
# Data
# -----------------
def parse_string_tenor(s):
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).upper().replace(' ', '')
    try:
        if 'M' in s:
            numeric_part = ''.join(c for c in s if c.isdigit() or c == '.')
            return float(numeric_part) / 12.0
        if 'Y' in s:
            numeric_part = ''.join(c for c in s if c.isdigit() or c == '.')
            return float(numeric_part)
        return float(s)
    except Exception as e:
        print(f"Error parsing tenor '{s}': {e}")
        return 0.0

def load_swaption_vols(vol_path):
    if not os.path.exists(vol_path):
        raise FileNotFoundError(f"Vol file not found at: {os.path.abspath(vol_path)}")
    df_vols = pd.read_csv(vol_path, index_col=0, sep=";")
    df_vols = df_vols.astype(str).replace(',', '.', regex=True)
    df_vols = df_vols.apply(pd.to_numeric, errors='coerce')
    return df_vols

def fetch_ecb_data(start_date='2023-01-01', curve_type='IF', verbose=True):
    import requests

    url = f"https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.?startPeriod={start_date}&detail=dataonly&format=csvdata"
    
    if verbose:
        log_step(f"Fetching ECB data (Type: {curve_type}, Start: {start_date})...")
    
    response = requests.get(url, headers={'Accept': 'text/csv'})
    if not response.ok:
        raise ConnectionError(f"ECB API Error: {response.status_code}")

    q = (
        pl.read_csv(io.BytesIO(response.content), ignore_errors=True)
        .filter(pl.col("KEY").str.contains("G_N_A") & pl.col("KEY").str.contains(curve_type))
        .with_columns([
            pl.col("KEY").str.split("_").list.last().alias("Tenor_Label")
        ])
        .filter(pl.col("Tenor_Label").str.contains(r"^\d+[MY]$"))
        .with_columns([
            pl.col("Tenor_Label").str.extract(r"(\d+)", 1).cast(pl.Float64).alias("Tenor_Val"),
            pl.col("Tenor_Label").str.extract(r"([MY])", 1).alias("Tenor_Unit"),
            (pl.col("OBS_VALUE").cast(pl.Float64) / 100.0).alias("OBS_VALUE")
        ])
        .with_columns([
            pl.when(pl.col("Tenor_Unit") == "M")
            .then(pl.col("Tenor_Val") / 12.0)
            .otherwise(pl.col("Tenor_Val"))
            .alias("Tenor")
        ])
        .filter((pl.col("Tenor") >= 0.4) & (pl.col("Tenor") <= 30))
        .select(["TIME_PERIOD", "Tenor", "OBS_VALUE"])
    )
    
    df_pivot = q.pivot(index="TIME_PERIOD", on="Tenor", values="OBS_VALUE", aggregate_function="first").sort("TIME_PERIOD")
    
    if df_pivot.height == 0:
        raise ValueError(f"No data found for curve type '{curve_type}'.")

    pdf = df_pivot.to_pandas()
    pdf["TIME_PERIOD"] = pd.to_datetime(pdf["TIME_PERIOD"])
    pdf = pdf.set_index("TIME_PERIOD")
    pdf.columns = pdf.columns.astype(float)
    pdf = pdf.sort_index(axis=1)
    
    if verbose:
        print(f"Loaded {len(pdf)} days. Shape: {pdf.shape}")
        
    return pdf

# -----------------
# Helpers
# -----------------
def forwards_to_yields(tenors, forwards):
    """ converts instantaneous forward rates to zero yields """
    from scipy.integrate import cumulative_trapezoid

    if tenors[0] > 1e-5:
        t_grid = np.insert(tenors, 0, 0.0)
        f_grid = np.insert(forwards, 0, forwards[0])
    else:
        t_grid = tenors
        f_grid = forwards

    # integrate f(t) dt -> integral values at each grid point
    integral = cumulative_trapezoid(f_grid, t_grid, initial=0)
    
    # avoid division by zero at t=0
    with np.errstate(divide='ignore', invalid='ignore'):
        yields = integral / t_grid
        
    # fix t=0 case (L'Hopital: limit is f(0))
    yields[0] = f_grid[0]
    
    # return only the points corresponding to original tenors
    if tenors[0] > 1e-5:
        return yields[1:]
    else:
        return yields

def get_annuity_and_atm_rate(times, zero_yields, T_exp, T_tenor):
    from scipy.interpolate import PchipInterpolator

    # P(0, t) = exp(-y(t) * t)
    log_p = -times * zero_yields
    p_spline = PchipInterpolator(times, log_p)
    
    def zc_price(t):
        if t <= 1e-6: return 1.0
        return np.exp(p_spline(t))

    pay_times = np.arange(1.0, T_tenor + 1e-6, 1.0) + T_exp    
    annuity = sum(zc_price(t) for t in pay_times)
    
    # swap rate = (P(Start) - P(End)) / annuity
    P_start = zc_price(T_exp)
    P_end = zc_price(T_exp + T_tenor)
    
    par_rate = (P_start - P_end) / annuity
    
    return annuity, par_rate

# -----------------
# Workflows
# -----------------
def load_and_prepare_data(start_date):
    """ fetches ECB data and calculates the zero curve """
    log_step(f"Connecting to ECB API (Start: {start_date})...")
    df_hist = fetch_ecb_data(start_date=start_date, curve_type='IF', verbose=False)
    
    times = df_hist.columns.values
    fwd_rates = df_hist.iloc[-1].values
    current_curve = forwards_to_yields(times, fwd_rates)
    
    log_info("Data Source", "ECB Statistical Warehouse")
    log_info("Historical Days", f"{len(df_hist)}")
    log_info("Latest Spot Rate", f"{current_curve[0]:.2%}")
    return df_hist, times, current_curve

def run_hjm_workflow(df_hist, times, current_curve, data_dir):
    from src.pricers import european_swaption_bachelier_pricer
    from src.models.hjm import HJM
    from src.calibration import calibrate_hjm

    log_subheader("CALIBRATION: HJM (PCA)")
    
    hjm = HJM(df_hist, times)
    hjm.run_pca(num_factors=int(PCA_F))
    
    # expl_var = np.sum(hjm.eigenvalues)/np.sum(hjm.eigenvalues * 0 + hjm.eigenvalues) * 100 
    log_info("PCA Factors", f"{int(PCA_F)}")

    if HJM_VOL_METHOD == 'PCA_SCALED':
        vol_path = os.path.join(data_dir, VOL_DATA_NAME)
        df_vols = load_swaption_vols(vol_path)
        
        try:
            vol_target_bps = df_vols.loc[f"{int(CALIB_SWAPTION_EXP)}Yr", f"{int(CALIB_SWAPTION_TEN)}Yr"]
        except KeyError:
            vol_target_bps = 85.0
        
        log_info("Target Instrument", f"{int(CALIB_SWAPTION_EXP)}y{int(CALIB_SWAPTION_TEN)}y Swaption")
        log_info("Market Vol", f"{vol_target_bps} bps")
        
        annuity, _ = get_annuity_and_atm_rate(times, current_curve, CALIB_SWAPTION_EXP, CALIB_SWAPTION_TEN)
        target_price = european_swaption_bachelier_pricer(vol_target_bps, annuity, CALIB_SWAPTION_EXP)
        
        calibrate_hjm(hjm, current_curve, target_price, CALIB_SWAPTION_EXP, CALIB_SWAPTION_TEN)

    log_subheader("SIMULATION CONFIGURATION")
    log_info("Paths", f"{SIM_PATHS}")
    log_info("Engine", "Sobol (Quasi-Monte Carlo)")
    
    log_step(f"Simulating {SIM_PATHS} paths...")
    paths = hjm.simulate(current_curve)
    return hjm, paths

def run_cheyette_workflow(times, current_curve, data_dir):
    from src.models.cheyette import Cheyette
    from src.calibration import calibrate_cheyette
    log_subheader("CALIBRATION: CHEYETTE")
    
    cheyette = Cheyette(times, current_curve, kappa=0.1, sigma=0.01, dt=1/252)
    vol_path = os.path.join(data_dir, VOL_DATA_NAME)
    df_vols = load_swaption_vols(vol_path)
    
    cal_kappa, cal_sigma = calibrate_cheyette(cheyette, df_vols)
    cheyette.kappa = cal_kappa
    cheyette.volatility_factors = np.array([[cal_sigma]])
    
    sim_horizon = max(SWP_MAT + OPT_EXP, 10.0)
    
    log_subheader("SIMULATION CONFIGURATION")
    log_info("Paths", f"{SIM_PATHS}")
    log_info("Engine", "Sobol (Quasi-Monte Carlo)")
    log_info("Horizon", f"{sim_horizon} Years")
    
    log_step("Generating paths...")
    sim_results = cheyette.simulate(current_curve, n_years=sim_horizon)
    paths, _, _, _ = sim_results 

    return cheyette, paths, sim_results

# -----------------
# Test
# -----------------
def test_pricing_consistency(model, current_curve, T_exp, T_tenor):
    """ compare european swaption price via analytival formula, MC, and Tree methods """
    from src.pricers import european_swaption_pricer, european_swaption_bachelier_pricer, LRS_TrinomialTree
    
    log_header("TEST: Pricing Consistency (ATM Swaption)")
    
    annuity, par_rate = get_annuity_and_atm_rate(model.tenors, current_curve, T_exp, T_tenor)
    strike = par_rate
    
    # Cheyette analytical
    try:
        k = model.kappa
        s = model.volatility_factors[0, 0]
        time_factor = np.sqrt((1.0 - np.exp(-2.0 * k * T_exp)) / (2.0 * k))
        damping_factor = (1.0 - np.exp(-k * T_tenor)) / (k * T_tenor)
        vol_normal = s * damping_factor * time_factor * 10000 
        price_anal = european_swaption_bachelier_pricer(vol_normal, annuity, T_exp)
    except AttributeError:
        price_anal = 0.0
        k, s = 0.0, 0.0
    
    price_mc = european_swaption_pricer(model, current_curve, T_exp, T_tenor, strike, n_sims=10000)
    
    try:
        n_tree_steps = int(T_exp * 100) 
        tree = LRS_TrinomialTree(model, T_exp, n_tree_steps)
        price_tree = tree.price_american_swaption(strike, T_tenor, exercise_dates=[T_exp])
    except:
        price_tree = 0.0
    
    log_info("Instrument", f"{T_exp}Y Expiry, {T_tenor}Y Tenor")
    log_info("Strike (ATM)", f"{strike:.4%}", "(Derived)")
    if k != 0:
        log_info("Params", f"k={k:.4f}, s={s:.4f}")

    headers = ["Method", "Price (bps)", "Diff (bps)"]
    rows = [
        ["Analytical", f"{price_anal*10000:.2f}", "-"],
        ["Monte Carlo", f"{price_mc*10000:.2f}", f"{(price_mc - price_anal)*10000:.2f}"],
        ["Trinomial Tree", f"{price_tree*10000:.2f}", f"{(price_tree - price_anal)*10000:.2f}"]
    ]
    log_table(headers, rows)

def test_mc_convergence(model, current_curve, T_exp, T_tenor, path_counts=[100, 1000, 5000, 20000]):
    """ check MC convergence """
    from src.pricers import european_swaption_pricer

    log_header("TEST: Monte Carlo Convergence")
    
    _, strike = get_annuity_and_atm_rate(model.tenors, current_curve, T_exp, T_tenor)
    
    log_info("Instrument", f"{T_exp}Y x {T_tenor}Y Swaption")
    log_info("Strike", f"{strike:.2%}")
    
    headers = ["Paths", "Price (bps)", "Diff Prev"]
    rows = []
    
    prev_price = 0.0
    
    for n in path_counts:
        price = european_swaption_pricer(model, current_curve, T_exp, T_tenor, strike, n_sims=n)
        price_bps = price * 10000
        diff = price_bps - prev_price if prev_price != 0 else 0.0
        rows.append([f"{n}", f"{price_bps:.2f}", f"{diff:.2f}"])
        prev_price = price_bps

    log_table(headers, rows)