# main.py
import os
import time
import numpy as np
from src.utils import fetch_ecb_data, load_swaption_vols, forwards_to_yields, get_annuity_and_atm_rate
from src.calibration import calibrate_cheyette, calibrate_hjm
from src.pricers import european_caplet_pricer, european_swaption_pricer, bermudan_swaption_pricer, price_american_tree_lrs,european_swaption_bachelier_pricer
from src.config import MODEL_SELECTION, OPT_EXP, SWP_MAT, STRIKE, DELTA_BUMP, VOL_BUMP, START_DATE, TESTS
from src.utils import log_header, log_subheader, log_info, log_step, load_and_prepare_data, run_hjm_workflow, run_cheyette_workflow
from src.utils import test_pricing_consistency, test_mc_convergence

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    
    log_header(f"MODEL: {MODEL_SELECTION}")

    # 1. data
    log_subheader("DATA")
    df_hist, times, current_curve = load_and_prepare_data(START_DATE)
    
    model_instance = None
    paths = None
    sim_results = None

    # 2. calibration and simulation
    if MODEL_SELECTION == 'HJM':
        model_instance, paths = run_hjm_workflow(df_hist, times, current_curve, DATA_DIR)
    elif MODEL_SELECTION == 'Cheyette':
        model_instance, paths, sim_results = run_cheyette_workflow(times, current_curve, DATA_DIR)
    else:
        raise ValueError(f"Unknown Model: {MODEL_SELECTION}")

    # 3. Pricing
    # European
    european_caplet_pricer(model_instance, current_curve, OPT_EXP, STRIKE, DELTA_BUMP, VOL_BUMP)
    european_swaption_pricer(model_instance, current_curve, OPT_EXP, SWP_MAT, STRIKE, DELTA_BUMP, VOL_BUMP)
    
    # Bermudan
    if MODEL_SELECTION == 'Cheyette':
        log_subheader("PRICING: Bermudan Swaption (Monte Carlo)")
        dt = model_instance.dt
        sim_horizon = max(SWP_MAT + OPT_EXP, 10.0)
        n_steps = int(sim_horizon / dt)
        time_grid = np.linspace(0, sim_horizon, n_steps + 1)
        
        exercise_dates = [1.0, 2.0, 3.0, 4.0, 5.0]
        bermudan_specs = {'Type': 'Bermudan Swaption','Strike': STRIKE,'Ex_Dates': exercise_dates,'Tenor': 5.0}
        
        # MC
        price_berm = bermudan_swaption_pricer(model_instance, sim_results, time_grid, bermudan_specs)
        log_info("MC Price", f"{price_berm*10000:.1f}", "bps")

        # Tree
        price_american_tree_lrs(model_instance, exercise_dates[-1], SWP_MAT, STRIKE)

        # 4. Tests    
        if TESTS :        
            # consistency
            test_pricing_consistency(model_instance, current_curve, T_exp=1.0, T_tenor=5.0)
            # convergence
            test_mc_convergence(model_instance, current_curve, T_exp=1.0, T_tenor=5.0)

    print(f"\n[Total Run Time: {time.time()-t0:.2f}s]")
    print("="*65 + "\n")