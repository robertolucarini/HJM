import numpy as np
from scipy.optimize import minimize,minimize_scalar
from src.utils import parse_string_tenor, log_info, log_step
import numpy as np


def calibrate_cheyette(model, df_vols):
    """ calibrates Cheyette kappa, sigma to swaptions vol surface """
    
    market_data = []
    log_step("Parsing Market Volatility Surface...")

    for i, exp_label in enumerate(df_vols.index):
        T_exp = parse_string_tenor(str(exp_label))
        
        # skip very short expiries (> 1 month)
        if T_exp < 0.08: continue
        
        # loop for tenors to get market data
        for j, dur_label in enumerate(df_vols.columns):
            T_dur = parse_string_tenor(str(dur_label))
            
            # market vol
            try: vol_bps = float(df_vols.iloc[i, j])
            except (ValueError, TypeError): continue
            
            if np.isnan(vol_bps) or vol_bps <= 0: continue

            # total swap time to mat    
            T_mat = T_exp + T_dur
            
            # pay dates
            coupon_times = np.arange(1.0, T_dur + 1e-6, 1.0) + T_exp
            # discount factors
            discounts = np.array([model.zc_price_market(t) for t in coupon_times])
            
            # annuity
            ann = np.sum(discounts)
            # fallback for short tenors
            if ann < 1e-10:
                ann = model.zc_price_market(T_mat)

            # bond price at option expiry    
            P_exp = model.zc_price_market(T_exp)
            # bond price at swap maturity
            P_mat = model.zc_price_market(T_mat)
            # swap par rate
            S0 = (P_exp - P_mat) / ann
            
            market_data.append({
                'T_exp': T_exp, 
                'T_mat': T_mat, 
                'tenor': T_dur,
                'S0': S0, 
                'annuity': ann, 
                'mkt_vol': vol_bps / 10000.0
            })
            
    log_info("Calibration Points", f"{len(market_data)} swaptions")
    if not market_data:
        raise ValueError("No valid swaption volatility points found for Cheyette calibration.")

    # analytical vol formula (cheyette/HW1F)
    def model_implied_vol(kappa, sigma, item):
        T_exp = item['T_exp']
        tau = item['tenor']

        # if no k, no mean-rev => Ho-Lee        
        if abs(kappa) < 1e-5:
            return sigma * np.sqrt(T_exp)
            
        # short rate variance (OU process)
        # => V(t) = sigma^2 * (1 - e^(-2*k*t)) / 2k
        time_part = np.sqrt((1.0 - np.exp(-2.0 * kappa * T_exp)) / (2.0 * kappa))
        # how vol is reduced by tenor lenght
        damping_part = (1.0 - np.exp(-kappa * tau)) / (kappa * tau)
        # model vol
        return sigma * damping_part * time_part

    def objective(params):
        kappa, sigma = params
        error_sq = 0.0

        # loop across different swaptions
        for item in market_data:
            model_vol = model_implied_vol(kappa, sigma, item)
            # sum of squared errors (SSE)
            error_sq += (model_vol - item['mkt_vol'])**2
        # rmse more intuitive than sse
        rmse = np.sqrt(error_sq / len(market_data))
        return rmse

    # initial guess
    x0 = [0.10, 0.01] 
    bounds = ((0.001, 3.0), (0.0001, 0.5))
    
    log_step("Starting Optimization (L-BFGS-B)...")
    try:
        res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        final_rmse = res.fun * 10000
        best_kappa, best_sigma = res.x
        
        log_info("Calibrated Kappa", f"{best_kappa:.4f}")
        log_info("Calibrated Sigma", f"{best_sigma:.4f}")
        log_info("Fit Error (RMSE)", f"{final_rmse:.2f}", "bps")
        
        return best_kappa, best_sigma
        
    except Exception as e:
        print(f"Optimization failed: {e}. Using defaults.")
        return 0.1, 0.01

def calibrate_hjm(hjm_model, current_curve, target_price, expiry, tenor):
    """ finds a scalar 'lambda' to multiply PCA factors for HJM Swaption Price == Market Price """    
    log_step(f" > Optimizing PCA scaler for Target Price: {target_price*10000:.1f} bps...")
    
    # store original factors to avoid overwrite during opt
    original_factors = hjm_model.volatility_factors.copy()
    
    # fast pricer specific for the opt loop
    def pricing_proxy(scaler):
        # scale
        hjm_model.volatility_factors = original_factors * scaler
        
        # run small simulation
        paths = hjm_model.simulate(current_curve, n_years=expiry, n_sims=2000)
        
        # price swaption
        t_idx = -1 
        # short rate
        r_t = paths[:t_idx, :, 0]
        # discount factor D(0, T_exp)
        D_0_T = np.exp(-np.sum(r_t, axis=0) * hjm_model.dt)

        # simulated fwd rates
        k_end = np.abs(hjm_model.tenors - tenor).argmin() + 1
        f_curve = paths[t_idx, :, :k_end]

        # assuming tenors start > 0, the first period is 0->tenors[0]
        tenor_slice = hjm_model.tenors[:k_end]
        d_tau = np.diff(np.concatenate(([0.0], tenor_slice)))
        
        # bond prices P(T_exp, Ti) -> cumsum of forwards gives integral r ds
        bond_prices = np.exp(-np.cumsum(f_curve * d_tau, axis=1)) 

        # bond price at swap maturity        
        P_final = bond_prices[:, -1]
        # annuity
        annuity = np.sum(bond_prices, axis=1)
        # swap rate at T_exp
        S_T = (1.0 - P_final) / annuity
        
        # S_T and not S0 to make the swaption ATM wrt to the simulation
        avg_S_T = np.mean(S_T) 

        # swaption payoff        
        payoff = np.maximum(S_T - avg_S_T, 0) * annuity
        # model price
        model_price = np.mean(payoff * D_0_T)
        # squared residuals
        return (model_price - target_price)**2

    # optimize scalar
    res = minimize_scalar(pricing_proxy, bounds=(0.1, 10.0), method='bounded')

    # optimal scalar
    best_scalar = res.x
    log_info("Optimal PCA Scaler", f"{best_scalar:.4f}")    
    
    # apply final scaling (permanently)
    hjm_model.volatility_factors = original_factors * best_scalar

    return best_scalar