import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from src.utils import log_subheader, log_info

# -------------------------------------------------------------------------
# Europeans
# -------------------------------------------------------------------------
def european_caplet_pricer(model, current_curve, expiry_years, strike, delta_bump=0.0001, vol_bump=0.0001, n_sims=None):

    def run_simulation(curve_input):
        count = n_sims if n_sims is not None else 4096
        return model.simulate(curve_input, n_years=expiry_years, n_sims=count)
    
    # Base Price
    sim_base = run_simulation(current_curve)
    price_base = _calculate_caplet_payoff(model, sim_base, expiry_years, strike, current_curve=current_curve)
    
    # Delta
    curve_bumped = current_curve + delta_bump
    sim_delta = run_simulation(curve_bumped)
    price_delta = _calculate_caplet_payoff(model, sim_delta, expiry_years, strike, current_curve=curve_bumped)
    delta_pnl = price_delta - price_base
    delta = delta_pnl / delta_bump
    
    # Vega
    original_vol = model.volatility_factors.copy()
    model.volatility_factors = original_vol + vol_bump
    sim_vega = run_simulation(current_curve)
    price_vega = _calculate_caplet_payoff(model, sim_vega, expiry_years, strike, current_curve=current_curve)
    vega_pnl = price_vega - price_base
    vega = vega_pnl / vol_bump
    
    # reset
    model.volatility_factors = original_vol
    
    # logs
    if n_sims is None:
        log_subheader(f"PRICING: European Caplet ({expiry_years}Y)")
        log_info("Strike", f"{strike:.2%}")
        log_info("Model Price", f"{price_base*10000:.1f}", "bps")

        log_info(f"Delta PnL (+{delta_bump*10000:.1f}bp)", f"{delta_pnl*10000:.3f}", "bps")
        log_info(f"Vega PnL (+{vol_bump*100:.2f}% vol)", f"{vega_pnl*10000:.3f}", "bps")
    
    return price_base

def european_swaption_pricer(model, current_curve, expiry_years, tenor_years, strike, delta_bump=0.0001, vol_bump=0.0001, n_sims=None):
    
    def run_simulation(curve_input):
        count = n_sims if n_sims is not None else 4096
        return model.simulate(curve_input, n_years=expiry_years, n_sims=count)

    # Base
    sim_base = run_simulation(current_curve)
    price_base = _calculate_swaption_payoff(model, sim_base, expiry_years, tenor_years, strike, current_curve)
    
    # Delta
    curve_bumped = current_curve + delta_bump
    sim_delta = run_simulation(curve_bumped)
    price_delta = _calculate_swaption_payoff(model, sim_delta, expiry_years, tenor_years, strike, curve_bumped)
    delta_pnl = price_delta - price_base
    delta = delta_pnl / delta_bump
    
    # Vega
    original_vol = model.volatility_factors.copy()
    model.volatility_factors = original_vol + vol_bump
    sim_vega = run_simulation(current_curve)
    price_vega = _calculate_swaption_payoff(model, sim_vega, expiry_years, tenor_years, strike, current_curve)
    vega_pnl = price_vega - price_base
    vega = vega_pnl / vol_bump
    
    # reset
    model.volatility_factors = original_vol
    
    if n_sims is None:
        log_subheader(f"PRICING: European Swaption ({expiry_years}y{tenor_years}y)")
        log_info("Strike", f"{strike:.2%}")
        log_info("Model Price", f"{price_base*10000:.1f}", "bps") 
        log_info(f"Delta PnL (+{delta_bump*10000:.1f}bp)", f"{delta_pnl*10000:.3f}", "bps")
        log_info(f"Vega PnL (+{vol_bump*100:.2f}% vol)", f"{vega_pnl*10000:.3f}", "bps")

    return price_base

def _integrate_hjm_curve(model_tenors, f_curve, target_tenors):
    """  integrate HJM forward curve f(T_exp, tau) to get Zero Coupon Bonds P(T_exp, tau) """
    
    # integration grid
    max_tenor = np.max(target_tenors)
    relevant_mask = model_tenors <= max_tenor + 1.0
    grid_tenors = model_tenors[relevant_mask]
    
    # add 0 if missing
    if grid_tenors[0] > 1e-6:
        full_grid = np.concatenate(([0.0], grid_tenors))
        # flat extrapolate rate at 0 -> f_curve shape: (Sims, Tenors)
        rate_0 = f_curve[:, 0:1] 
        full_rates = np.concatenate((rate_0, f_curve[:, relevant_mask]), axis=1)
    else:
        full_grid = grid_tenors
        full_rates = f_curve[:, relevant_mask]

    # cumulative integral f(u) du -> (Sims, GridPoints)
    integrals = cumulative_trapezoid(full_rates, full_grid, axis=1, initial=0)
    
    # interpolate
    interpolator = interp1d(full_grid, integrals, kind='linear', axis=1, fill_value="extrapolate")
    target_integrals = interpolator(target_tenors)
    
    return np.exp(-target_integrals)

def _calculate_caplet_payoff(model, paths_input, expiry, strike, tenor=0.5, current_curve=None):
    if isinstance(paths_input, tuple):
        paths, x, y, r = paths_input
        return _calculate_caplet_payoff_exact(model, x, y, r, current_curve, expiry, strike, tenor)

    paths = paths_input
    n_steps_expected = int(expiry / model.dt)
    if abs(paths.shape[0] - n_steps_expected) > 5 and abs(paths.shape[1] - n_steps_expected) <= 5:
        # swap axes 0 and 1 to get (Steps, Sims, Tenors)
        paths = np.transpose(paths, (1, 0, 2))

    t_idx = -1 
    
    # short rate from simulation
    r_t = paths[:t_idx, :, 0] 
    dt = model.dt
    # Rienmann sum
    integral_r = np.sum(r_t, axis=0) * dt
    # stochastic disc factor
    df = np.exp(-integral_r)

    # forward curve at expiry ->  (Sims, Tenors)
    f_curve = paths[t_idx, :, :]
    
    # P(T, T+tenor) => integrate f(T, u) from 0 to tenor.
    target_tenor = np.array([tenor])
    P_fwd = _integrate_hjm_curve(model.tenors, f_curve, target_tenor)[:, 0]
    
    # libor L(T, T+tenor)
    libor_rate = (1.0 / tenor) * ((1.0 / P_fwd) - 1.0)
    
    # payoff
    payoff = np.maximum(libor_rate - strike, 0.0) * tenor
    
    # mean of discounted payoff
    return np.mean(payoff * df)

def _calculate_swaption_payoff(model, paths_input, expiry, tenor, strike, current_curve=None):
    if isinstance(paths_input, tuple):
        paths, x, y, r = paths_input
        from src.pricers import _calculate_swaption_payoff_exact
        return _calculate_swaption_payoff_exact(model, x, y, r, current_curve, expiry, tenor, strike)

    paths = paths_input
    
    n_steps_expected = int(expiry / model.dt)
    if abs(paths.shape[0] - n_steps_expected) > 5 and abs(paths.shape[1] - n_steps_expected) <= 5:
        paths = np.transpose(paths, (1, 0, 2))

    t_idx = -1
    
    # simulated short rate from 0 to T 
    r_t = paths[:t_idx, :, 0]
    # stochastic discount factor from T to 0
    D_0_T = np.exp(-np.sum(r_t, axis=0) * model.dt)
    
    # forward rate from T, as of T
    f_curve = paths[t_idx, :, :]
    
    pay_tenors = np.arange(1.0, tenor + 0.001, 1.0)
    
    # bond prices
    bond_prices = _integrate_hjm_curve(model.tenors, f_curve, pay_tenors)
    
    P_final = bond_prices[:, -1]
    annuity = np.sum(bond_prices, axis=1)
    S_T = (1.0 - P_final) / annuity
    
    # swaption payoff
    payoff = np.maximum(S_T - strike, 0.0) * annuity
    # mean of discounted (to 0) payoff

    return np.mean(payoff * D_0_T)

def _calculate_caplet_payoff_exact(model, x_sim, y_sim, r_sim, current_curve, expiry, strike, tenor=0.5):
    """Exact Cheyette caplet payoff using simulated short rate and bond formula."""
    from scipy.interpolate import PchipInterpolator

    curve_to_use = current_curve if current_curve is not None else model.curve_spline(model.tenors)
    log_p = -model.tenors * curve_to_use
    p_spline = PchipInterpolator(model.tenors, log_p)

    def get_P0_t(t):
        if t <= 1e-6:
            return 1.0
        return np.exp(p_spline(t))

    x_T = x_sim[-1, :]
    y_T = y_sim[-1, :]
    D_0_T = np.exp(-np.sum(r_sim[:-1, :], axis=0) * model.dt)

    P_0_T = get_P0_t(expiry)
    P_0_Tf = get_P0_t(expiry + tenor)
    B = tenor if abs(model.kappa) < 1e-8 else (1.0 - np.exp(-model.kappa * tenor)) / model.kappa

    P_T_Tf = (P_0_Tf / P_0_T) * np.exp(-B * x_T - 0.5 * (B**2) * y_T)
    libor_rate = (1.0 / tenor) * ((1.0 / P_T_Tf) - 1.0)
    payoff = np.maximum(libor_rate - strike, 0.0) * tenor

    return np.mean(payoff * D_0_T)

def _calculate_swaption_payoff_exact(model, x_sim, y_sim, r_sim, current_curve, expiry, tenor, strike):
    """ exact Cheyette payoff """
    from scipy.interpolate import PchipInterpolator
    
    log_p = -model.tenors * current_curve
    p_spline = PchipInterpolator(model.tenors, log_p)
    
    def get_P0_t(t):
        if t <= 1e-6: return 1.0
        return np.exp(p_spline(t))

    x_T = x_sim[-1, :]
    y_T = y_sim[-1, :]
    D_0_T = np.exp(-np.sum(r_sim[:-1, :], axis=0) * model.dt)

    pay_times = np.arange(1.0, tenor + 0.001, 1.0)
    annuity = np.zeros_like(x_T)
    P_final = np.zeros_like(x_T)
    P_0_Texp = get_P0_t(expiry)
    
    for mat in pay_times:
        T_pay = expiry + mat
        tau = T_pay - expiry
        P_0_Tpay = get_P0_t(T_pay)
        ratio = P_0_Tpay / P_0_Texp
        B = tau if abs(model.kappa) < 1e-8 else (1.0 - np.exp(-model.kappa * tau)) / model.kappa
        stoch = np.exp(-B * x_T - 0.5 * (B**2) * y_T)
        P_T_pay = ratio * stoch
        annuity += P_T_pay
        if mat == pay_times[-1]:
            P_final = P_T_pay
            
    S_T = (1.0 - P_final) / annuity
    payoff = np.maximum(S_T - strike, 0.0) * annuity
    
    return np.mean(payoff * D_0_T)

def european_swaption_bachelier_pricer(vol_normal_bps, annuity, T_exp):
    """ ATM Swaption - Bachelier """
    sigma = vol_normal_bps / 10000.0
    return annuity * sigma * np.sqrt(T_exp) * (1/np.sqrt(2*np.pi))

# -------------------------------------------------------------------------
# Bermudans with MC
# -------------------------------------------------------------------------
def bermudan_swaption_pricer(model, sim_data, time_grid, trade_specs):

    # unpack stoc processes from simulaiton data
    if isinstance(sim_data, tuple):
        _, x_sim, y_sim, r_sim = sim_data
    else:
        # fallback
        return 0.0 
    
    strike = trade_specs['Strike']
    # exercise dates
    dates = trade_specs['Ex_Dates']
    swap_tenor = trade_specs.get('Tenor', 5.0)
    # simulated paths
    n_paths = x_sim.shape[1]
    ex_steps = [np.abs(time_grid - t).argmin() for t in dates]
    # initialize cashflows
    cashflows = np.zeros(n_paths)
    
    # numeraire D(0,t)
    dt = model.dt
    # numeraire, cumulative stoc disc factor -> (Paths, Steps) -> transpose
    disc_factors = np.exp(-np.cumsum(r_sim, axis=0) * dt).T 

    # backward induction
    for i in range(len(ex_steps)-1, -1, -1):
        step = ex_steps[i]
        t_val = time_grid[step]
        # disc factor for this time step
        df_t = disc_factors[:, step]
        
        # cheyette processes
        x_t = x_sim[step, :]
        y_t = y_sim[step, :]
        
        # underlying swap pay dates
        pay_dates = np.arange(1.0, swap_tenor + 0.001, 1.0)
        # init
        annuity = np.zeros(n_paths)
        p_final = np.zeros(n_paths)
        
        # disc factor from t to 0
        P_0_t = model.zc_price_market(t_val)
        
        # calc annuity at each swap paydate
        for mat in pay_dates:
            # total swap time-to-maturity from now
            T_mat = t_val + mat
            # disc factor from T_mat to 0
            P_0_T = model.zc_price_market(T_mat)
            tau = T_mat - t_val
            # Cheyette B
            B = tau if abs(model.kappa) < 1e-8 else (1.0 - np.exp(-model.kappa * tau)) / model.kappa

            # Cheyette bond price
            P_t_T = (P_0_T / P_0_t) * np.exp(-B * x_t - 0.5 * (B**2) * y_t)
            # annuity
            annuity += P_t_T
            # last paydate
            if mat == pay_dates[-1]:
                p_final = P_t_T
        
        # swap rate
        par_rate = (1.0 - p_final) / annuity
        # discounted intrinsic value -> exercise today 
        intrinsic_t0 = np.maximum((par_rate - strike) * annuity, 0.0) * df_t
        
        # regression
        if i == len(ex_steps) - 1:
            cashflows = intrinsic_t0
        else:
            # basis function
            X = np.column_stack([np.ones(n_paths), par_rate, par_rate**2, par_rate*annuity])
            # in-the-money mask
            itm = intrinsic_t0 > 0
            # initialize continuation value
            continuation_t0 = np.zeros(n_paths)
            
            if np.sum(itm) > 50:
                y_target = cashflows[itm]
                # least squares
                coeffs = np.linalg.lstsq(X[itm], y_target, rcond=None)[0]
                # update continuation value
                continuation_t0[itm] = X[itm] @ coeffs
            # decision
            do_ex = intrinsic_t0 > continuation_t0
            # update cashflow with intrinsic value only for exercising paths
            cashflows[do_ex] = intrinsic_t0[do_ex]
            
    return np.mean(cashflows)

# -------------------------------------------------------------------------
# Bermudans with LRS 1995' Trinomial Tree
# -------------------------------------------------------------------------
class LRS_TrinomialTree:
    """ Li-Ritchken-Sankarasubramanian (LRS) framework """
    
    def __init__(self, model, T_horizon, n_steps):
        self.model = model
        self.dt = T_horizon / n_steps
        self.n_steps = n_steps
        self.T_horizon = T_horizon
        
        self.sigma = model.volatility_factors[0, 0]
        self.kappa = model.kappa
        # vertical distance 
        self.dx = self.sigma * np.sqrt(3.0 * self.dt)
        
        self._build_deterministic_shifts()
        # control max number of nodes up or down
        self.j_max = min(int(0.184 / (self.model.kappa * self.dt)) + 1, self.n_steps)

    def _build_deterministic_shifts(self):
        """ 
        LRS short-rate: r = f + y + x 
        -> returns f, y (same for each node the the given time step) 
        """
        self.y_grid = np.zeros(self.n_steps + 1)
        self.f_grid = np.zeros(self.n_steps + 1)
        self.time_grid = np.linspace(0, self.T_horizon, self.n_steps + 1)
        
        y_t = 0.0
        for i in range(self.n_steps):
            self.y_grid[i] = y_t
            # inst forward rate for this time step -> centers the grid
            self.f_grid[i] = self.model.get_instantaneous_forward_rate(self.time_grid[i])
            # change in convexity -> Euler => (d'y / d't) = sigma**2 - 2*k*y
            dy = (self.sigma**2 - 2 * self.model.kappa * y_t) * self.dt
            # update y over time steps 
            y_t += dy
        
        # add y at last step
        self.y_grid[-1] = y_t
        # set inst forward rate at last step
        self.f_grid[-1] = self.model.get_instantaneous_forward_rate(self.time_grid[-1])

    def _calc_intrinsic(self, step_idx, j, strike, tenor):
        """ payoff engine -> swap value in each node """

        # step
        t = self.time_grid[step_idx]
        # x at this step, this node
        x = j * self.dx
        # y at this step (same for all nodes)
        y = self.y_grid[step_idx]
        
        # swap schedule
        pay_times = np.arange(1.0, tenor + 0.001, 1)
        annuity = 0.0
        P_final = 0.0
        
        # disc factor to current node
        P_0_t = self.model.zc_price_market(t)
        
        # get P(t,T) for each paydate
        for mat in pay_times:
            T_mat = t + mat
            tau = T_mat - t
            # disc factor to paydate
            P_0_T = self.model.zc_price_market(T_mat)
            # cheyette B
            B = (1.0 - np.exp(-self.kappa * tau)) / self.kappa
            # cheyette bond price
            val = (P_0_T / P_0_t) * np.exp(-B * x - 0.5 * (B**2) * y)
            # annuity, sum of all bond prices
            annuity += val
            if mat == pay_times[-1]:
                # bond price at swap maturity
                P_final = val

        # swap par rate
        swap_rate = (1.0 - P_final) / annuity
        # swaption payoff
        return max(swap_rate - strike, 0) * annuity

    def price_american_swaption(self, strike, tenor, exercise_dates):
        """ Longstaff-Swartz for Trinomial """
        
        # set() for faster lookup
        ex_steps = set()
        # from yearsto tree steps
        for t in exercise_dates:
            step = int(round(t / self.dt))
            if step <= self.n_steps:
                ex_steps.add(step)

        # Terminal Condition (Expiring Option)
        j_limit = min(self.n_steps, self.j_max)
        j_min_curr, j_max_curr = -j_limit, j_limit
        size = j_max_curr - j_min_curr + 1
        values = np.zeros(size)
        
        # at maturity, value = intrinsic
        for idx, j in enumerate(range(j_min_curr, j_max_curr + 1)):
             values[idx] = self._calc_intrinsic(self.n_steps, j, strike, tenor)

        # backward induction
        for i in range(self.n_steps - 1, -1, -1):
            # trinomial tree goes one up, one down -> the tree grows as a cone until the boudles, then becomes a tube
            j_limit = min(i, self.j_max)
            # active nodes at this time step
            j_min, j_max = -j_limit, j_limit
            # number of nodes at this time step
            width = j_max - j_min + 1
            # dynamically reshape 1d array of values in each time step, as tree evolves
            new_values = np.zeros(width)
            # shift value to align the tree indices with numpy indeces 
            # => at i=2, j_min = -2 -> cannot ask numpy to fecth [-2], array starts at 0 => shift to zero 
            offset_map = -j_min_curr 
            
            for idx, j in enumerate(range(j_min, j_max + 1)):
                # x at this step, for this node
                x_val = j * self.dx
                # x at this step, for this node 
                drift = self.y_grid[i] - self.kappa * x_val
                                
                # lattice is fixed, knows only 0, delta_x, 2delta_x  
                # re-centered the tree based on: distance to move (drift*dt) expressed in grid point (/dx)
                # index shift 
                k = int(round(drift * self.dt / self.dx))
                
                # discretization error from above rounding 
                # => used to adjust probabilities
                eta = drift * self.dt / self.dx - k
                
                # prob of moving up, middle, down 
                pu = 1/6 + 0.5 * (eta**2 + eta)
                pm = 2/3 - eta**2
                pd = 1/6 + 0.5 * (eta**2 - eta)
                
                # tree indices (shifted with offset_map)
                idx_u = (j + k + 1) + offset_map
                idx_m = (j + k) + offset_map
                idx_d = (j + k - 1) + offset_map
                
                # control for out of bound movements
                def get_v(ix):
                    if 0 <= ix < len(values): 
                        return values[ix]
                    return 0.0
                
                # short rate: r = f + x + y
                r = self.f_grid[i] + x_val + self.y_grid[i]
                # disc factor for this step 
                df = np.exp(-r * self.dt)
                # contnuation value
                # -> Expected Value from the next step
                v_cont = (pu * get_v(idx_u) + pm * get_v(idx_m) + pd * get_v(idx_d)) * df
                
                # decision
                if i in ex_steps:
                    intrinsic = self._calc_intrinsic(i, j, strike, tenor)
                    # update flows for exercising paths
                    new_values[idx] = max(v_cont, intrinsic)
                else:
                    new_values[idx] = v_cont
            
            values = new_values
            j_min_curr, j_max_curr = j_min, j_max
    
        # swaption price 
        return values[0]

def price_american_tree_lrs(model, expiry_years, tenor_years, strike):
    log_subheader("PRICING: Bermudan Swaption (LRS Tree)")    # daily steps
    
    n_steps = int(expiry_years * 252)
    
    # init
    tree = LRS_TrinomialTree(model, expiry_years, n_steps)
    # exercise dates
    ex_dates = np.arange(1.0, expiry_years + 0.001, 1.0)
    
    # final price
    price = tree.price_american_swaption(strike, tenor_years, ex_dates)
    
    # logs
    log_info("Exercise Type", "Annual (Co-Terminal)")
    log_info("Tree Price", f"{price*10000:.1f}", "bps")
    return price