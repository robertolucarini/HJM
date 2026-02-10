
import numpy as np
from scipy.stats import qmc, norm
from src.config import SEED,DT_STEPS,SIM_YEARS, SIM_PATHS, SIM_METHOD, PCA_F


class HJM:
    def __init__(self, forward_rates, tenors, dt=DT_STEPS):
        self.raw_forward_rates = forward_rates
        self.tenors = np.array(tenors)
        self.dt = dt
        self.volatility_factors = None
        self.eigenvalues = None
        self.d_tau = np.mean(np.diff(self.tenors))

    def run_pca(self, num_factors=PCA_F):
        """ pca with 3 comps for rate vol """
        # diff fwd rates
        df = self.raw_forward_rates.diff().dropna().values
        # cov matrix
        cov_matrix = np.cov(df.T)
        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # sort descending
        idx = np.argsort(eigenvalues)[::-1]
        # take top 3 comps 
        self.eigenvalues = eigenvalues[idx][:num_factors]
        self.top_vectors = eigenvectors[:, idx][:, :num_factors]
        # vol factors
        self.volatility_factors = self.top_vectors * np.sqrt(self.eigenvalues)

        print(f"PCA Complete. Variance Explained: {np.sum(self.eigenvalues)/np.sum(eigenvalues)*100:.2f}%")

    def get_hjm_drift(self):
        """ HJM no-arb drift """
        # cumulated vol up to tenor 
        integral_sigma = np.cumsum(self.volatility_factors, axis=0) * self.d_tau
        # this tenor vol * cumulative vol (for all factors)
        term = self.volatility_factors * integral_sigma
        # sum drift contribution of each vol fator
        return np.sum(term, axis=1)

    # ---------------------------
    # Simulation
    # ---------------------------
    def get_sobol_paths(self, total_factors, n_paths, n_steps, seed=SEED):
        """ Sobol Quasi-Monte Carlo generator """
        dim = n_steps * total_factors
        
        # Sobol requires power of 2 paths
        m = int(np.ceil(np.log2(n_paths)))
        n_paths_optimal = 2**m 
        
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        # Uniform
        u_vec = sampler.random(n=n_paths_optimal)
        # inverse CDF to Gaussian
        z_vec = norm.ppf(u_vec) 
        
        # reshape to (Paths, Steps, Factors)
        return z_vec.reshape(n_paths_optimal, n_steps, total_factors)

    def get_brownian_bridge(self, z_input, n_steps):
        """ from standard Gaussian noise to Brownian Bridge increments """

        # z_input shape: (n_steps, n_paths) -> treating one factor at a time
        _, n_paths = z_input.shape
        W = np.zeros((n_steps + 1, n_paths))
        
        # Breadth-First Search
        # first bridge index is END
        bridge_indices = [n_steps]
        # stores which pairs of indeces returned the bridge index 
        map_dependency = {n_steps: (0, 0)}
        # store hierarchical order, start from (0,END)
        queue = [(0, n_steps)]
        
        while queue:
            # initial index-pair 
            left, right = queue.pop(0)
            # average point
            mid = (left + right) // 2
            if mid != left and mid != right:
                # store next bridge index
                bridge_indices.append(mid)
                # remember the indeces-pair that returned the above bridge index
                map_dependency[mid] = (left, right)
                # store in queu the new new intervals (the MIDs, the QUARTERs, the EIGHTs...)
                queue.append((left, mid))
                queue.append((mid, right))
        
        dt_step = 1.0 
        time_grid = np.arange(n_steps + 1) * dt_step

        for i, target_idx in enumerate(bridge_indices):
            # safety break
            if i >= n_steps: 
                break
            # shock for this hierarchical level
            z_val = z_input[i, :]
            
            if i == 0:
                # the END
                T = time_grid[n_steps]
                W[target_idx, :] = np.sqrt(T) * z_val
            else:
                # bridging
                left, right = map_dependency[target_idx]
                # left, target, right indeces
                ti, tj, tk = time_grid[left], time_grid[target_idx], time_grid[right]
                # distances
                dt_ki, dt_ji, dt_kj = tk - ti, tj - ti, tk - tj
                # bridge conditional mean: linear itnerpolation of left and right  
                mu = (dt_kj / dt_ki) * W[left, :] + (dt_ji / dt_ki) * W[right, :]
                # bridge conditiona variance
                sigma = np.sqrt((dt_ji * dt_kj) / dt_ki)
                # stoch process
                W[target_idx, :] = mu + sigma * z_val

        # stoch increments (dW)
        return np.diff(W, axis=0)

    def prepare_drivers(self, n_steps, n_sims, method, seed=SEED):
        """ Generates Correlated Brownian Increments (dW) using Sobol + Bridge + Antithetic """

        # vol factors
        num_factors = self.volatility_factors.shape[1]
        
        if method == 'standard':
            np.random.seed(seed)
            # random Normal noise
            Z = np.random.normal(0, 1, (n_sims, n_steps, num_factors))
            # scale by sqrt(dt)
            return Z * np.sqrt(self.dt)
        
        # antithetic variates
        n_gen = n_sims // 2
        
        # Sobol -> (n_gen, n_steps, factors)
        Z_sobol = self.get_sobol_paths(num_factors, n_gen, n_steps, seed)
        
        # Brownian Bridge for each factor
        Z_bridged = np.zeros_like(Z_sobol)
        for f in range(num_factors):
            # transpose to (Steps, Paths) for Bridge
            z_slice = Z_sobol[:, :, f].T 
            # brownian increments
            dw_bridge = self.get_brownian_bridge(z_slice, n_steps)
            # transpose back to (Paths, Steps)
            Z_bridged[:, :, f] = dw_bridge.T
    
        # antithetic variates -> cncatenate Z and -Z
        Z_final = np.concatenate([Z_bridged, -Z_bridged], axis=0)
        
        # scale increments by sqrt(dt)
        dW = Z_final * np.sqrt(self.dt)
        
        return dW

    def simulate(self, initial_curve=None, n_years=SIM_YEARS, n_sims=SIM_PATHS, method=SIM_METHOD, seed=SEED):
        """ MC engine """

        n_steps = int(n_years / self.dt)
        
        # increments (already sqrtd) -> (sims, time, factors)
        dW = self.prepare_drivers(n_steps, n_sims, method, seed)
        
        # adjust n_sims for sobol
        real_n_sims = dW.shape[0]
        
        # HJM drift -> constant for a given tenor
        drift_vector = self.get_hjm_drift()
        # reshape to (1,1,Tenor) to add it to each time step and each path automatically
        drift_inc = (drift_vector * self.dt)[np.newaxis, np.newaxis, :]
        
        # diffusion
        # dW: (Sims, Time, Factors) -> transpose to (Time, Sims, Factors) -> time should be primary axis
        dW = dW.transpose(1, 0, 2)
        # einstein summation: diffusion_t,sim,n = sum_f [dW_t,sim_f] * sigma_n,f
        # 'tsf' = time, sim, factor; 'nf' = tenor, factor -> 'tsn' = time, sim, tenor
        diffusion_inc = np.einsum('tsf,nf->tsn', dW, self.volatility_factors)
        
        # total change in rate for a single step
        total_inc = drift_inc + diffusion_inc
        # cumulative rate changes
        cumulative_changes = np.cumsum(total_inc, axis=0)
        
        # simulated path = start + changes
        paths = initial_curve[np.newaxis, np.newaxis, :] + cumulative_changes
        
        # manually add initial curve as step 0 
        start_block = np.tile(initial_curve, (1, real_n_sims, 1))
        # simulated paths
        paths = np.vstack([start_block, paths])
        
        return paths

