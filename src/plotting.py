
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm


class Controls:
    def __init__(self, hjm_model, paths, initial_curve):
        self.hjm = hjm_model
        self.paths = paths
        self.initial_curve = initial_curve
        self.tenors = hjm_model.tenors
        self.dt = hjm_model.dt
        
        # theme
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.titleweight'] = 'bold'

    def plot_pca_factors(self):
        """  PCA Factors """
        plt.figure(figsize=(12, 6))
        
        # data prep
        factors = self.hjm.volatility_factors[:, :3]
        labels = ['Factor 1 (Level)', 'Factor 2 (Slope)', 'Factor 3 (Curvature)']
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"] # Standard Tableau colors
        
        for i in range(3):
            plt.plot(self.tenors, factors[:, i], 
                     label=labels[i], color=colors[i], 
                     marker='o', markersize=6, linewidth=2.5, alpha=0.9)
        
        # styling
        plt.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
        plt.title("Principal Component Loadings (Volatility Factors)", fontsize=16, pad=15)
        plt.xlabel("Tenor (Years)")
        plt.ylabel("Volatility Loading")
        plt.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_fan_chart(self, tenor_idx=-1):
        """  Fan Chart """
        
        # data prep
        tenor_paths = self.paths[:, :, tenor_idx]
        days = np.arange(tenor_paths.shape[0]) * self.hjm.dt
        target_tenor = self.tenors[tenor_idx]
        
        # percentiles -> bands: 5-95% (lightest), 15-85%, 25-75% (darkest)
        p5 = np.percentile(tenor_paths, 5, axis=1)
        p95 = np.percentile(tenor_paths, 95, axis=1)
        p15 = np.percentile(tenor_paths, 15, axis=1)
        p85 = np.percentile(tenor_paths, 85, axis=1)
        p25 = np.percentile(tenor_paths, 25, axis=1)
        p75 = np.percentile(tenor_paths, 75, axis=1)
        median = np.median(tenor_paths, axis=1)

        # plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # sample paths
        idx_samples = np.random.choice(tenor_paths.shape[1], 100, replace=False)
        for i in idx_samples:
            ax.plot(days, tenor_paths[:, i], color='grey', alpha=0.3, lw=0.8)

        # bands
        # outer (5-95%)
        ax.fill_between(days, p5, p95, color='#1f77b4', alpha=0.10, label='5-95% Range')
        # middle (15-85%)
        ax.fill_between(days, p15, p85, color='#1f77b4', alpha=0.20)
        # inner (25-75%)
        ax.fill_between(days, p25, p75, color='#1f77b4', alpha=0.30)
        
        # median
        ax.plot(days, median, color='#0b3d91', lw=3, label='Median Path')
        
        # styling
        ax.set_title(f"Stochastic Evolution: {target_tenor}-Year Forward Rate", fontsize=16, pad=15)
        ax.set_xlabel("Simulation Time (Years)")
        ax.set_ylabel("Yield (%)")
        
        final_rate = median[-1] * 100
        textstr = f'Terminal Median: {final_rate:.2f}%'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        
        ax.legend(loc='upper left', frameon=True)
        sns.despine() 
        plt.tight_layout()
        plt.show()

    def plot_3d_evolution(self, sim_index=1):
        """ Surface """
        
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(self.tenors, np.arange(self.paths.shape[0]))
        Z = self.paths[:, sim_index, :]
        
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.9, antialiased=True)
        
        # remove the grey panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # grid
        ax.grid(False)
        
        ax.set_title(f"Yield Curve Surface (Simulation #{sim_index})", fontsize=16, pad=20)
        ax.set_xlabel("Tenor (Years)", fontsize=12, labelpad=10)
        ax.set_ylabel("Time Steps (Days)", fontsize=12, labelpad=10)
        ax.set_zlabel("Rate", fontsize=12, labelpad=10)
        
        ax.view_init(elev=25, azim=-60)
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label("Interest Rate Level", rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.show()

    def check_martingale(self, t_maturity=1.0):
        """ Martingale Test with Implied Yield Info """

        print(f"\n--- Martingale Test (Maturity {t_maturity}Y) ---")
        n_steps = int(t_maturity / self.hjm.dt)
        
        # 1. Market Price (Theoretical)
        market_integral = self.initial_curve[0] * t_maturity 
        market_price = np.exp(-market_integral)
        
        # Calculate Implied Yield from Price
        # y = -ln(P) / T
        implied_yield = -np.log(market_price) / t_maturity
        
        # 2. Monte Carlo Price
        r_t_paths = self.paths[:n_steps, :, 0] 
        mc_integral = np.sum(r_t_paths, axis=0) * self.dt
        mc_price = np.mean(np.exp(-mc_integral))
        
        # 3. Compare
        diff = mc_price - market_price
        error_bps = diff * 10000
        
        # Print with Implied Yield
        print(f"Market ZCB Price: {market_price:.5f} (Implied Yield: {implied_yield:.2%})")
        print(f"Model  ZCB Price: {mc_price:.5f}")
        print(f"Diff (bps):       {error_bps:.2f} bps")
        
        if abs(error_bps) < 10.0:
            print("RESULT: PASS")
        else:
            print("RESULT: WARNING")

    def plot_rate_distribution(self, tenor_idx=0):
        """ Distribution """

        final_rates = np.array(self.paths[-1, :, tenor_idx] * 100).flatten()
        target_tenor = self.tenors[tenor_idx]
        
        mu, std = norm.fit(final_rates)
        
        plt.figure(figsize=(12, 7))
        
        plt.hist(final_rates, bins=60, density=True, 
                 color='#69b3a2', alpha=0.7, edgecolor='white', linewidth=0.5,
                 label='Simulated Distribution')
        
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k--', linewidth=2.5, label=f'Normal Fit ($\mu$={mu:.2f}%)')
        
        plt.title(f"Terminal Distribution: {target_tenor}-Year Rate", fontsize=16, pad=15)
        plt.xlabel("Interest Rate (%)")
        plt.ylabel("Probability Density")
        
        stats_text = (f"Mean: {mu:.2f}%\n"
                      f"StdDev: {std:.2f}%")
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgrey')
        plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.legend(loc='upper left', frameon=True)
        sns.despine()
        plt.tight_layout()
        plt.show()
