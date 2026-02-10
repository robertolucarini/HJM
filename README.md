# HJM Framework

Framework for interest rate modeling, featuring both **Historical HJM** (PCA-based) and **Cheyette** (Markovian HJM) models. The project includes calibration to full volatility surfaces (Cheyette), Quasi-Monte Carlo simulations, and pricing engines for Caplets, European and Bermudan swaptions (Cheyette).

## Features

### Models
* **HJM (Heath-Jarrow-Morton):**
    * **Calibration:** Historical Principal Component Analysis (PCA). Extracts Level, Slope, and Curvature factors from ECB historical data.
    * **Use Case:** Realistic historical dynamics and fan-chart forecasting.
* **Cheyette (Markovian HJM / Hull-White):**
    * **Calibration:** Calibrates $\kappa$ (mean reversion) and $\sigma$ (volatility) to a full **Market Swaption Surface** (200+ instruments) using L-BFGS-B optimization.
    * **Use Case:** Market-consistent pricing of derivatives (Swaptions, Bermudans).

### Simulation Engine
* **Quasi-Monte Carlo (QMC):** Utilizes Scipy's **Sobol** sequence generator for faster convergence than standard pseudo-random numbers.
* **Variance Reduction:** Implements Brownian Bridges and Antithetic Variates.
* **State Variables:** Simulates the $x(t)$ and $y(t)$ state variables for the Cheyette model to avoid path-dependent integrals.

### Pricing Engines
* **European Products:**
    * **Caplets & Swaptions:** Monte Carlo pricing with Delta and Vega sensitivities.
    * **Analytical:** Generalized Bachelier formula for validating Monte Carlo results (Cheyette only).
* **Bermudan Swaptions:**
    * **Least Squares Monte Carlo (LSM):** Longstaff-Schwartz algorithm.
    * **Trinomial Tree:** Li-Ritchken-Sankarasubramanian (LRS, 1995) tree implementation for fast American/Bermudan pricing.

## Data Source
* **Live API:** Connects to the **ECB Statistical Data Warehouse** to fetch the latest AAA-rated EUR Instantaneous Forward curve (Svensson model).
* **Local Fallback:** Reads `data/ecb_data.csv` for offline development.

## Output Example

The framework generates the following report in the console:

```text
=================================================================
  MODEL: CHEYETTE
=================================================================

-----------------------------------------------------------------
  DATA
-----------------------------------------------------------------
  => Connecting to ECB API (Start: 2023-01-01)...
  > Data Source                 : ECB Statistical Warehouse 
  > Historical Days             : 790 
  > Latest Spot Rate            : 2.02%

-----------------------------------------------------------------
  CALIBRATION: CHEYETTE
-----------------------------------------------------------------
  ... Parsing Market Volatility Surface...
  > Calibration Points          : 251 swaptions 
  ... Starting Optimization (L-BFGS-B)...
  > Calibrated Kappa            : 0.0663 
  > Calibrated Sigma            : 0.0038 
  > Fit Error (RMSE)            : 22.35 bps

-----------------------------------------------------------------
  SIMULATION CONFIGURATION
-----------------------------------------------------------------
  > Paths                       : 4096
  > Engine                      : Sobol (Quasi-Monte Carlo)
  > Horizon                     : 10.0 Years

-----------------------------------------------------------------
  PRICING: European Swaption (1.0y5.0y)
-----------------------------------------------------------------
  > Strike                      : 2.00%
  > Model Price                 : 329.7 bps
  > Delta                       : 0.000522
  > Vega                        : 0.010927

-----------------------------------------------------------------
  PRICING: Bermudan Swaption (Monte Carlo)
-----------------------------------------------------------------
  > Structure                   : 5Y into 5Y (Annual Exercise) 
  > MC Price                    : 594.9 bps

-----------------------------------------------------------------
  PRICING: Bermudan Swaption (LRS Tree)
-----------------------------------------------------------------
  > Exercise Type               : Annual (Co-Terminal) 
  > Tree Price                  : 595.3 bps

```

## References

* **Cheyette, O. (1992).** "Term Structure Dynamics and Mortgage Valuation." *The Journal of Fixed Income*, 1(4), 28-41.
* **Clewlow, L., and Strickland, C. (1998).** *Implementing Derivative Models.* Wiley.
* **Glasserman, P. (2003).** *Monte Carlo Methods in Financial Engineering.* Springer.
* **Heath, D., Jarrow, R., and Morton, A. (1992).** "Bond Pricing and the Term Structure of Interest Rates." *Econometrica*, 60(1), 77-105.
* **Li, A., Ritchken, P., and Sankarasubramanian, L. (1995).** "Lattice Models for Pricing American Interest Rate Claims." *The Journal of Finance*, 50(1), 239-259.
* **Ritchken, P., and Sankarasubramanian, L. (1995).** "Volatility Structures of Forward Rates and the Dynamics of the Term Structure." *Mathematical Finance*, 5(1), 55-72.
