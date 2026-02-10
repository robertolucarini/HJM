
# model
MODEL_SELECTION = 'HJM'     # 'HJM' or 'Cheyette'

# vol surface filename 
VOL_DATA_NAME = "estr_vol_surface.csv" 

# HJM config
HJM_VOL_METHOD = 'PCA'   # 'PCA' or 'PCA_SCALED'

# general
START_DATE = "2023-01-01"
PCA_F = 3.0
DT_STEPS = 1/252
SIM_YEARS = 1.0
SIM_PATHS = 4096
SIM_METHOD = 'sobol' # 'sobol' or 'standard'
SEED = 56

# calibration
CALIB_SWAPTION_EXP = 1.0   # expiry (years)
CALIB_SWAPTION_TEN = 5.0   # tenor (years)

# pricing
DELTA_BUMP = 0.0001
VOL_BUMP = 0.01 
OPT_EXP = 1.0
SWP_MAT = 5.0
STRIKE = 0.015

TESTS = True