"""SDWPF / KDD Cup timeline shared with sdwpf_weather_v2 (Tmstamp from 2020-05-01)."""

SDWPF_V1_ANCHOR = "2020-05-01"

ERA5_EXTRA_COLS: tuple[str, ...] = (
    "T2m",
    "Sp",
    "RelH",
    "Wspd_w",
    "Wdir_w",
    "Tp",
)

# 10-minute grid: 6 steps/hour, 144 steps/day
STEPS_PER_HOUR = 6
STEPS_PER_DAY = 144

# Retard météo (ERA5) en mode ``time_meteo_only`` : t-1 … t-k (pas de 10 min)
DEFAULT_METEO_MAX_LAG = 12

# Colonnes dérivées (physique / encodage) ajoutées en mode météo-only
METEO_WSPD_CUBE_COL = "Wspd_w_cube"
METEO_WDIR_SIN_COL = "Wdir_w_sin"
METEO_WDIR_COS_COL = "Wdir_w_cos"

# Modèle pooled : identifiant numérique turbine dans X
POOL_TURB_ID_COL = "turb_id"

# Reproductibilité (XGBoost et tout estimateur futur qui expose random_state)
XGBOOST_RANDOM_STATE = 42
