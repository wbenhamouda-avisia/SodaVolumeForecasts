# src/soda_forecast/features/config.py

# Variables communes à tous les modèles
COMMON_FEATURES = [
    "avg_max_temp",
    "price_actual",
    "discount_in_percent",
    "month_sin",
    "month_cos",
]

# Features par target
FEATURES_MAP = {
    "market_share": COMMON_FEATURES + [
        "soda_penetration_rate",
        "lag_1_market_share",
        "lag_3_market_share",
        "lag_6_market_share",
        "lag_12_market_share",
    ],
    "volume": COMMON_FEATURES + [
        "soda_volume",
        "lag_1_volume",
        "lag_3_volume",
        "lag_6_volume",
        "lag_12_volume",
    ],
}

# Variables catégorielles
CATEGORICALS = ["agency", "sku"]