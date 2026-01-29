# soda_forecast/pipeline/trainer.py

import pandas as pd
from soda_forecast.config import load_settings, ensure_dirs
from soda_forecast.data.loader import load_data
from soda_forecast.data.split import temporal_split, temporal_validation_split
from soda_forecast.features.engineering import build_features
from soda_forecast.models.forecasters import (
    PanelEconometricForecaster,
    LGBMForecaster,
    XGBForecaster,
    RFForecaster,
    ElasticNetForecaster,
)
from soda_forecast.evaluation.metrics import Evaluator
from soda_forecast.forecasting.preparation import get_prepared_data
from soda_forecast.pipeline.registry import ModelRegistry
from soda_forecast.features.config import FEATURES_MAP, CATEGORICALS

def train_pipeline(config_path: str):
    """
    Exécute le cycle de vie complet de l'entraînement : du chargement des données à la sauvegarde du modèle champion.
    """
    # ------------------ INITIALISATION ------------------
    # Chargement des paramètres depuis le fichier YAML et création des dossiers (models/, metrics/)
    settings = load_settings(config_path)
    ensure_dirs(settings)

    # ------------------ CHARGEMENT & FEATURES ------------------
    # Nettoyage des données et calcul des variables explicatives (lags, indicateurs marché)
    df = load_data(settings.data_path)
    df = build_features(df)

    # ------------------ SPLITS TEMPORELS ------------------
    # Séparation stricte pour garantir l'absence de fuite de données (Train / Test / Validation)
    train, test = temporal_split(df, settings.test_start_date)
    train_sub, val_sub, _ = temporal_validation_split(train, settings.val_size)

    target = settings.target
    scenario = settings.pricing_scenario
    features = FEATURES_MAP[target]

    # Préparation réaliste des données de validation (proxys météo/prix)
    val_prepared = get_prepared_data(val_sub, train_sub, scenario)

    # ------------------ INITIALISATION DES MODÈLES ------------------
    # Instanciation des différentes familles d'algorithmes (Économétrie, Boosting, Forêt, Linéaire)
    models = {
        "Panel_Econometrics": PanelEconometricForecaster(features, CATEGORICALS, target),
        "LGBM_Tweedie": LGBMForecaster(features, CATEGORICALS, target),
        "XGB_Tweedie": XGBForecaster(features, CATEGORICALS, target),
        "RandomForest": RFForecaster(features, CATEGORICALS, target),
        "ElasticNet": ElasticNetForecaster(features, CATEGORICALS, target),
    }

    all_results = {}

    # ------------------ ENTRAÎNEMENT & VALIDATION CROISÉE ------------------
    # Boucle d'évaluation systématique pour comparer les performances de chaque modèle
    for name, model in models.items():
        # Entraînement avec ou sans optimisation d'hyperparamètres (Optuna)
        model.train(train_sub, val_prepared, tune=settings.tune_models)
        
        # Simulation de prédiction récursive sur l'horizon de validation (Backtest)
        comp = Evaluator.run_recursive_backtest(model, val_prepared)
        
        # Calcul des métriques de précision (WAPE, RMSE, Biais) par horizon
        stats = Evaluator.analyze_performance(comp, len(features))
        all_results[name] = {"stats": stats}

    # ------------------ SÉLECTION DU CHAMPION ------------------
    # Application du système de classement par points sur l'ensemble des métriques
    ranking = Evaluator.compute_final_ranks(all_results)
    best_name = ranking.iloc[0]["Modèle"]
    best_model = models[best_name]

    # Définition :
    # Cette étape automatise le choix du modèle le plus équilibré, évitant ainsi 
    # une sélection subjective basée sur une seule mesure de performance.

    # ------------------ ENTRAÎNEMENT FINAL ------------------
    # Une fois le meilleur algorithme identifié, on le ré-entraîne sur l'ensemble des données disponibles 
    # (Apprentissage + Validation) pour maximiser sa capacité de généralisation.
    full_train = pd.concat([train_sub, val_prepared])
    best_model.train(full_train, full_train)

    # ------------------ SAUVEGARDE DES ARTEFACTS ------------------
    # Enregistrement du modèle champion et de son contexte d'exécution (métadonnées)
    ModelRegistry.save(
        best_model,
        settings.model_path,
        metadata={
            "best_model": best_name,
            "target": target,
            "features": features,
            "scenario": scenario,
            "training_date": str(pd.Timestamp.now())
        },
    )

    # Exportation des rapports de performance pour audit ultérieur
    ranking.to_csv(settings.metrics_dir / "model_ranking.csv", index=False)

    print(f"✅ Training terminé. Modèle sauvegardé : {best_name}")

    # Définition globale :
    # Ce pipeline est le socle de l'industrialisation. Sur Google Cloud, il peut être 
    # encapsulé dans un Job Vertex AI déclenché automatiquement chaque mois.