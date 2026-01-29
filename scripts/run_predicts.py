# scripts/run_predicts.py

import pandas as pd
from soda_forecast.pipeline.predictor import predict_batch
from soda_forecast.config import load_settings
from soda_forecast.data.loader import load_data
from soda_forecast.data.split import temporal_split
from soda_forecast.forecasting.generate_future_input import extend_timeseries

if __name__ == "__main__":
    """
    Script principal d'exécution des prédictions (Inférence Batch).
    
    Définition : 
    Ce script orchestre le chargement des données de test, l'extension temporelle 
    pour le futur, et l'appel au moteur de prédiction pour générer les volumes cibles.
    """
    # 1. Chargement de la configuration centralisée
    # Permet de récupérer les chemins de données et les paramètres de scénario (Réaliste/Oracle).
    settings = load_settings("configs/config.yaml")

    # 2. Préparation des données d'entrée
    # Chargement de l'historique complet pour servir de référence aux variables retardées (lags).
    df = load_data(settings.data_path)
    
    # Isolation du jeu de test (données récentes servant de base au forecast futur).
    _, test = temporal_split(df, settings.test_start_date)

    # 3. Extension de la structure temporelle
    # Création des lignes pour les 4 prochains mois (horizon=4) pour chaque couple agence/sku.
    future_df = extend_timeseries(test, horizon=4)

    # 4. Exécution de l'inférence
    # Appel de la logique de prédiction qui gère le feature engineering et les scénarios de prix.
    preds = predict_batch(
        input_df=future_df,
        model_path=settings.model_path,
        reference_df=df,              # Utilisation de l'historique pour le calcul des lags
        scenario=settings.pricing_scenario,
    )

    # 5. Formatage et Exportation des résultats
    # Sélection des colonnes essentielles pour le rapport final destiné aux équipes métier.
    preds = preds[
        ["agency", "sku", "timeseries", "date", "volume", "pred_volume"]
    ].sort_values(["agency", "sku", "date"])

    # Sauvegarde de l'artefact de prédiction dans le dossier de rapports.
    preds.to_csv("artifacts/reports/predictions.csv", index=False)

    print(f"✅ Prédictions générées avec succès dans artifacts/reports/predictions.csv")
    print(f"📈 Nombre de lignes prédites : {len(preds)}")