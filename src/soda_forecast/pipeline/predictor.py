# soda_forecast/pipeline/predictor.py

import pandas as pd
from soda_forecast.pipeline.registry import ModelRegistry
from soda_forecast.features.engineering import build_features
from soda_forecast.forecasting.preparation import get_prepared_data

def predict_batch(
    input_df: pd.DataFrame,
    model_path: str,
    reference_df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """
    Exécute le pipeline complet de prédiction sur un lot de données (Batch Inference).
    """

    # 1. Chargement de l'artefact du modèle
    # Le ModelRegistry garantit que le modèle chargé (.pkl) contient ses encodeurs de catégories.
    model = ModelRegistry.load(model_path)

    # 2. Feature Engineering
    # Application des mêmes transformations (lags, sinus/cosinus) que lors de l'entraînement.
    df = build_features(input_df)
    
    # 3. Enrichissement via Référence
    # ✅ IMPORTANT : On doit calculer les features sur le référentiel historique pour obtenir les lags
    # nécessaires à la prédiction des premières dates du futur.
    ref = build_features(reference_df)

    # 4. Alignement des Scénarios
    # Application des proxys (Météo/Prix) selon le choix "Réaliste" ou "Oracle".
    df_prepared = get_prepared_data(df, ref, scenario)

    # 5. Inférence (Prédiction)
    # Appel de la méthode .predict() héritée de BaseForecaster.
    preds = model.predict(df_prepared)
    
    # Ajout des résultats au DataFrame final
    df_prepared["pred_volume"] = preds

    return df_prepared

    # Définition :
    # Cette fonction est le point d'entrée unique pour toute prédiction de masse. 
    # Elle garantit la "Training-Serving Consistency" : les données sont traitées 
    # exactement de la même manière que pendant la phase de R&D.