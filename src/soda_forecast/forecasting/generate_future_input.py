# forecasting/generate_future_input.py

import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import Optional

from soda_forecast.config import load_settings
from soda_forecast.data.loader import load_data
from soda_forecast.data.split import temporal_split


def generate_future_input(config_path: str, horizon: int = 4) -> None:
    """
    Génère un fichier CSV contenant les structures vides pour les prévisions futures.
    """
    # Chargement de la configuration centralisée (.yaml)
    settings = load_settings(config_path)

    # 1. Charger les données historiques pour connaître les références (SKU/Agency)
    df = load_data(settings.data_path)

    # 2. Reproduire le split TRAIN / TEST du pipeline pour s'aligner sur la logique d'évaluation
    train, test = temporal_split(df, settings.test_start_date)

    # 3. Déterminer le point de départ : la dernière date connue dans le jeu de test
    last_test_date = test["date"].max()

    # 4. Génération de la liste des dates futures (ex: M+1 à M+4)
    future_dates = [
        last_test_date + relativedelta(months=i)
        for i in range(1, horizon + 1)
    ]

    # 5. Extraction de tous les couples uniques Agency / SKU existants
    keys = df[["agency", "sku"]].drop_duplicates()

    # 6. Produit cartésien (Cross Join) entre les couples Agency/SKU et les dates futures
    # Cela crée une ligne pour chaque produit dans chaque magasin pour chaque mois futur
    future_df = (
        keys.assign(key=1)
        .merge(
            pd.DataFrame({"date": future_dates, "key": 1}),
            on="key",
        )
        .drop(columns="key")
    )

    # Définition :
    # Cette étape prépare le "template" de prédiction. En production, cela garantit que 
    # le modèle ne manquera aucun couple produit/magasin lors de la génération du forecast.

    # 7. Placeholder explicite pour la cible à prédire
    future_df["volume"] = None

    # 8. Sauvegarde du fichier prêt pour l'inférence
    output_path = "data/predict/future_input.csv"
    future_df.to_csv(output_path, index=False)

    print("✅ Future input généré")
    print(f"📅 Dernière date test : {last_test_date.date()}")
    print(f"🔮 Horizon forecast : {horizon} mois")
    print(f"📁 Fichier : {output_path}")

    # Définition :
    # Ce fichier généré sera ensuite envoyé à l'API ou au script d'inférence. Sur GCP, 
    # ce processus peut être automatisé par une Cloud Function déclenchée chaque mois.


def extend_timeseries(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """
    Prolonge les séries temporelles existantes en répétant la dernière observation connue.
    """
    out = []
    # Itération sur chaque série individuelle (couple Agency_SKU)
    for ts, g in df.groupby("timeseries"):
        g = g.sort_values("date")
        last_row = g.iloc[-1] # Récupération du dernier état connu (prix, météo, etc.)

        # Création des N nouvelles lignes futures
        for i in range(1, horizon + 1):
            new_row = last_row.copy()
            new_row["date"] = last_row["date"] + relativedelta(months=i)
            out.append(new_row)

    # Fusion des données historiques et des extensions vides
    return pd.concat([df, pd.DataFrame(out)], ignore_index=True)

    # Définition :
    # Cette fonction est utile pour l'inférence récursive. Elle permet de maintenir 
    # la structure des données (lags) en attendant que le modèle remplisse les prédictions.


if __name__ == "__main__":
    # Point d'entrée pour l'exécution manuelle du script
    generate_future_input("configs/config.yaml", horizon=4)