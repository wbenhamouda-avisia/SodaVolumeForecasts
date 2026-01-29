# features/engineering

import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les données brutes en variables explicatives (features) pour les modèles de ML.
    """
    # Création d'une copie pour éviter les effets de bord sur le DataFrame original
    df = df.copy()

    # 1. Calcul des cibles de base et indicateurs de marché
    # Calcul de la part de marché relative au volume total de soda
    df['market_share'] = df['volume'] / df['soda_volume'].replace(0, np.nan)
    # Calcul du taux de pénétration du soda par rapport à l'industrie globale
    df['soda_penetration_rate'] = df['soda_volume'] / df['industry_volume'].replace(0, np.nan)
    # Nettoyage des divisions par zéro
    df[['market_share', 'soda_penetration_rate']] = df[['market_share', 'soda_penetration_rate']].fillna(0)

    # Définition :
    # Cette étape transforme les volumes bruts en ratios. En production, prédire une Part de Marché 
    # est souvent plus stable que de prédire un volume brut car cela normalise les effets de saisonnalité globale du marché.

    # 2. Encodage cyclique de la temporalité
    # Transformation du mois en coordonnées sinus/cosinus pour conserver la proximité entre Décembre (12) et Janvier (1)
    month = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Définition :
    # L'encodage cyclique permet au modèle de comprendre que le temps est une boucle. 
    # Sans cela, un modèle de type arbre (XGBoost/LGBM) traiterait '12' et '1' comme étant aux opposés, 
    # perdant la continuité saisonnière.

    # 3. Tri pour les séries temporelles
    # Crucial pour que les opérations de 'shift' (lags) correspondent aux bonnes périodes
    df = df.sort_values(['timeseries', 'date'])

    # 4. GÉNÉRATION DES LAGS (Variables retardées)
    # On crée des fenêtres historiques (1, 3, 6, 12 mois) pour donner de la mémoire au modèle
    for l in [1, 3, 6, 12]:
            # Lags pour le modèle basé sur la Part de Marché (Market Share)
            df[f'lag_{l}_market_share'] = df.groupby('timeseries')['market_share'].shift(l)
            # Lags pour le modèle basé sur le Volume direct
            df[f'lag_{l}_volume'] = df.groupby('timeseries')['volume'].shift(l)

    # Définition :
    # Les 'lags' injectent l'historique passé dans la ligne actuelle. Le lag_12 est particulièrement 
    # puissant pour capturer la saisonnalité annuelle (ventes du même mois l'année précédente).

    # Remplissage des NaNs (valeurs manquantes créées par les lags en début de série)
    # On remplit par 0 pour conserver toutes les lignes lors de l'entraînement
    df = df.fillna(0)

    return df.reset_index(drop=True)

    # Définition globale :
    # Cette fonction est le pont entre la donnée brute et le modèle. En production sur GCP, 
    # elle peut être intégrée dans un composant "Dataflow" ou "Vertex AI Feature Store" 
    # pour garantir que les mêmes calculs sont appliqués à l'entraînement et à l'inférence (API).