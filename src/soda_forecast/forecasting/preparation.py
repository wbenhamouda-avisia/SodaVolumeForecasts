# forecasting/preparation.py

import pandas as pd
import numpy as np

def get_prepared_data(df_target: pd.DataFrame, df_reference: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Prépare les données exogènes pour le futur en appliquant des stratégies de proxy (Météo/Prix).
    
    Cette fonction est essentielle pour l'inférence : elle remplace les données futures "parfaites" 
    par des estimations réalistes basées sur l'historique.
    """
    # Création d'une copie pour éviter de modifier le DataFrame original (Side Effect)
    df = df_target.copy()
    
    # --- 1. VARIABLES TOUJOURS "RÉALISTES" (Météo & Marché) ---
    # Stratégie : On utilise la moyenne historique par mois comme proxy pour la météo future.
    weather_proxy = df_reference.groupby('month')['avg_max_temp'].mean()
    
    # Stratégie : On récupère la dernière valeur connue pour chaque série temporelle.
    last_vals = df_reference.sort_values('date').groupby('timeseries').last()

    # Application des proxys météo basés sur le mois calendaire
    df['avg_max_temp'] = df['month'].map(weather_proxy)
    
    # Application du dernier taux de pénétration connu pour chaque couple agence/sku
    df['soda_penetration_rate'] = df['timeseries'].map(last_vals['soda_penetration_rate'])
    
    # Sécurité NaNs : Remplissage par la moyenne globale si une série est totalement nouvelle
    df['soda_penetration_rate'] = df['soda_penetration_rate'].fillna(df_reference['soda_penetration_rate'].mean())

    # Définition :
    # En production, cette étape garantit que le modèle ne "triche" pas en utilisant une météo exacte 
    # qu'il n'aurait pas en temps réel. Cela permet d'obtenir une mesure de performance fiable.

    # --- 2. VARIABLES DE PRIX (Selon le scénario choisi) ---
    if scenario == "Réaliste (Dernier prix connu)":
        # Mode Simulation : On fait l'hypothèse que les prix futurs sont identiques aux derniers prix observés.
        df['price_actual'] = df['timeseries'].map(last_vals['price_actual'])
        df['discount_in_percent'] = df['timeseries'].map(last_vals['discount_in_percent'])
        
        # Remplissage par la moyenne en cas de données manquantes
        df['price_actual'] = df['price_actual'].fillna(df_reference['price_actual'].mean())
    else:
        # Mode "Oracle Prix" : On conserve les données de prix déjà présentes dans df_target.
        # Utile pour simuler l'impact d'un changement de prix planifié (Plan de Promotion).
        pass
    
    return df

    # Définition globale :
    # Cette fonction implémente le "Feature Alignment". Elle assure que les variables exogènes 
    # injectées dans le modèle lors de la prédiction à 4 mois respectent les contraintes 
    # opérationnelles de disponibilité de la donnée.