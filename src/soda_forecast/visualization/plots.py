# plots.py

import plotly.graph_objects as go
import pandas as pd

def plot_full_comparison(df_train: pd.DataFrame, df_forecast: pd.DataFrame, target_agency: str, target_sku: str):
    """
    Génère un graphique interactif comparant l'historique de ventes, les valeurs réelles de test et les prévisions.
    """
    # 1. Extraction de l'historique (Données d'entraînement)
    # Filtrage sur le couple spécifique agence/produit pour isoler la série temporelle
    mask_hist = (df_train['agency'] == target_agency) & (df_train['sku'] == target_sku)
    df_hist = df_train.loc[mask_hist, ['date', 'volume']].sort_values('date')

    # 2. Extraction du Forecast et du Réel (Données de test)
    # Récupération des prédictions générées par le modèle et des observations réelles correspondantes
    mask_fore = (df_forecast['agency'] == target_agency) & (df_forecast['sku'] == target_sku)
    df_fore = df_forecast.loc[mask_fore, ['date', 'volume', 'pred_volume']].sort_values('date')
    
    # 3. Préparation de la jonction graphique
    # On récupère le dernier point historique pour assurer une continuité visuelle entre le passé et le futur
    last_point = df_hist.iloc[[-1]].copy()
    
    # Création de DataFrames de liaison pour éviter les ruptures de ligne sur le graphique
    df_pred_conn = pd.concat([
        last_point.rename(columns={'volume': 'pred_volume'})[['date', 'pred_volume']], 
        df_fore[['date', 'pred_volume']]
    ])
    
    df_real_conn = pd.concat([
        last_point[['date', 'volume']], 
        df_fore[['date', 'volume']]
    ])

    # Définition : 
    # Cette étape de "jonction" est cruciale pour l'UX (expérience utilisateur). Elle permet de voir 
    # exactement où s'arrête l'historique connu et où commence la prédiction du modèle.

    # 4. Construction du graphique interactif Plotly
    fig = go.Figure()

    # --- TRACE 1 : HISTORIQUE (BLEU) ---
    # Représente le passé réel utilisé pour entraîner le modèle
    fig.add_trace(go.Scatter(
        x=df_hist['date'], y=df_hist['volume'],
        mode='lines', name='Historique Réel',
        line=dict(color='#1f77b4', width=2)
    ))

    # --- TRACE 2 : RÉALITÉ OBSERVÉE SUR TEST (VERT) ---
    # Représente ce qui s'est réellement passé pendant la période de prévision
    fig.add_trace(go.Scatter(
        x=df_real_conn['date'], y=df_real_conn['volume'],
        mode='lines+markers', name='Vrai Volume (Observé)',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=4)
    ))

    # --- TRACE 3 : PRÉDICTION SUR TEST (ORANGE POINTILLÉ) ---
    # Représente la performance du modèle champion
    fig.add_trace(go.Scatter(
        x=df_pred_conn['date'], y=df_pred_conn['pred_volume'],
        mode='lines+markers', name='Prédiction (Forecast)',
        line=dict(color='#ff7f0e', width=3, dash='dot'),
        marker=dict(size=6, symbol='diamond')
    ))

    # 5. Mise en forme, Design et Annotations
    fig.update_layout(
        title=f"<b>Analyse de Performance : Réel vs Forecast</b><br>SKU: {target_sku} | Agence: {target_agency}",
        xaxis_title="Chronologie",
        yaxis_title="Volume",
        template="plotly_white",
        hovermode="x unified", # Affiche toutes les valeurs au survol pour une comparaison facilitée
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=[
            # Ligne de séparation verticale entre l'historique (Train) et le futur (Test)
            dict(type="line", x0=df_hist['date'].max(), x1=df_hist['date'].max(),
                 y0=0, y1=1, yref="paper", line=dict(color="black", width=1, dash="dash"))
        ]
    )

    fig.show()

    # Définition : 
    # Cette visualisation est l'outil principal de validation pour le métier. En production, 
    # ce type de graphique peut être intégré dans un dashboard Streamlit ou Looker (GCP) 
    # pour monitorer la santé des prévisions au quotidien.