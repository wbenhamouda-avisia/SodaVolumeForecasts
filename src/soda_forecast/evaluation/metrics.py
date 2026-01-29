# metrics.py
 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from typing import Dict, Any

class Evaluator:
    """
    Classe utilitaire centralisant les logiques d'évaluation et de validation croisée temporelle.
    
    L'avantage d'utiliser une classe ici est de regrouper des méthodes statiques (outils) 
    qui partagent une logique commune de traitement des séries temporelles sans nécessiter 
    d'instanciation répétée.
    """

    @staticmethod
    def run_recursive_backtest(forecaster: Any, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute une simulation de prédiction itérative mois par mois (Backtesting récursif).
        """
        res_df = test_df.copy().sort_values(['date', 'timeseries'])
        months = sorted(res_df['date'].unique())
        
        # Détection automatique de la cible (market_share ou volume)
        target_col = forecaster.target 
        
        for i, month in enumerate(months):
            mask = (res_df['date'] == month)
            preds = forecaster.predict(res_df.loc[mask])
            
            if target_col == 'market_share':
                res_df.loc[mask, 'pred_market_share'] = preds
                res_df.loc[mask, 'pred_volume'] = preds * res_df.loc[mask, 'soda_volume']
                # On réinjecte le market_share pour le lag_1 du mois suivant
                val_to_map = res_df.loc[mask, 'pred_market_share']
                lag_name = 'lag_1_market_share'
            else:
                res_df.loc[mask, 'pred_volume'] = preds
                # On réinjecte le volume pour le lag_1 du mois suivant
                val_to_map = res_df.loc[mask, 'pred_volume']
                lag_name = 'lag_1_volume'
            
            # Mise à jour du LAG 1 pour le mois T+1 : Simulation d'un contexte réel
            if i < len(months) - 1:
                mapping = res_df.loc[mask].set_index('timeseries')[val_to_map.name]
                next_month_mask = (res_df['date'] == months[i+1])
                res_df.loc[next_month_mask, lag_name] = res_df.loc[next_month_mask, 'timeseries'].map(mapping)
                
        return res_df

        # Définition : 
        # Cette méthode simule le comportement du modèle en production. Pour chaque mois futur, 
        # elle utilise les prédictions du mois précédent pour mettre à jour les variables retardées (lags). 
        # C'est crucial pour évaluer la dégradation de la précision au fil du temps (horizon 1 à 4).

    @staticmethod
    def analyze_performance(comp_df: pd.DataFrame, nb_features: int) -> pd.DataFrame:
        """
        Calcule un ensemble de métriques statistiques pour chaque horizon de prévision.
        """
        horizons = sorted(comp_df['date'].unique())
        metrics = []
        for i, month in enumerate(horizons):
            sub = comp_df[comp_df['date'] == month]
            y_true, y_pred = sub['volume'], sub['pred_volume']
            
            # Calcul des métriques standards
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            wape = (np.abs(y_true - y_pred).sum() / y_true.sum()) * 100
            bias = (y_pred.sum() - y_true.sum()) / y_true.sum() * 100
            r2 = r2_score(y_true, y_pred)
            
            # R2 Ajusté pour tenir compte de la complexité du modèle
            n, k = len(sub), nb_features
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
            
            metrics.append({
                'Horizon': f"M+{i+1}",
                'MAE': mae,
                'RMSE': rmse,
                'WAPE%': wape,
                'BIAS%': bias,
                'R2_Adj': r2_adj
            })
        return pd.DataFrame(metrics)

        # Définition : 
        # Fournit une vue détaillée de la performance par horizon (M+1, M+2, etc.). 
        # L'utilisation du WAPE et du BIAS permet d'évaluer respectivement l'erreur relative pondérée 
        # et la tendance du modèle à sur/sous-estimer les volumes.
    
    @staticmethod
    def compute_final_ranks(all_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Détermine le modèle 'Champion' via un système de classement par points sur plusieurs métriques.
        """
        # 1. Compilation des moyennes par modèle
        summary_data = []
        for name, content in all_results.items():
            summary_data.append({
                'Modèle': name,
                'R2_Adj': content['stats']['R2_Adj'].mean(),
                'MAE': content['stats']['MAE'].mean(),
                'RMSE': content['stats']['RMSE'].mean(),
                'WAPE%': content['stats']['WAPE%'].mean(),
                'BIAS_abs': abs(content['stats']['BIAS%'].mean()) # Proximité de 0 souhaitée
            })
        
        df_metrics = pd.DataFrame(summary_data)
        df_scores = df_metrics[['Modèle']].copy()
        df_scores['Score_Total'] = 0

        # 2. Attribution des points (Pire = 0 points, Meilleur = N-1 points)
        metrics_to_rank = [
            ('R2_Adj', True), 
            ('MAE', False), 
            ('RMSE', False), 
            ('WAPE%', False), 
            ('BIAS_abs', False)
        ]

        for metric, higher_is_better in metrics_to_rank:
            df_metrics = df_metrics.sort_values(metric, ascending=higher_is_better)
            df_metrics[f'Points_{metric}'] = range(len(df_metrics))
            
            df_scores = df_scores.merge(df_metrics[['Modèle', f'Points_{metric}']], on='Modèle')
            df_scores['Score_Total'] += df_scores[f'Points_{metric}']

        return df_scores.sort_values('Score_Total', ascending=False)

        # Définition : 
        # Cette méthode implémente un système de vote robuste pour choisir le meilleur modèle. 
        # En ne se basant pas sur une seule métrique, on s'assure de choisir le modèle le plus équilibré 
        # (précis, stable et peu biaisé).