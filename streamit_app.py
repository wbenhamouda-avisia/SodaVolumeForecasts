# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from project.artifacts.data.data_loader import load_data
from project.src.features.engineering import build_features
from project.src.evaluation.metrics import Evaluator
from project.src.config import FEATURES_MAP, CATEGORICALS, VAL_SIZE
from project.src.forecasters.models import (PanelEconometricForecaster, LGBMForecaster, 
                    XGBForecaster, RFForecaster, ElasticNetForecaster)
from project.src.models.preparation import get_prepared_data

# --- INITIALISATION DU SESSION STATE ---
# Crucial pour que les données persistent quand on clique sur les onglets
if 'all_results' not in st.session_state:
    st.session_state.all_results = None
if 'final_forecast_df' not in st.session_state:
    st.session_state.final_forecast_df = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'active_scenario' not in st.session_state:
    st.session_state.active_scenario = None

st.set_page_config(page_title="Carrefour Soda Forecast", layout="wide")
st.title("🥤 Soda Demand Forecasting")

# --- ÉTAPE 1 : CONFIGURATION DYNAMIQUE ---
st.sidebar.header("🎯 Configuration du Modèle")
target_type = st.sidebar.radio("Choisir la Target :", ["market_share", "volume"])

st.sidebar.subheader("Hypothèses de Forecast")
pricing_scenario = st.sidebar.radio(
    "Scénario de Prix :",
    ["Réaliste (Dernier prix connu)", "Oracle (Prix réels planifiés)"]
)

# --- ÉTAPE 2 : CHARGEMENT DES DONNÉES ---
@st.cache_data
def get_data():
    # Remplacez par votre chemin relatif ou absolu correct
    path = "/Users/waelbenhamouda/Documents/AVISIA/Use_cases/Use_case_Carrefour/ds_assortiment_dataset.csv" 
    data = load_data(path)
    return build_features(data)

df = get_data()

# Splits Temporels
train_full = df[df['date'] < '2017-09-01']
test_final = df[df['date'] >= '2017-09-01']

dates = sorted(train_full['date'].unique())
split_date = dates[-VAL_SIZE]
train_sub = train_full[train_full['date'] < split_date]
val_sub = train_full[train_full['date'] >= split_date]

# --- SIDEBAR PARAMÈTRES D'AFFICHAGE ---
st.sidebar.divider()
sel_agency = st.sidebar.selectbox("Agence", df['agency'].unique())
sel_sku = st.sidebar.selectbox("SKU", df[df['agency']==sel_agency]['sku'].unique())

# --- ÉTAPE 3 : INITIALISATION DES MODÈLES ---
FEATURES = FEATURES_MAP[target_type]

models_to_test = {
    "Panel_Econometrics": PanelEconometricForecaster(FEATURES, CATEGORICALS),
    "LGBM_Tweedie": LGBMForecaster(FEATURES, CATEGORICALS),
    "XGBoost_Tweedie": XGBForecaster(FEATURES, CATEGORICALS),
    "RandomForest": RFForecaster(FEATURES, CATEGORICALS),
    "ElasticNet_Baseline": ElasticNetForecaster(FEATURES, CATEGORICALS)
}

# --- ÉTAPE 4 : CALCULS ---
if st.sidebar.button("🚀 Lancer l'Optimisation et le Forecast"):
    st.session_state.all_results = {} 
    st.session_state.active_scenario = pricing_scenario # On mémorise le scénario utilisé
    
    with st.status(f"Phase 1 : Sélection (Mode: {pricing_scenario})") as status:
        # Préparation des données de validation selon le scénario
        val_prepared = get_prepared_data(val_sub, train_sub, pricing_scenario)

        for name, model in models_to_test.items():
            model.features = FEATURES
            model.target = target_type
            
            st.write(f"Entraînement et Tuning de {name}...")
            # PanelEconometrics ne gère pas l'argument 'tune' dans certaines versions, 
            # mais nous avons harmonisé la signature dans models.py
            model.train(train_sub, val_prepared, tune=(name != "Panel_Econometrics"))
            
            # Backtest récursif
            comp_val = Evaluator.run_recursive_backtest(model, val_prepared)
            
            # Sauvegarde des stats en session
            st.session_state.all_results[name] = {
                "stats": Evaluator.analyze_performance(comp_val, len(FEATURES))
            }

        # Détermination du Champion Global
        rank_summary = Evaluator.compute_final_ranks(st.session_state.all_results)
        best_name = rank_summary.iloc[0]['Modèle']
        st.session_state.best_model_name = best_name
        
        st.write(f"🏆 Champion Global : {best_name}")

        # Phase Finale : Ré-entraînement et Forecast
        champion_model = models_to_test[best_name]
        test_final_prepared = get_prepared_data(test_final, train_full, pricing_scenario)
        
        # On entraîne sur tout le train disponible (Train + Validation préparée)
        full_train_final = pd.concat([train_sub, val_prepared])
        champion_model.train(full_train_final, full_train_final)
        
        st.session_state.final_forecast_df = Evaluator.run_recursive_backtest(champion_model, test_final_prepared)
        status.update(label="Forecast terminé !", state="complete")

# --- ÉTAPE 5 : AFFICHAGE DES ONGLETS ---
tab1, tab2, tab3 = st.tabs(["📈 Prévisions", "🔬 Performances Modèles", "📊 Statistiques"])

with tab1:
    if st.session_state.final_forecast_df is not None:
        st.subheader(f"Résultats pour {sel_sku} à {sel_agency}")
        
        f_df = st.session_state.final_forecast_df
        mask_test = (f_df['agency'] == sel_agency) & (f_df['sku'] == sel_sku)
        df_plot_test = f_df[mask_test].sort_values('date')
        
        mask_train = (train_full['agency'] == sel_agency) & (train_full['sku'] == sel_sku)
        df_plot_train = train_full[mask_train].sort_values('date')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot_train['date'], y=df_plot_train['volume'], name="Historique"))
        fig.add_trace(go.Scatter(x=df_plot_test['date'], y=df_plot_test['volume'], name="Réel (Test)", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df_plot_test['date'], y=df_plot_test['pred_volume'], name="Forecast", line=dict(dash='dot', color='orange')))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Utilisez la barre latérale pour lancer le calcul.")

with tab2:
    if st.session_state.all_results is not None:
        st.header("Analyse de la Performance")
        
        # Rappel du contexte
        scenario = st.session_state.active_scenario
        if scenario == "Oracle (Prix réels planifiés)":
            st.success(f"✅ **Scénario Oracle** : Connaissance parfaite des prix futurs.")
        else:
            st.warning(f"⚠️ **Scénario Réaliste** : Prix futurs estimés via l'historique.")

        st.info(f"🏆 **Champion sélectionné : {st.session_state.best_model_name}**")
        
        # Métriques Globales sur le Test Set
        f_df = st.session_state.final_forecast_df
        wape_global = (np.abs(f_df['volume'] - f_df['pred_volume']).sum() / f_df['volume'].sum()) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("WAPE Global (Test)", f"{wape_global:.2f}%")
        col2.metric("Précision (100-WAPE)", f"{100 - wape_global:.2f}%")
        
        st.divider()
        st.subheader("Classement comparatif (Validation)")
        ranks = Evaluator.compute_final_ranks(st.session_state.all_results)
        st.dataframe(ranks, use_container_width=True)
    else:
        st.info("Aucune performance à afficher. Lancez le forecast.")

with tab3:
    st.header("📊 Analyses Exploratoires (EDA)")
    
    # On s'assure que les graphiques utilisent le dataframe complet
    df_sku_date = df.groupby(['sku', 'date'])[['price_actual', 'market_share']].mean().reset_index()
    df_agency_date = df.groupby(['agency', 'date'])[['avg_max_temp', 'soda_penetration_rate']].mean().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(df_sku_date, x='date', y='market_share', color='sku', title='Parts de Marché'), use_container_width=True)
        st.plotly_chart(px.line(df_agency_date, x='date', y='avg_max_temp', color='agency', title='Températures'), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(df_sku_date, x='date', y='price_actual', color='sku', title='Évolution des Prix'), use_container_width=True)
        st.plotly_chart(px.line(df_agency_date, x='date', y='soda_penetration_rate', color='agency', title='Pénétration Soda'), use_container_width=True)

    st.subheader("Matrice de Corrélation")
    corr = df[['volume', 'avg_max_temp', 'price_actual', 'discount_in_percent', 'soda_volume']].corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, title="Corrélation Variables/Volume"), use_container_width=True)