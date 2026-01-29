=============================================================================== SODA SALES FORECASTING - DOCUMENTATION TECHNIQUE (v1.0)

Ce projet propose une solution de pointe pour la prévision des ventes mensuelles par couple Magasin (Agency) et Produit (SKU). L'architecture a été pensée pour passer d'un prototype de Data Science a une application "Production-Ready".

** STATUT DU PROJET : Mode "Volume" valide et juste.**


# THEORIE DE L'ESTIMATION & MODELISATION

## A. Strategie de Partitionnement (Data Split) : 
Pour garantir la robustesse et eviter le "look-ahead bias", nous utilisons un decoupage chronologique strict :
Train Set : Donnees historiques servant a l'apprentissage.
Validation Set : 4 derniers mois du Train Set (M-4 a M) pour le backtest.
Test Set : Donnees isolees pour l'evaluation de la performance reelle.

## B. Cibles et Fiabilite de l'Estimation : 
Le projet supporte deux methodologies de prediction :
Target "Volume" (Operationnel) : Actuellement la methode la plus fiable. Les tests confirment que les predictions sont precises et pretes a l'usage.
Target "Market Share" (Experimental) : Approche theoriquement superieure pour normaliser le marche, mais integration API en cours de finalisation.

## C. Optimisation & Selection :
Modeles : XGBoost, LightGBM, Random Forest, ElasticNet et PanelOLS.
Optimisation : Recherche bayesienne via Optuna (Fonction Tweedie).
Champion : Systeme de rangs (WAPE, Biais et R2 Ajuste) pour isoler automatiquement le meilleur modele.


# ARBORESCENCE & RESPONSABILITES
.
├── src/soda_forecast/        # 🧠 COEUR LOGIQUE (Bibliothèque packagée)
│   ├── data/                 # Chargement (loader.py) et split temporel (split.py)
│   ├── features/             # Feature engineering (engineering.py) et mapping (config.py)
│   ├── models/               # Classes Forecasters (forecasters.py) uniformisées
│   ├── pipeline/             # Orchestration : Entraînement (trainer.py) et Inférence (predictor.py)
│   ├── evaluation/           # Moteur de calcul des métriques de précision (metrics.py)
│   ├── forecasting/          # Préparation des scénarios et squelette futur (preparation.py)
│   └── visualization/        # Génération et sauvegarde automatique des plots (plots.py)
├── scripts/                  # 🚀 POINTS D'ENTRÉE (Exécution terminal)
│   ├── run_train.py          # Lance l'entraînement et la sélection du champion
│   └── run_predicts.py       # Génère les prédictions sur l'horizon futur
├── artifacts/                # 📦 OUTPUTS (Modèles .pkl, rapports .csv, graphiques .png/.html)
├── data/                     # 💾 DONNÉES (raw/ pour l'historique, predict/ pour le futur)
├── configs/                  # ⚙️ CONFIGURATION (config.yaml : la seule vérité du projet)
└── streamit_app.py           # 📊 UI (Interface de visualisation Streamlit)


# GUIDE D'EXECUTION (TERMINAL)

Pour garantir la resolution des modules, utilisez le PYTHONPATH :
Entrainement complet : PYTHONPATH=src python scripts/run_train.py
Generation des predictions : PYTHONPATH=src python scripts/run_predicts.py
Note : Les resultats (CSV et graphiques) sont generes dans artifacts/reports/.


# LIMITES ACTUELLES & EVOLUTIONS (ROADMAP)

Visualisation : Utiliser le module visualization pour enregistrer les graphiques dans artifacts/ pour audit rapide.
Optimisation : Nettoyage du requirements.txt (suppression packages inutiles).
Bug d'Horizon : Correction de la boucle recursive (actuellement 8 mois au lieu de 4).
Stabilisation : Finaliser la re-injection des lags pour le mode Market Share.
Validation : Etendre les tests sur l'annee complete (Janvier a Aout).
VISION ARCHITECTURE GCP (INDUSTRIALISATION)
Pour operer cette solution a l'echelle sur Google Cloud Platform :
Data : Migration vers BigQuery.
Entrainement : Jobs containerises sur Vertex AI Training.
Pipelines : Orchestration via Vertex AI Pipelines (Kubeflow).
Inference : API exposee via Cloud Run.
Monitoring : Surveillance du drift avec Vertex AI Model Monitoring.

=============================================================================== AUTEUR : Solution de Forecast Soda - v1.0