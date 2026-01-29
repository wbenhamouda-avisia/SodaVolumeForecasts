# scripts/run_train.py

from soda_forecast.pipeline.trainer import train_pipeline

if __name__ == "__main__":
    """
    Point d'entrée principal pour l'entraînement du pipeline de prévision.
    
    Définition : 
    Ce script exécute la fonction 'train_pipeline' en lui passant le fichier de 
    configuration YAML. Il automatise la recherche du meilleur modèle (Champion) 
    parmi plusieurs algorithmes (XGBoost, LightGBM, etc.).
    """
    
    # Appel du chef d'orchestre de l'entraînement. 
    # Le chemin "configs/config.yaml" contient tous les hyperparamètres et réglages 
    # nécessaires à l'industrialisation sans modification du code source.
    train_pipeline("configs/config.yaml")

    # Définition :
    # En production, ce script est celui qui est appelé par un orchestrateur de tâches.
    # Son exécution garantit la création des artefacts (.pkl) et des rapports 
    # de performance (model_ranking.csv) indispensables au monitoring.