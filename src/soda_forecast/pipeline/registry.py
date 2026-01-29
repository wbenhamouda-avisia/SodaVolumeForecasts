# soda_forecast/pipeline/registry.py

import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

class ModelRegistry:
    """
    Gestion centralisée des artefacts modèles (sauvegarde, chargement et métadonnées).
    
    L'utilisation d'une classe ici permet de standardiser le format de stockage des modèles 
    pour qu'ils soient exploitables par l'API et les pipelines de production.
    """

    @staticmethod
    def save(model: Any, path: Path, metadata: Dict[str, Any]) -> None:
        """
        Sauvegarde le modèle entraîné et ses métadonnées associées.
        """
        # S'assurer que le dossier de destination existe (ex: dossier 'models/')
        path.parent.mkdir(parents=True, exist_ok=True)

        # 1️⃣ Sauvegarde de l'objet modèle (incluant ses encodeurs et hyperparamètres)
        # Joblib est préféré à pickle pour les modèles contenant de gros arrays NumPy (XGB/LGBM).
        joblib.dump(model, path)

        # 2️⃣ Sauvegarde des métadonnées au format JSON (Traçabilité)
        # On enregistre les performances, les features utilisées et la date d'entraînement.
        meta_path = path.with_name(f"{path.stem}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Définition :
        # En production, sauvegarder les métadonnées avec le modèle est crucial pour 
        # le "Model Lineage" (savoir exactement quel dataset a produit quel modèle).

    @staticmethod
    def load(path: str) -> Any:
        """
        Charge un modèle sauvegardé pour l'utiliser lors de l'inférence.
        """
        # Définition :
        # Cette méthode est appelée par l'API ou le Predictor. Elle restaure l'objet 
        # Forecaster complet, prêt à recevoir des données via sa méthode .predict().
        return joblib.load(path)