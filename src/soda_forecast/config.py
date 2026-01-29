# soda_forecast/config.py

import yaml
from dataclasses import dataclass
from pathlib import Path

# Définition dynamique de la racine du projet pour garantir la portabilité du code
# Quel que soit l'endroit où le script est lancé (local ou serveur GCP).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Settings:
    """
    Objet de configuration structuré utilisant des dataclasses pour un accès typé aux paramètres.
    
    L'avantage d'une classe ici est de transformer un dictionnaire YAML brut en un objet 
    facile à manipuler avec auto-complétion, réduisant les erreurs de frappe sur les clés.
    """
    raw: dict

    # ----------- DATA -----------
    @property
    def data_path(self) -> str:
        """Chemin vers le dataset source (CSV)."""
        return self.raw["data"]["input_path"]

    # ----------- SPLIT -----------
    @property
    def test_start_date(self) -> str:
        """Date pivot pour séparer l'historique du jeu de test."""
        return self.raw["split"]["test_start_date"]

    @property
    def val_size(self) -> int:
        """Nombre de mois à isoler pour la validation interne (Backtesting)."""
        return self.raw["split"]["val_size"]

    # ----------- TARGET / SCENARIO -----------
    @property
    def pricing_scenario(self) -> str:
        """Type de scénario utilisé pour les prix futurs (Oracle ou Réaliste)."""
        return self.raw["target"]["pricing_scenario"]

    @property
    def target(self):
        """Variable cible à prédire (market_share ou volume)."""
        return self.raw["target"]["default"]

    # ----------- ARTEFACTS -----------
    @property
    def model_path(self) -> Path:
        """Chemin absolu vers le fichier .pkl du modèle champion."""
        return PROJECT_ROOT / self.raw["artifacts"]["model_path"]

    @property
    def metrics_dir(self) -> Path:
        """Dossier de stockage des rapports de performance (CSV, JSON)."""
        return PROJECT_ROOT / self.raw["artifacts"]["metrics_dir"]
    
    @property
    def tune_models(self) -> bool:
        """Indicateur pour activer ou non l'optimisation des hyperparamètres (Optuna)."""
        return self.raw["training"]["tune_models"]


def load_settings(path: str) -> Settings:
    """
    Charge le fichier YAML et l'instancie sous forme d'objet Settings.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Settings(raw=raw)

    # Définition : 
    # Cette fonction est le premier point d'entrée de chaque script (Train, API, App). 
    # Elle assure que tout le projet partage la même vérité de configuration.


def ensure_dirs(settings: Settings) -> None:
    """
    Vérifie et crée les dossiers nécessaires au stockage des artefacts (modèles, rapports).
    """
    # Création récursive des dossiers si absent (parents=True) sans erreur si déjà présent (exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metrics_dir.mkdir(parents=True, exist_ok=True)

    # Définition : 
    # Indispensable pour l'industrialisation. En production (ex: Docker), le système de fichiers 
    # peut être vide ; ce script garantit que l'application ne plantera pas faute de dossiers cibles.