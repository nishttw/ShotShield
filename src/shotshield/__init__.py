"""ShotShield: gunshot detection in urban audio."""
from .detector import Detection, Result, ShotShieldDetector, Window
from .features import FeatureConfig

__all__ = ["ShotShieldDetector", "Result", "Window", "Detection", "FeatureConfig"]
__version__ = "2.0.0"
