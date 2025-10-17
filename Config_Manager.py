import os
import sys
import configparser
from typing import Dict


class ConfigManager:
    _instance = None
    _config_paths = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_config_paths()
            self._initialized = True

    def _get_base_directory(self):
        if hasattr(sys, '_MEIPASS'):
            result = os.path.dirname(sys.executable)
            print(f"DEBUG: _get_base_directory returning (frozen): {result}")
            print(f"DEBUG: sys.executable = {sys.executable}")
            print(f"DEBUG: sys._MEIPASS = {sys._MEIPASS}")
            return result
        else:
            result = os.path.dirname(os.path.abspath(__file__))
            print(f"DEBUG: _get_base_directory returning (source): {result}")
            return result

    def _setup_config_paths(self):
        """Setup all config file paths"""
        base_dir = self._get_base_directory()

        self._config_paths = {
            'main': os.path.join(base_dir, 'config.ini'),
            'calibration': os.path.join(base_dir, 'calibration.ini'),
            'sam_parameters': os.path.join(base_dir, 'samParameters.ini')
        }

    def get_config_path(self, config_name: str) -> str:
        """Get specified config file path"""
        if config_name not in self._config_paths:
            raise ValueError(f"Unknown config file: {config_name}")

        path = self._config_paths[config_name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file does not exist: {path}")

        return path

    def get_config(self, config_name: str) -> configparser.ConfigParser:
        """Get config parser object"""
        config = configparser.ConfigParser()
        config.read(self.get_config_path(config_name))
        return config


# Global function to get SAM parameters path for backward compatibility
def get_sam_parameters_path():
    """Get SAM parameters config file path - replaces the original parameterPath logic"""
    try:
        config_manager = ConfigManager()
        return config_manager.get_config_path('sam_parameters')
    except FileNotFoundError:
        # Fallback to original logic if file not found in exe directory
        original_path = os.path.abspath(os.path.join(os.getcwd(), "imageAnalysis", "samParameters.ini"))
        if os.path.exists(original_path):
            return original_path
        raise FileNotFoundError(
            "Cannot find samParameters.ini in either exe directory or original imageAnalysis folder")


def get_calibration_config_path():
    """Get calibration config file path"""
    try:
        config_manager = ConfigManager()
        return config_manager.get_config_path('calibration')
    except FileNotFoundError:
        # Fallback to original logic if needed
        original_path = os.path.abspath(os.path.join(os.getcwd(), "ImagePreprocessing", "calibration.ini"))
        if os.path.exists(original_path):
            return original_path
        raise FileNotFoundError(
            "Cannot find calibration.ini in either exe directory or original ImagePreprocessing folder")