"""Locations shared by repository scripts and the optional capstone project."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPSTONE_ROOT = Path(os.environ.get('ECOMMERCE_CAPSTONE_ROOT', PROJECT_ROOT)).expanduser().resolve()
