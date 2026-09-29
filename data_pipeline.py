"""Compatibility shim for legacy root-level imports.

Older training artifacts were pickled under the module name ``data_pipeline``.
Newer code uses the package module ``src.data_pipeline``. This file keeps both
import paths working so the existing joblib artifacts can still be loaded.
"""

from src.data_pipeline import *  # noqa: F401,F403
