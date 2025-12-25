from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import pandas as pd
import joblib

# --- Fix imports when runing `streamlit` from `/app`
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

