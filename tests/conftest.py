import os
import sys

# Ensure project root is on PYTHONPATH for test imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(PROJECT_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
