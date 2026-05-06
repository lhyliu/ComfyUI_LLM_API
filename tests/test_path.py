import os
import sys


CUSTOM_NODES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CUSTOM_NODES_DIR not in sys.path:
    sys.path.insert(0, CUSTOM_NODES_DIR)
