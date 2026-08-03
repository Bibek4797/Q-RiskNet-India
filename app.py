import os
import sys
import importlib

# Add root directory to python path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import dashboard.app
importlib.reload(dashboard.app)


