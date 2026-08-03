import os
import sys

# Add root directory to python path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

dashboard_app = os.path.join(root_dir, "dashboard", "app.py")

with open(dashboard_app, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals())



