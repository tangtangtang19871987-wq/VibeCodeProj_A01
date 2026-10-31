"""The repository root is itself a package (array_generator) whose __init__ only imports
when installed under that name.  pytest walks up __init__.py files and would import it while
setting up the ilt_agent package; register a path-only stub instead of executing it."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_name = os.path.basename(ROOT)
if _name not in sys.modules:
    stub = types.ModuleType(_name)
    stub.__path__ = [ROOT]
    sys.modules[_name] = stub
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
