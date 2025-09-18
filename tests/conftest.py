"""
Pytest configuration file to handle imports and setup
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Configure pytest
def pytest_configure():
    """Configure pytest settings"""
    pass