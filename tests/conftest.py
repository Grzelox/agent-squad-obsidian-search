import sys
import pytest
from pathlib import Path

# Add the project root to the Python path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Provide the project root path for tests."""
    return Path(__file__).parent.parent 