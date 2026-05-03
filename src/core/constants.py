

from pathlib import Path
from .config import load_config

_config = load_config()
_ROOT_DIR = Path(__file__).parent.parent.parent


PORT_API = _config.get("port_api", 2156)


MODELS_CONFIG_FILE = str(_ROOT_DIR / "config" / "models.json")
STATS_FILE = str(_ROOT_DIR / "config" / "stats.json")
CONFIG_FILE = str(_ROOT_DIR / "config" / "config.json")
