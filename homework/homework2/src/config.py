from dotenv import load_dotenv
from pathlib import Path
import os

def get_key(name, default):
    return os.getenv(name, default)

load_dotenv()
PROJECT_ROOT = Path.cwd().parent.resolve()
