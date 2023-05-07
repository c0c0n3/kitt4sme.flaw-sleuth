# __version__ = '0.1.0'
from flawsleuth.utils.batch_preprocess import Preprocessing
from flawsleuth.utils.shift_detect import *
from flawsleuth.utils.kalman import *

from pathlib import Path


__version__ = '0.1.0'


def pyproject_file() -> Path:
    this_file = Path(__file__)
    root_dir = this_file.parent.parent
    return root_dir / 'pyproject.toml'