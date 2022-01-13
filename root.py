import os
from functools import partial

ROOT = os.path.abspath(os.path.dirname(__file__))
absolute = partial(os.path.join, ROOT)
