import os

DATA_DIR = os.getenv("DATA_DIR", os.path.expanduser("~/data"))
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
