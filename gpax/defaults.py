import os


def set_default_jitter(jitter):
    os.environ["DEFAULT_JITTER"] = str(jitter)


def get_default_jitter():
    return float(os.environ["DEFAULT_JITTER"])
