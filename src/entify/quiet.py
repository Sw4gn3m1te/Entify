import contextlib
import logging
import os
import sys
import warnings


@contextlib.contextmanager
def silenced_stderr():
    # TensorFlow's native startup messages ignore TF_CPP_MIN_LOG_LEVEL, so the stderr file descriptor itself is redirected.
    sys.stderr.flush()
    saved = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)


def silence_library_logs():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
