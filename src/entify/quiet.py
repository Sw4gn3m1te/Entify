import contextlib
import logging
import os
import sys
import warnings


@contextlib.contextmanager
def silenced_stderr():
    # TensorFlow's native startup messages ignore TF_CPP_MIN_LOG_LEVEL, so the stderr file descriptor itself is redirected.
    # Python's sys.stderr keeps writing to the terminal, so progress bars such as the model download stay visible.
    python_stderr = sys.stderr
    python_stderr.flush()
    saved = os.dup(2)
    terminal = open(saved, "w", encoding=python_stderr.encoding, errors="replace", closefd=False)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 2)
        sys.stderr = terminal
        try:
            yield
        finally:
            terminal.flush()
            sys.stderr = python_stderr
            os.dup2(saved, 2)
            terminal.close()
            os.close(saved)


def silence_library_logs():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
