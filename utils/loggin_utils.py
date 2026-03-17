import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def log_method(level=logging.DEBUG):
    """
    Decorator to automatically log method execution.

    Logs:
        - method start
        - method end
        - execution time
        - errors

    Example
    -------
    @log_method()
    def crop(self):
        ...
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            class_name = args[0].__class__.__name__ if args else ""
            method_name = func.__name__

            logger.log(level, f"START {class_name}.{method_name}")

            start = time.time()

            try:
                result = func(*args, **kwargs)

            except Exception as e:
                logger.exception(
                    f"ERROR in {class_name}.{method_name}: {e}"
                )
                raise

            runtime = time.time() - start

            logger.log(
                level,
                f"END {class_name}.{method_name} | runtime={runtime:.2f}s"
            )

            return result

        return wrapper

    return decorator