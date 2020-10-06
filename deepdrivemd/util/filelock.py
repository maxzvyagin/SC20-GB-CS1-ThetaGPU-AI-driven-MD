import os
import random
import time
import logging

logger = logging.getLogger(__name__)


class FileLock:
    EXPIRATION_SECONDS = 20

    def __init__(self, path):
        self.lock_path = path + ".lock"

    def check_stale(self):
        try:
            mtime = os.path.getmtime(self.lock_path)
        except OSError:
            return False
        else:
            return time.time() - mtime > self.EXPIRATION_SECONDS

    def acquire_lock(self, timeout=40.0):
        acquired = False
        start = time.time()
        newline = False

        def _time_left():
            if timeout is None:
                return True
            return time.time() - start < timeout

        logger.debug(f"Attempting to acquire lock: {self.lock_path}")
        while not acquired and _time_left():
            try:
                os.mkdir(self.lock_path)
            except OSError:
                time.sleep(1.0 + random.uniform(0, 0.5))
                newline = True
                if self.check_stale():
                    try:
                        os.rmdir(self.lock_path)
                    except FileNotFoundError:
                        pass
            else:
                acquired = True
                if newline:
                    logger.debug(f"Acquired lock: {self.lock_path}")
        if not acquired:
            raise TimeoutError(f"Failed to acquire {self.lock_path} for {timeout} sec")

    def release_lock(self):
        try:
            os.rmdir(self.lock_path)
        except FileNotFoundError:
            pass

    def __enter__(self):
        self.acquire_lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release_lock()
