import subprocess
import logging
import time

logger = logging.getLogger(__name__)


class CopySender:
    def __init__(self, target, method="cp"):
        self.method = method
        self.target = target
        self.processes = []

    def send(self, path):
        args = f"{self.method} {path} {self.target}"
        logger.debug(f"Starting transfer: {args}")
        p = subprocess.Popen(args, shell=True)
        self.processes.append(p)
        self.processes = [p for p in self.processes if p.poll() is None]

    def wait_all(self, timeout=30):
        logger.debug("Waiting on all processes to join...")
        start = time.time()
        for p in self.processes:
            elapsed = time.time() - start
            max_wait = max(0, timeout - elapsed)
            try:
                p.wait(timeout=max_wait)
            except subprocess.TimeoutExpired:
                logger.warning("Transfer {p.args} did not finish!")
