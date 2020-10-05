from pathlib import Path
import subprocess
import logging
import time

logger = logging.getLogger(__name__)


class CopySender:
    def __init__(self, target, method="cp"):
        self.method = method
        self.target = target
        self.processes = []

    def send(
        self,
        path,
        touch_done_file=False,
    ):
        args = f"{self.method} {path} {self.target}"
        if touch_done_file:
            # TODO: Socket for confirming transfer done would be better
            done_file = Path(self.target).joinpath(path.name).joinpath("DONE")
            args += f" && touch {done_file}"

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
