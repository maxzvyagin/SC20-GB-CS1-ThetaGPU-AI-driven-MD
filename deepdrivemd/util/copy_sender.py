from pathlib import Path
import subprocess
import logging
import json
import time

logger = logging.getLogger(__name__)


def retry_safe_Popen(args, max_retry=6):
    p = None
    for attempt in range(max_retry):
        try:
            p = subprocess.Popen(
                args,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
            )
            break
        except Exception as e:
            logger.warning(f"Popen raised: {e}")
            logger.warning(f"Popen attempt {attempt+1}/{max_retry}")
            time.sleep(3)
    if p is None:
        logger.error(f"Failed to Popen scp after {max_retry} attempts!")
        logger.error("Continuing and hoping for the best.")
    return p


class CopySender:
    def __init__(self, target):
        self.target = target
        self.processes = []

    def send(self, path):
        raise NotImplementedError

    def check_processes(self):
        remaining_processes = []
        for p in self.processes:
            retcode = p.poll()
            if retcode is None:
                remaining_processes.append(p)
            elif retcode == 0:
                self.log_transfer_success(p)
            else:
                self.log_transfer_error(p)
        self.processes = remaining_processes
        logger.debug(
            f"{self.__class__.__name__} has {len(self.processes)} active Popens left"
        )

    def wait_all(self, timeout=30):
        logger.debug(f"Waiting for {len(self.processes)} transfer processes to join")
        start = time.time()
        for p in self.processes:
            elapsed = time.time() - start
            max_wait = max(0, timeout - elapsed)
            try:
                p.wait(timeout=max_wait)
            except subprocess.TimeoutExpired:
                logger.warning(f"Transfer {p.args} did not finish!")
        self.check_processes()
        logger.debug("Sender wait_all finished")

    @staticmethod
    def log_transfer_success(process):
        stdout, _ = process.communicate()
        logger.info(stdout)

    @staticmethod
    def log_transfer_error(process):
        stdout, _ = process.communicate()
        logger.error(stdout)


class LocalCopySender(CopySender):
    def __init__(self, target):
        super().__init__(target)
        self.pid_size_map = {}

    def send(self, path, touch_done_file=False):
        if Path(path).is_dir():
            cmd = "time -p cp -r"
            size = sum(f.stat().st_size for f in Path(path).glob("**/*") if f.is_file())
        else:
            size = Path(path).stat().st_size
            cmd = "time -p cp"

        args = f"{cmd} {path} {self.target}"

        if touch_done_file:
            # TODO: Socket for confirming transfer done would be better
            done_file = Path(self.target).joinpath(Path(path).name).joinpath("DONE")
            args += f" && touch {done_file}"

        logger.debug(f"Starting transfer: {args}")
        p = retry_safe_Popen(args)
        if p is not None:
            self.processes.append(p)
        self.pid_size_map[p.pid] = size
        self.check_processes()

    def log_transfer_success(self, process):
        stdout, _ = process.communicate()
        sent_bytes = self.pid_size_map.pop(process.pid)
        line = next((l for l in stdout.split("\n") if "real" in l), None)
        if line is None:
            logger.warning("Could not parse cp performance from stdout")
            return
        try:
            time = float(line.split()[1])
        except:
            logger.warning("Could not parse cp timing from stdout")
            return
        else:
            time = max(time, 1e-32)
            rate = sent_bytes / time / 1e6
            perf = {
                "sent_bytes": sent_bytes,
                "send_time_sec": time,
                "rate_MB_per_sec": rate,
            }
            logger.info(f"COMPLETED_CP: {json.dumps(perf)}")
        logger.info(stdout)

    @staticmethod
    def log_transfer_error(process):
        stdout, _ = process.communicate()
        logger.error(f"cp subprocess had returncode {process.returncode}: {stdout}")


class RemoteCopySender(CopySender):
    def send(self, path):
        if Path(path).is_dir():
            send_cmd = "scp -v -r"
        else:
            send_cmd = "scp -v"
        args = f"{send_cmd} {path} {self.target}"
        logger.debug(f"Starting transfer: {args}")
        p = retry_safe_Popen(args)
        if p is not None:
            self.processes.append(p)
        self.check_processes()

    @staticmethod
    def log_transfer_success(process):
        stdout, _ = process.communicate()
        line = next((l for l in stdout.split("\n") if "Transferred:" in l), None)
        if line is None:
            logger.warning("Could not parse scp performance from stdout")
            return
        try:
            dat = line.split()
            sent_bytes = int(dat[2].strip(","))
            time = float(dat[-2])
        except:
            logger.warning("Could not parse scp performance from stdout")
            return
        else:
            time = max(time, 1e-32)
            rate = sent_bytes / time / 1e6
            perf = {
                "sent_bytes": sent_bytes,
                "send_time_sec": time,
                "rate_MB_per_sec": rate,
            }
            logger.info(f"COMPLETED_SCP: {json.dumps(perf)}")

    @staticmethod
    def log_transfer_error(process):
        stdout, _ = process.communicate()
        logger.error(f"scp subprocess had returncode {process.returncode}: {stdout}")