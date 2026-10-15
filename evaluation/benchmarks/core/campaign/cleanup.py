"""Poll lane cleanup independently; only the controller commits queue receipts.

The verifier runs outside the controller thread. Expired lanes stay reserved
until their in-flight verifier returns, so a port is never reused on a timeout.
"""
from concurrent.futures import ThreadPoolExecutor
import time


class CleanupChecks:
    def __init__(self, *, clock=time.monotonic, budget=120):
        self.clock, self.budget = clock, budget
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lane-cleanup")
        self.jobs = {}

    def start(self, lane, verify):
        if lane in self.jobs:
            raise ValueError("Lane already has cleanup pending")
        deadline = time.monotonic()+self.budget
        def bounded_verify():
            from src.sim_server.remote.ssh import ssh_deadline
            with ssh_deadline(deadline):
                return verify()
        self.jobs[lane] = {"verify": bounded_verify, "started": self.clock(), "checks": 0,
                           "future": self.pool.submit(bounded_verify), "history": []}

    def poll(self, lane):
        job = self.jobs[lane]
        now = self.clock()
        future = job.get("future")
        if future is None:
            if now < job["next_check"]:
                return None
            job["future"] = self.pool.submit(job["verify"])
            return None
        if not future.done():
            return None
        receipt = future.result()
        job["checks"] += 1
        job["history"].append({"elapsed_seconds": now-job["started"],
                               "verified": receipt.get("cleanup_verified", False)})
        if receipt.get("cleanup_verified") or now-job["started"] >= self.budget:
            receipt["cleanup_checks"] = job["history"]
            del self.jobs[lane]
            return receipt
        job.update(future=None, next_check=min(job["started"]+self.budget,
                   now+(2, 5, 10, 20)[min(job["checks"]-1, 3)]))
        return None

    def close(self):
        # A verifier may still own SSH resources; never start competing cleanup.
        self.pool.shutdown(wait=True, cancel_futures=True)
