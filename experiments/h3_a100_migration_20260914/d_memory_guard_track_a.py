"""Track A only: non-file guard; file cache is excluded, not guaranteed reclaimable."""
import json
import threading
import time

from d_memory_probe import snapshot

GIB = 1024 ** 3
# User-requested decimal GB threshold; strictly greater than, not >=.
MAX_USAGE_BYTES = 250_000_000_000


def pressure_reason(sample, initial_oom):
    usage, limit = sample['usage_in_bytes'], sample['limit_in_bytes']
    oom = sample['oom_control']['oom_kill']
    stat = sample['stat']
    # Cache includes shmem, which must NOT be treated as reclaimable file cache.
    file_cache = max(0, stat['cache'] - stat['shmem'])
    non_file = max(0, usage - file_cache)
    if oom > initial_oom:
        return f'cgroup OOM count increased {initial_oom}->{oom}; attribution needs process evidence'
    if non_file > MAX_USAGE_BYTES:
        return (f'proactive non-file memory stop: usage={usage}, threshold={MAX_USAGE_BYTES}, '
                f'non_file={non_file}, limit={limit}')
    return None


class MemoryGuard:
    def __init__(self, output, stage):
        self.output = output
        self.stage = stage
        self.stop = threading.Event()
        self.reason = None
        self.thread = None

    def start(self):
        baseline = snapshot()
        self.initial_oom = baseline['oom_control']['oom_kill']
        pressure_reason(baseline, self.initial_oom)  # validate expected cgroup fields
        self.handle = self.output.open('x')
        self.handle.write(json.dumps({'baseline': baseline,
                                      'usage_threshold_bytes': MAX_USAGE_BYTES,
                                      'threshold_comparison': 'strictly_greater_than',
                                      'threshold_metric': 'usage_minus_cache_excluding_shmem'}) + '\n')
        self.handle.flush()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                sample = snapshot()
                sample['queue_stage'] = self.stage()
                sample['guard_reason'] = pressure_reason(sample, self.initial_oom)
                self.handle.write(json.dumps(sample) + '\n')
                self.handle.flush()
                if sample['guard_reason']:
                    self.reason = sample['guard_reason']
                    return
                self.stop.wait(0.25)
        except BaseException as exc:
            self.reason = 'memory observer failed: ' + repr(exc)

    def check(self):
        if self.reason:
            raise RuntimeError(self.reason)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        if hasattr(self, 'handle') and not (self.thread and self.thread.is_alive()):
            self.handle.close()
