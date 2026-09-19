#!/usr/bin/env python3
"""Run only H3 Ref2VA baseline A on the four explicitly allocated physical NPUs.

Linux-only, standard library. No installation, model modification, training, or
operation on another experiment's processes. Lock files owned by the other
experiment are opened read-only and are never rewritten or removed.
"""

from __future__ import annotations

import datetime as dt
import fcntl
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


CARDS = (2, 3, 6, 7)
PORT = 19098
OUTPUT_ROOT = Path('/cache/zhonghao/h3_v2v_minimal_20260913')
LEASE_ROOT = Path('/cache/zihaohe/generation-supervision/runs')
BASE_URL = f'http://127.0.0.1:{PORT}'
HEALTH_TIMEOUT = 1800
REQUEST_TIMEOUT = 1800


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def require_selected_cards_idle(output):
    """Fail closed against unknown process-table formats or inconsistent rows."""
    headers = list(re.finditer(
        r'^\|\s*NPU\s+Chip\s*\|\s*Process id\s*\|\s*Process name\s*\|'
        r'\s*Process memory\(MB\)\s*\|\s*$', output, re.MULTILINE))
    if len(headers) != 1:
        raise RuntimeError('Unrecognized or ambiguous npu-smi process table')
    empty = set()
    busy = set()
    for line in output[headers[0].end():].splitlines():
        line = line.strip()
        if not line or re.fullmatch(r'[+\-=]+', line):
            continue
        idle = re.fullmatch(r'\|\s*No running processes found in NPU (\d+)\s*\|', line)
        if idle:
            card = int(idle.group(1))
            if card in empty:
                raise RuntimeError(f'Duplicate idle entry for NPU {card}')
            empty.add(card)
            continue
        process = re.fullmatch(
            r'\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*[^|]+\|\s*\d+\s*\|', line)
        if process:
            busy.add(int(process.group(1)))
            continue
        raise RuntimeError(f'Unrecognized npu-smi process row: {line!r}')
    missing = set(CARDS) - empty
    conflict = set(CARDS) & busy
    if missing or conflict:
        raise RuntimeError(f'Selected cards not explicitly idle: missing={missing}, busy={conflict}')
    return {'selected_physical_cards': list(CARDS), 'explicitly_idle_cards': sorted(empty),
            'other_busy_cards': sorted(busy)}


def proc_identity(pid):
    """Return Linux start ticks and process/session group without importing NPU code."""
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': pid, 'state': fields[0], 'pgrp': int(fields[2]),
                'session': int(fields[3]), 'start_ticks': int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


class Supervisor:
    def __init__(self, config, script_dir):
        self.config = config
        self.script_dir = script_dir
        self.run_dir = None
        self.locks = []
        self.children = []
        self.server = None
        self.handles = []
        self.run_id = uuid.uuid4().hex
        self.status = {'case': 'A', 'run_id': self.run_id, 'supervisor_pid': os.getpid(),
                       'supervisor_proc_identity': proc_identity(os.getpid()),
                       'started_at': utc_now(), 'allocated_physical_npu_ids': list(CARDS),
                       'port': PORT, 'actual_model_forward_calls': None,
                       'new_acceleration_results': None}

    def update(self, phase, **values):
        self.status.update(values)
        self.status.update(phase=phase, updated_at=utc_now())
        if self.run_dir is not None:
            atomic_json(self.run_dir / 'a_status.json', self.status)
            atomic_json(OUTPUT_ROOT / 'a_status.json', self.status)
        print(json.dumps({'phase': phase, 'updated_at': self.status['updated_at'], **values},
                         ensure_ascii=False), flush=True)

    def acquire(self):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        own_lock = (OUTPUT_ROOT / 'run.lock').open('a+b')
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            own_lock.close()
            raise RuntimeError('Another supervised experiment holds our run.lock')
        self.locks.append(own_lock)
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.run_dir = OUTPUT_ROOT / 'runs' / ('a_' + stamp)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / 'output').mkdir()
        self.status['run_directory'] = str(self.run_dir)
        atomic_json(self.run_dir / 'experiment.json', self.config)
        self.update('acquiring_device_leases')
        for card in CARDS:
            path = LEASE_ROOT / f'new_layout_training_device{card}.lock'
            # Existing cooperative lease only: fail if missing; never create it.
            handle = path.open('rb')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise RuntimeError(f'Physical NPU {card} lease is held; no model was launched')
            self.locks.append(handle)
        self.update('device_leases_acquired')

    def preflight(self):
        source = Path(self.config['source_video'])
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f'Missing or empty source video: {source}')
        generation = self.config['requested_generation']
        if int(generation['num_inference_steps']) != 50:
            raise RuntimeError('This baseline expects the reviewed 50-step formal request')
        configured_cards = self.config.get('allocated_physical_npu_ids')
        if configured_cards and list(configured_cards) != list(CARDS):
            raise RuntimeError('experiment.json card allocation conflicts with fixed 2,3,6,7')
        for tool in ('npu-smi', 'curl', 'bash'):
            if shutil.which(tool) is None:
                raise RuntimeError(f'Required executable missing from PATH: {tool}')
        if not (self.script_dir / 'launch_a_server.sh').is_file():
            raise RuntimeError('Missing same-directory launch_a_server.sh')
        check = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True,
                               timeout=30, check=False)
        (self.run_dir / 'npu_before.txt').write_text(check.stdout + '\n' + check.stderr)
        if check.returncode:
            raise RuntimeError(f'npu-smi failed with code {check.returncode}')
        idle = require_selected_cards_idle(check.stdout)
        # bind() also detects wildcard listeners; close immediately before launch.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', PORT))
        self.update('preflight_passed', resource_snapshot=idle)

    def env(self):
        env = os.environ.copy()
        env.update(H3_ROOT=str(self.run_dir), H3_OUTPUT=str(self.run_dir / 'output'),
                   H3_PORT=str(PORT), ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, CARDS)),
                   H3_MINIMAL_RUN_ID=self.run_id, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONNOUSERSITE='1')
        return env

    def spawn(self, command, **kwargs):
        child = subprocess.Popen(command, start_new_session=True, env=self.env(),
                                 cwd=self.script_dir,
                                 pass_fds=tuple(handle.fileno() for handle in self.locks),
                                 **kwargs)
        identity = proc_identity(child.pid)
        self.children.append((child, identity))
        return child

    def launch(self):
        log = (self.run_dir / 'server.log').open('ab', buffering=0)
        self.handles.append(log)
        self.server = self.spawn(['bash', str(self.script_dir / 'launch_a_server.sh')],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.update('server_starting', server_pid=self.server.pid,
                    server_started_at=utc_now(), server_proc_identity=proc_identity(self.server.pid))
        deadline = time.monotonic() + HEALTH_TIMEOUT
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f'Server exited during startup: {self.server.returncode}; see server.log')
            try:
                with opener.open(BASE_URL + '/health', timeout=5) as response:
                    if response.status == 200:
                        self.update('server_healthy', server_ready_at=utc_now())
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError('Server did not become healthy within 30 minutes')

    def probe_video(self, path):
        executable = shutil.which('ffprobe')
        if executable is None and os.access('/usr/local/ffmpeg/bin/ffprobe', os.X_OK):
            executable = '/usr/local/ffmpeg/bin/ffprobe'
        if executable is None:
            return {'verified': False, 'reason': 'ffprobe unavailable; video metadata unverified'}
        result = subprocess.run([executable, '-v', 'error', '-show_entries',
                                 'stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames',
                                 '-show_entries', 'format=duration,size', '-of', 'json', str(path)],
                                capture_output=True, text=True, timeout=60, check=False)
        if result.returncode:
            return {'verified': False, 'executable': executable, 'returncode': result.returncode,
                    'stderr': result.stderr}
        try:
            metadata = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {'verified': False, 'reason': 'ffprobe returned invalid JSON',
                    'stdout': result.stdout, 'stderr': result.stderr}
        videos = [stream for stream in metadata.get('streams', []) if stream.get('codec_type') == 'video']
        failures, warnings = [], []
        checks = {}
        expected = self.config['requested_generation']
        if not videos:
            failures.append('No video stream')
        else:
            video = videos[0]
            checks['dimensions'] = [video.get('width'), video.get('height')]
            if checks['dimensions'] != [expected['width'], expected['height']]:
                failures.append('Output dimensions do not match requested dimensions')
            try:
                actual_fps = float(Fraction(video.get('r_frame_rate', '0/0')))
                checks['fps'] = actual_fps
                if abs(actual_fps - float(expected['fps'])) > 0.01:
                    failures.append('Output FPS does not match requested FPS')
            except (ValueError, ZeroDivisionError):
                failures.append('Video FPS unavailable or invalid')
            frames = video.get('nb_frames')
            if frames in (None, 'N/A'):
                checks['frame_count'] = None
                warnings.append('nb_frames unavailable; frame count not verified')
            else:
                try:
                    checks['frame_count'] = int(frames)
                    if checks['frame_count'] <= 0:
                        failures.append('Video frame count is not positive')
                except (TypeError, ValueError):
                    failures.append('Video frame count is invalid')
        try:
            duration = float(metadata.get('format', {}).get('duration', 0))
            checks['container_duration_seconds'] = duration
            checks['duration_tolerance_seconds'] = 0.5
            if not math.isfinite(duration) or duration <= 0 or abs(duration - float(expected['duration_seconds'])) > 0.5:
                failures.append('Container duration invalid or differs from requested duration by over 0.5 seconds')
        except (TypeError, ValueError):
            failures.append('Container duration unavailable or invalid')
        return {'verified': not failures, 'executable': executable, 'metadata': metadata,
                'checks': checks, 'failures': failures, 'warnings': warnings}

    def request(self, name, steps):
        self.update(name + '_request_running')
        generation = self.config['requested_generation']
        directory = self.run_dir / name
        directory.mkdir()
        headers = directory / 'response.headers'
        body = directory / 'response.body'
        extra = {'task': 'ref2va', 'duration': generation['duration_seconds'],
                 'audio_flow_shift': generation['audio_flow_shift']}
        fields = {'prompt': self.config['edit_prompt'], 'width': generation['width'],
                  'height': generation['height'], 'fps': generation['fps'],
                  'num_inference_steps': steps, 'flow_shift': generation['flow_shift'],
                  'seed': generation['seed'], 'extra_params': json.dumps(extra)}
        request_record = {'started_at': utc_now(), 'fields': fields,
                          'source_video': self.config['source_video'],
                          'requested_steps': steps, 'actual_model_forward_calls': None,
                          'quality_evidence': False,
                          'note': 'Two-step smoke is connectivity only; formal output needs visual review.'}
        atomic_json(directory / 'request.json', request_record)
        command = ['curl', '--silent', '--show-error', '--noproxy', '*',
                   '--connect-timeout', '10', '--max-time', str(REQUEST_TIMEOUT),
                   '--dump-header', str(headers), '--output', str(body),
                   '--write-out', '%{http_code}', '--request', 'POST',
                   BASE_URL + '/v1/videos/sync']
        for key, value in fields.items():
            command.extend(['--form-string', f'{key}={value}'])
        command.extend(['--form', f'input_references=@{self.config["source_video"]};type=video/mp4'])
        stdout = (directory / 'curl.stdout').open('wb')
        stderr = (directory / 'curl.stderr').open('wb')
        self.handles.extend([stdout, stderr])
        started = time.monotonic()
        child = self.spawn(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        deadline = started + REQUEST_TIMEOUT + 20
        while child.poll() is None:
            if self.server.poll() is not None:
                raise RuntimeError('Server exited during request; see server.log')
            if time.monotonic() >= deadline:
                raise TimeoutError('curl exceeded its request timeout plus cleanup allowance')
            time.sleep(1)
        stdout.flush()
        stderr.flush()
        elapsed = time.monotonic() - started
        code = (directory / 'curl.stdout').read_text(errors='replace').strip()
        header_text = headers.read_text(errors='replace') if headers.exists() else ''
        content_types = re.findall(r'^content-type:\s*([^\r\n]+)', header_text,
                                  re.IGNORECASE | re.MULTILINE)
        content_type = content_types[-1].strip().lower() if content_types else None
        success = (child.returncode == 0 and code == '200' and content_type is not None
                   and content_type.startswith('video/mp4') and body.exists() and body.stat().st_size > 0)
        result = {'completed_at': utc_now(), 'http_code': code, 'content_type': content_type,
                  'curl_returncode': child.returncode, 'request_end_to_end_seconds': elapsed,
                  'requested_steps': steps, 'actual_model_forward_calls': None,
                  'dit_latency_seconds': None, 'success': success,
                  'quality_evidence': False}
        atomic_json(directory / 'result.json', result)
        if not success:
            raise RuntimeError(f'{name} failed: curl={child.returncode}, HTTP={code}, type={content_type}')
        output_video = self.run_dir / 'output' / (name + '.mp4')
        body.rename(output_video)
        probe = self.probe_video(output_video)
        atomic_json(directory / 'ffprobe.json', probe)
        result.update(output_video=str(output_video), ffprobe=probe)
        atomic_json(directory / 'result.json', result)
        if not probe['verified'] and probe.get('reason') != 'ffprobe unavailable; video metadata unverified':
            raise RuntimeError(f'{name} response could not be verified as video by ffprobe')
        self.update(name + '_completed', **{name: result})

    def owned_group_members(self, pgid):
        """Only consider the session created by this supervisor, with its run marker."""
        members = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                identity = proc_identity(int(entry.name))
                if not identity or identity['pgrp'] != pgid or identity['session'] != pgid:
                    continue
                if identity['state'] == 'Z':
                    continue
                environ = (entry / 'environ').read_bytes().split(b'\0')
                if f'H3_MINIMAL_RUN_ID={self.run_id}'.encode() in environ:
                    members.append(identity['pid'])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return members

    def cleanup(self):
        if self.run_dir is not None:
            self.update('cleaning_up_own_processes')
        # Only sessions we created; never discover or signal another experiment PID.
        pending = []
        for child, identity in reversed(self.children):
            current = proc_identity(child.pid)
            leader_matches = (identity is not None and current is not None
                              and current['start_ticks'] == identity['start_ticks']
                              and current['pgrp'] == child.pid and current['session'] == child.pid)
            members = self.owned_group_members(child.pid)
            if leader_matches or members:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    pending.append(child)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30
        while pending and time.monotonic() < deadline:
            pending = [child for child in pending if self.owned_group_members(child.pid)]
            if pending:
                time.sleep(1)
        for child in pending:
            if self.owned_group_members(child.pid):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for child, _ in self.children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        remaining = {child.pid: self.owned_group_members(child.pid) for child, _ in self.children}
        remaining = {group: pids for group, pids in remaining.items() if pids}
        self.status['remaining_owned_process_groups'] = remaining
        self.status['cleanup_completed'] = not remaining
        self.status['needs_attention'] = bool(remaining)
        if self.server is not None:
            try:
                after = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True,
                                       timeout=30, check=False)
                (self.run_dir / 'npu_after.txt').write_text(after.stdout + '\n' + after.stderr)
                if after.returncode:
                    raise RuntimeError(f'npu-smi after cleanup returned {after.returncode}')
                self.status['resource_release_check'] = require_selected_cards_idle(after.stdout)
                self.status['selected_cards_verified_idle_after_cleanup'] = True
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.status['selected_cards_verified_idle_after_cleanup'] = False
                self.status['resource_release_check_error'] = str(exc)
                self.status['needs_attention'] = True
        for stream in self.handles:
            stream.close()
        return not remaining

    def release_locks(self):
        # Final status is written while our run.lock is still held. Close rather
        # than LOCK_UN: an inherited descriptor keeps its lease if an unkillable
        # owned child remains. Never delete either kind of lock.
        for handle in reversed(self.locks):
            handle.close()
        self.locks.clear()


def interrupted(signum, _frame):
    raise InterruptedError(f'Received signal {signum}; stopping only our new processes')


def main():
    script_dir = Path(__file__).resolve().parent
    config = json.loads((script_dir / 'experiment.json').read_text(encoding='utf-8-sig'))
    supervisor = Supervisor(config, script_dir)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    error = None
    try:
        supervisor.acquire()
        supervisor.preflight()
        supervisor.launch()
        supervisor.request('smoke_2step', 2)
        supervisor.request('a_50step', 50)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        if supervisor.run_dir is not None:
            supervisor.update('failed_before_cleanup', error=error)
        else:
            print(error, file=sys.stderr, flush=True)
    finally:
        # Repeated Ctrl-C must not interrupt lease-preserving cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            clean = supervisor.cleanup()
            if supervisor.run_dir is not None:
                if not clean and error is None:
                    error = 'Owned processes remain after cleanup; do not reuse cards until checked'
                if supervisor.status.get('needs_attention') and error is None:
                    error = 'Post-run resource release could not be verified; inspect npu_after.txt'
                phase = 'needs_attention' if supervisor.status.get('needs_attention') else ('failed' if error else 'completed')
                supervisor.update(phase, finished_at=utc_now(), error=error,
                                  note='Scope: baseline A only. No B/C result or acceleration claim; NFE unmeasured.')
        finally:
            supervisor.release_locks()
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
