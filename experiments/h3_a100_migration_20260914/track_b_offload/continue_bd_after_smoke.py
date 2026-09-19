"""One-shot continuation: launch B/D comparison only after D smoke passes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/cache/zhonghao/h3')
RUN=ROOT/'bd_prepared_20260915_v2'
KIT=ROOT/'track_b_validation_20260915'


def main():
    output=RUN/'continuation.json'
    if sys.argv[1:]==['--launch']:
        with (RUN/'continuation.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',__file__],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'continuation_pid':child.pid}));return
    assert not sys.argv[1:]
    output.write_text(json.dumps({'pid':os.getpid(),'status':'waiting_for_D_smoke'}))
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        if (RUN/'STOP').exists():
            output.write_text(json.dumps({'status':'stopped'}));return
        status_path=RUN/'queue_status.json'
        if status_path.exists():
            status=json.loads(status_path.read_text())
            if status.get('status')=='failed':
                output.write_text(json.dumps({'status':'smoke_failed_no_comparison','error':status.get('error')}));return
            if status.get('status')=='smoke_passed_comparison_not_started':
                # Wait for the smoke controller to finish its cleanup/lock.
                if Path(f"/proc/{status['pid']}").exists():
                    time.sleep(5);continue
                result=json.loads((RUN/'smoke_D/smoke_2step/result.json').read_text())
                assert result['http_success'] and result['shape_verified']
                outcome=subprocess.run([sys.executable,str(KIT/'run_prepared_bd.py'),'--phase','compare','--launch'],
                    text=True,capture_output=True)
                output.write_text(json.dumps({'status':'comparison_dispatched' if outcome.returncode==0 else 'dispatch_failed',
                    'returncode':outcome.returncode,'stdout':outcome.stdout,'stderr':outcome.stderr},indent=2))
                print(outcome.stdout,outcome.stderr,flush=True);return
        time.sleep(10)
    output.write_text(json.dumps({'status':'smoke_wait_timeout_no_comparison'}))


if __name__=='__main__':main()
