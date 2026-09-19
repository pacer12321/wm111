"""User explicitly allows coexistence; hold70GiB while leaving >=8GiB free."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import reserve_gpus as reservation

assert socket.gethostname() == 'os-node-created-mgf6h'
reservation.ROOT = Path(__file__).resolve().parent
TARGET = 'GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670'
reservation.TARGETS = {'0': TARGET}


def check_headroom():
    rows = reservation.query('gpu', 'index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu')
    selected = [r for r in rows if r[0] == '0']
    assert len(selected) == 1 and selected[0][1] == TARGET
    index, uuid, bus, used, total, utilization = selected[0]
    assert int(utilization) == 0 and int(used) <= 1024, 'Existing workload changed; do not allocate'
    assert (int(total) - int(used)) * 1024**2 >= reservation.HOLD_BYTES + 8 * 1024**3
    processes = reservation.query('compute-apps', 'gpu_uuid,pid')
    existing = [pid for gpu, pid in processes if gpu == TARGET]
    assert not existing or existing == ['1550469'], 'Unexpected GPU0 process; refuse'
    return {index: bus}


reservation.check_idle = check_headroom

if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        check_headroom()
        assert not (reservation.ROOT / 'reservation.json').exists()
        assert not (reservation.ROOT / 'STOP').exists()
        with (reservation.ROOT / 'reservation.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__],
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'launched_pid': child.pid, 'gpu': 0,
                          'coexists_with_existing_850MiB_process': True}))
    else:
        assert len(sys.argv) == 1
        reservation.main()
