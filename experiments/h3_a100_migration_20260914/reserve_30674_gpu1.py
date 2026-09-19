"""Hold only idle GPU1 on the user-assigned 30674 machine; never stop jobs."""
from pathlib import Path
import socket
import reserve_gpus as reservation

assert socket.gethostname() == 'os-node-created-mgf6h'
reservation.ROOT = Path(__file__).resolve().parent
reservation.TARGETS = {'1': 'GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3'}
if __name__ == '__main__':
    import json
    import subprocess
    import sys
    if sys.argv[1:] == ['--launch']:
        reservation.check_idle()
        assert not (reservation.ROOT / 'reservation.json').exists()
        assert not (reservation.ROOT / 'STOP').exists()
        with (reservation.ROOT / 'reservation.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__],
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'launched_pid': child.pid, 'gpu': 1}))
    else:
        assert len(sys.argv) == 1
        reservation.main()
