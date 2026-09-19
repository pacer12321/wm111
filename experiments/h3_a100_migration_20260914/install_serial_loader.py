"""Install only the reviewed optional CLI-extension hook, retaining original."""
import fcntl
import hashlib
from pathlib import Path
import shutil
import socket


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert socket.gethostname() == 'os-node-created-mgf6h'
    root=Path('/cache/zhonghao/h3')
    work=root/'a100_v1'
    lock=(work/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target=work/'run_abcd_cuda.py'
    staged=root/'run_abcd_cuda_serial_staged.py'
    original=target.read_bytes()
    new=staged.read_bytes()
    assert sha(original)=='f53ec2fcbabc7221e1d0fdcb5541c439a915649b525023c6fd6606e603e03bfa'
    assert sha(new)=='1093af62b304f54c98ba507908de71a76b73d1d2b02230db84fe07dad0671851'
    text=new.decode().replace('\r\n','\n')
    reverted=text.replace('SERVER_EXTRA_ARGS = ()  # Explicit per-attempt startup options; default remains unchanged.\n','').replace('    command.extend(SERVER_EXTRA_ARGS)\n','')
    assert reverted.encode()==original, 'Any unrelated edit refuses installation'
    backup=work/'history/run_abcd_cuda_before_serial_20260915.py'
    with backup.open('xb') as f:
        f.write(original)
    # Target is our exact validated queue, lock held, original preserved.
    staged.replace(target)
    print('Installed native serial-loading option hook; original backed up:',backup)


if __name__=='__main__':
    main()
