"""Diagnose emulated process identity and file locks; no ROS/hardware calls."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def identity(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    return dict(pid=pid, start=text[text.rindex(')')+2:].split()[19],
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


if len(sys.argv) > 1:
    print(json.dumps(identity(int(sys.argv[1]))))
else:
    before = identity(os.getpid())
    child = subprocess.run([sys.executable, __file__, str(os.getpid())], capture_output=True, text=True, timeout=30)
    print(json.dumps(dict(parent_before=before, parent_after=identity(os.getpid()), child_stdout=child.stdout,
                          child_stderr=child.stderr, child_returncode=child.returncode)), flush=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'probe.lock'
        with path.open('w') as first, path.open('w') as second:
            print('Taking first lock', flush=True)
            fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print('Trying second NONBLOCKING lock', flush=True)
            try:
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('PASS: second lock rejected without waiting', flush=True)
            else:
                raise RuntimeError('Second lock was incorrectly granted')
