#!/usr/bin/env python3
"""Bind a successful prepare --build test run to the installed candidate libraries."""
import argparse, hashlib, json, platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['write','check']);p.add_argument('--package-share',type=Path,required=True);a=p.parse_args()
    prefix=a.package_share.parent.parent
    receipt=prefix/'notebook-lift-tested.json'
    current={'release_sha256':digest(ROOT/'hardware_testing/notebook_lift/release.json'),
             'machine':platform.machine(),
             'libraries':{name:digest(prefix/'lib'/name) for name in ['libnotebook_lift_controller.so','libneural_controller.so']}}
    if a.action=='write':receipt.write_text(json.dumps(current,indent=2)+'\n')
    elif json.loads(receipt.read_text())!=current:raise ValueError('Candidate changed since robot-side build/tests; rerun prepare_notebook_lift.sh --build')
    print('PASS installed build/test receipt' if a.action=='check' else 'Recorded successful installed build/test run')
if __name__=='__main__':main()
