#!/usr/bin/env python3
"""Low-rate Pi diagnostics for supervised motion tests; no robot commands.

Writes and fsyncs locally and streams JSON lines on stdout for a second copy over
SSH. Measurements after a reboot cannot reconstruct a prior power transient.
"""
import argparse,json,os,subprocess,time
from pathlib import Path

def command(*args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=2)
        return {'returncode':p.returncode,'output':p.stdout.strip(),'error':p.stderr.strip()}
    except Exception as e:return {'error':str(e)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=900)
    a=p.parse_args()
    if not 1<=a.seconds<=3600:p.error('Duration must be 1–3600 seconds')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    start=time.monotonic()
    with a.output.open('x',buffering=1) as f:
        while time.monotonic()-start<a.seconds:
            tick=time.monotonic()
            mem={k:int(v.split()[0]) for k,v in
                 (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                 if k in ('MemAvailable','MemFree','SwapFree','SwapTotal')}
            processes=[]
            for d in Path('/proc').iterdir():
                if not d.name.isdigit():continue
                try:
                    name=(d/'comm').read_text().strip()
                    if name not in ('ros2_control_no','python3','cc1plus','ld'):continue
                    status=(d/'status').read_text().splitlines()
                    processes.append({'pid':int(d.name),'name':name,
                                      'rss':[v for v in status if v.startswith('VmRSS:')],
                                      'stat':(d/'stat').read_text().strip()})
                except (OSError,ValueError):pass
            row={'time_unix':time.time(),'boot_id':boot,'uptime':Path('/proc/uptime').read_text().strip(),
                 'load':list(os.getloadavg()),'memory_kb':mem,'processes':processes,
                 'throttled':command('vcgencmd','get_throttled'),
                 'temperature':command('vcgencmd','measure_temp'),
                 'pmic':command('vcgencmd','pmic_read_adc')}
            line=json.dumps(row);f.write(line+'\n');f.flush();os.fsync(f.fileno())
            print(line,flush=True)
            time.sleep(max(0,1-(time.monotonic()-tick)))
if __name__=='__main__':main()
