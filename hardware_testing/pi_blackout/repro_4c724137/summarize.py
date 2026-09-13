import json,re
from pathlib import Path
root=Path(__file__).resolve().parent
def read(name):
    rows=[]
    for line in (root/name).read_text(encoding='utf-8-sig').splitlines():
        try:rows.append(json.loads(line))
        except (ValueError,TypeError):pass # SSH copy may have an incomplete last line.
    return rows
health=read('health_laptop.jsonl');results={}
for direction,name in [('up','up_retry_laptop.jsonl'),('down','down_laptop.jsonl')]:
    rows=read(name);results[direction]=next(r for r in reversed(rows) if r.get('completed'))
volts=[float(m.group(1)) for r in health if (m:=re.search(r'EXT5V_V volt\(24\)=([0-9.]+)V',r['pmic'].get('output','')))]
temps=[float(m.group(1)) for r in health if (m:=re.search(r'temp=([0-9.]+)',r['temperature'].get('output','')))]
report={'motions':results,'health_samples':len(health),'first_health_time':health[0]['time_unix'],
        'last_health_time':health[-1]['time_unix'],'boot_ids':sorted({r['boot_id'] for r in health}),
        'throttle_readings':sorted({r['throttled'].get('output','') for r in health}),
        'temperature_range_c':[min(temps),max(temps)],'sampled_ext5v_range_v':[min(volts),max(volts)],
        'min_available_memory_kb':min(r['memory_kb']['MemAvailable'] for r in health),
        'blackout_reproduced':False,'second_power_cycle_comparison':'see reboot_comparison.json' if (root/'reboot_comparison.json').exists() else 'pending',
        'limitations':'One supervised stand test. One-Hz power samples cannot exclude brief transients; failing GUI/voice services were stopped, and upper joints retained the calibrated hanging pose.'}
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='motions'},indent=2))
