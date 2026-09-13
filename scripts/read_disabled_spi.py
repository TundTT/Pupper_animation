#!/usr/bin/env python3
"""Bounded diagnostic: exchange only all-zero, motor-disabled SPI packets.

No ROS startup, homing, enable bit, gain or torque command. Firmware may return
cached values while disabled; these packets alone do not prove fresh motor feedback.
Wire layout and byte swapping follow control_board_hardware_interface/src/rt/rt_spi.cpp.
"""
import argparse,json,math,struct,subprocess,time
from pathlib import Path

def decode(packet):
    if len(packet)!=132:raise ValueError('Incomplete SPI transaction')
    raw=bytearray(packet[:60])
    for i in range(0,60,2):raw[i],raw[i+1]=raw[i+1],raw[i]
    words=struct.unpack('<15I',raw);checksum=0
    for word in words[:14]:checksum^=word
    if checksum!=words[14]:raise ValueError('Invalid reply checksum')
    values=struct.unpack('<12f2iI',raw)
    if not all(math.isfinite(v) for v in values[:12]):raise ValueError('Nonfinite feedback')
    return {'q':list(values[:6]),'qd':list(values[6:12]),'flags':list(values[12:14]),
            'all_zero_packet':not any(raw),'wire_hex':raw.hex()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous evidence')
    busy=subprocess.run(['fuser','/dev/spidev0.0','/dev/spidev0.1'],capture_output=True)
    if busy.returncode!=1:raise RuntimeError('SPI in use or ownership check failed')
    import spidev
    buses=[];rows=[]
    try:
        for channel in (0,1):
            s=spidev.SpiDev();s.open(0,channel);s.mode=0;s.max_speed_hz=6000000;s.bits_per_word=8;buses.append(s)
        for _ in range(100):
            boards=[decode(s.xfer2([0]*132)) for s in buses]
            rows.append({'time_unix':time.time(),'boards':boards});time.sleep(.02)
    finally:
        for s in buses:s.close()
    record={'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'purpose':'disabled_spi_observation','all_tx_bytes_zero':True,'motor_enabled':False,
            'feedback_freshness_proven':False,'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps({'output':str(a.output),'boot_id':record['boot_id'],
                      'samples':len(rows),'first':rows[0],'last':rows[-1]}))
if __name__=='__main__':main()
