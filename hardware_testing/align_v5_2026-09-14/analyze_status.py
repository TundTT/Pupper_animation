import rosbag2_py, json
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float64MultiArray
r=rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri='/home/pi/robot-code-leglift/hardware_testing/align_v5_2026-09-14/trial_bag',storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
r.set_filter(rosbag2_py.StorageFilter(topics=['/neural_controller_wheel_align_hybrid/alignment_status']))
segments=[]; cur=None
while r.has_next():
    topic,raw,t=r.read_next(); d=list(deserialize_message(raw,Float64MultiArray).data)
    key=(int(d[2]),int(d[0]))
    if cur is None or cur['key']!=key:
        cur={'key':key,'start':t/1e9,'rows':[]};segments.append(cur)
    cur['end']=t/1e9;cur['rows'].append(d)
for s in segments:
    rows=s.pop('rows'); mature=[d for d in rows if d[3]>=.999] or rows
    s['seconds']=round(s['end']-s['start'],2);s['samples']=len(rows)
    s['mature_samples']=len(mature)
    s['margin_minmax_mm']=[[round(min(d[i] for d in mature)*1000,2),round(max(d[i] for d in mature)*1000,2)] for i in (4,5,6)]
    s['margin_pass_fraction']=[round(sum(d[i]>v for d in mature)/len(mature),3) for i,v in [(4,.01),(5,.01),(6,.005)]]
    s['rotation_samples']=sum(d[7]>0 for d in rows)
    s['error_start_end']=[rows[0][9],rows[-1][9]]
    s['residual_end']=rows[-1][10];s['completed_end']=rows[-1][8]
print(json.dumps(segments,indent=2))
