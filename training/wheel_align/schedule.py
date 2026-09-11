"""Completion-driven command sequencing shared by training, audit and video."""
import numpy as np
from . import configs as c,contract as ct

def reset(xp=np):
    return dict(index=xp.asarray(0),age=xp.asarray(0),interrupted=xp.asarray(False),
                timeouts=xp.zeros(4,dtype=bool))

def advance(s,motion,order,interrupt=False,count=4,xp=np):
    s=dict(s);idx=xp.minimum(s['index'],3);command=order[idx]
    k=xp.asarray(ct.COMMAND_LEG)[command]
    finished=motion['completed'][k]&(motion['phase']==ct.HOLD)
    timeout=(s['age']>=c.LEG_TIMEOUT_STEPS+xp.where(s['interrupted'],20*52,0))&~finished&(s['index']<count)
    advance=(finished|timeout)&(s['index']<count)
    s['timeouts']=s['timeouts']|((xp.arange(4)==k)&timeout)
    s['index']+=advance.astype(int)
    s['age']=xp.where(advance,0,s['age']+1)
    s['interrupted']=xp.where(advance,False,s['interrupted'])
    # Deliberate interruption happens once per requested wheel. The same wheel
    # is retried after its supported descent; it is not silently counted done.
    cut=interrupt&(s['age']>=520)&(s['age']<624)&(s['index']<count)
    s['interrupted']=s['interrupted']|cut
    command=order[xp.minimum(s['index'],3)]
    command=xp.where((s['age']<104)|cut|(s['index']>=count),0,command)
    return s,command
