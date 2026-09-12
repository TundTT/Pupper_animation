import json,hashlib,sys
from pathlib import Path
root=Path('runs/inverted_triangle');out=root/sys.argv[1];out.mkdir(exist_ok=False)
base=Path('motion/inverted_triangle/results/pc_20260912/plan');p=json.loads((base/'plan.json').read_text())
names=sys.argv[2:]
for i,e in enumerate(p['stages']):
 src=base/e['candidate'] if names[i]=='base' else root/names[i]/'candidate.json'
 c=json.loads(src.read_text());c.pop('start_state',None)
 dest=out/e['candidate'];dest.write_text(json.dumps(c,indent=2));e['candidate_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
p['parent_plan_sha256']=hashlib.sha256((base/'plan.json').read_bytes()).hexdigest()
p['selection_sources']=names
(out/'plan.json').write_text(json.dumps(p,indent=2));print(out/'plan.json')
