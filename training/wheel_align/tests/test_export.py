"""Randomly initialized network fixtures validate export; no learning occurs."""
import json
import os
import subprocess
import jax
from jax import numpy as jp
import numpy as np
from brax.training.acme import running_statistics,specs
from brax.training.agents.ppo import networks
from training.wheel_align.train import network_factory
from training.wheel_align.export import payload_from_params,numpy_inference

def test_export(tmp_path):
    net=network_factory()(82,8,preprocess_observations_fn=running_statistics.normalize)
    stats=running_statistics.init_state(specs.Array((82,),jp.float32))
    params=(stats,net.policy_network.init(jax.random.PRNGKey(42)))
    payload=payload_from_params(params,dict(source_commit='UNTRAINED_TEST_FIXTURE',source_hashes={}))
    x=np.random.default_rng(12).normal(0,.3,(128,82)).astype(np.float32)
    infer=networks.make_inference_fn(net)(params,deterministic=True)
    expected=np.asarray(infer(jp.asarray(x),jax.random.PRNGKey(0))[0])
    np.testing.assert_allclose(numpy_inference(payload,x),expected,atol=3e-5,rtol=0)
    output=tmp_path/'untrained-test-fixture.json';output.write_text(json.dumps(payload))
    fixtures=tmp_path/'untrained-test-fixture.csv';np.savetxt(fixtures,np.c_[x,expected],delimiter=',',fmt='%.9g')
    exe=os.environ.get('ALIGN_EXPORT_TEST_EXE')
    if exe:subprocess.run([exe,str(output),str(fixtures)],check=True)
