from types import SimpleNamespace
import numpy as np
from workspace.export_walk import convert_walk_params


def test_export_matches_normalized_network_with_constant_upright_channels():
    rng = np.random.default_rng(23)
    mean = rng.normal(0, .1, 144)
    std = rng.uniform(.03, .3, 144)
    fixed = np.concatenate([np.arange(9, 12) + 36*i for i in range(4)])
    mean[fixed] = np.tile([0., 0., 1.], 4)
    std[fixed] = 1e-6
    first = rng.normal(0, .1, (144, 8)).astype(np.float32)
    last = rng.normal(0, .1, (8, 24)).astype(np.float32)
    b1 = rng.normal(0, .1, 8).astype(np.float32)
    b2 = rng.normal(0, .1, 24).astype(np.float32)
    params = (SimpleNamespace(mean=mean, std=std), {'params': {
        'hidden_0': {'kernel': first, 'bias': b1},
        'hidden_1': {'kernel': last, 'bias': b2},
    }})
    exported = convert_walk_params(params, 4, 'elu')
    assert np.all(np.array(exported['layers'][0]['weights'][0])[fixed] == 0)
    for _ in range(8):
        obs = mean + std*rng.normal(0, .2, 144)
        obs[fixed] = mean[fixed]
        hidden = ((obs-mean)/std) @ first + b1
        hidden = np.where(hidden > 0, hidden, np.expm1(np.minimum(hidden, 0)))
        expected = np.tanh((hidden @ last + b2)[:12])
        actual = obs.astype(np.float32)
        for layer in exported['layers']:
            actual = actual @ np.array(layer['weights'][0], dtype=np.float32) + np.array(layer['weights'][1], dtype=np.float32)
            actual = np.tanh(actual) if layer['activation'] == 'tanh' else np.where(actual > 0, actual, np.expm1(np.minimum(actual, 0)))
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_action_scale_warm_start_preserves_history_normalization_and_small_targets():
    from jax import numpy as jp
    from brax.training.acme import running_statistics
    from workspace.walk_warm_start import rescale_action_head
    stats=running_statistics.init_state(jp.zeros(144)).replace(mean=jp.full(144,.2),std=jp.full(144,.3),summed_variance=jp.full(144,9.))
    params=(stats,{'params':{'head':{'kernel':jp.zeros((8,24)),'bias':jp.concatenate([jp.full(12,.1),jp.full(12,-3.)])}}},{})
    old=np.array([.5,.25,.5]*4);new=np.array([.5,.25,1.1]*4)
    result=rescale_action_head(params,old,new,4)
    obs=np.linspace(-.1,.5,144);changed=obs.copy()
    indices=np.concatenate([np.arange(24,36)+36*i for i in range(4)])
    changed[indices]*=np.tile(old/new,4)
    np.testing.assert_allclose((obs-stats.mean)/stats.std,(changed-result[0].mean)/result[0].std,atol=1e-6)
    before=np.tanh(np.asarray(params[1]['params']['head']['bias'][:12]))*old
    after=np.tanh(np.asarray(result[1]['params']['head']['bias'][:12]))*new
    np.testing.assert_allclose(before,after,atol=.0002)
    np.testing.assert_allclose(params[1]['params']['head']['bias'][:12],.1)
