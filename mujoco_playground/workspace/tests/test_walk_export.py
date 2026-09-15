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
