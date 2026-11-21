import mlx.core as mx
from inkling_kernel import mix

for seed in range(8):
    mx.random.seed(seed)
    states = mx.random.normal((1, 1, 8, 2304)).astype(mx.bfloat16)
    weights = mx.softmax(mx.random.normal((1, 1, 8)), axis=-1)
    actual = mix(states, weights, states.dtype)
    expected = (states.astype(mx.float32) * weights[..., None]).sum(-2).astype(states.dtype)
    assert mx.allclose(actual, expected, atol=0.008, rtol=0.008).item()
