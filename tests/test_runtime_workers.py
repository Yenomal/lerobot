import torch

from lerobot.runtime_workers import _decode_cache, _encode_cache


def test_dynamic_cache_codec_round_trips_cpu_tensor_layers():
    from transformers import DynamicCache

    keys = torch.randn(1, 2, 3, 4)
    values = torch.randn(1, 2, 3, 4)
    cache = DynamicCache(ddp_cache_data=((keys, values),))

    encoded = _encode_cache(cache)
    decoded = _decode_cache(encoded, "cpu")

    assert len(decoded.layers) == 1
    torch.testing.assert_close(decoded.layers[0].keys, keys)
    torch.testing.assert_close(decoded.layers[0].values, values)
