import torch
from minisgl.layers.norm import DistributedRMSNorm
from minisgl.models.weight import _shard_tensor


class _AddRemoteSquareSum:
    def __init__(self, remote_square_sum):
        self.remote_square_sum = remote_square_sum

    def all_reduce(self, value):
        return value + self.remote_square_sum


def test_distributed_rmsnorm_matches_full_projection_reference():
    norm = object.__new__(DistributedRMSNorm)
    norm.eps = 1e-6
    norm.full_size = 4
    norm.weight = torch.tensor([1.5, 0.5])
    norm._comm = _AddRemoteSquareSum(torch.tensor([[25.0]]))
    local = torch.tensor([[1.0, 2.0]])

    norm.forward_inplace(local)

    full = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    inv_rms = torch.rsqrt(full.square().mean(dim=-1, keepdim=True) + norm.eps)
    expected = full[:, :2] * inv_rms * norm.weight
    assert torch.allclose(local, expected, rtol=1e-6, atol=1e-6)


def test_distributed_rmsnorm_rounds_only_after_weight_multiply():
    torch.manual_seed(42)
    full = torch.randn(32, 8).to(torch.bfloat16)
    local = full[:, :4].clone()
    remote_square_sum = full[:, 4:].float().square().sum(dim=-1, keepdim=True)
    norm = object.__new__(DistributedRMSNorm)
    norm.eps = 1e-6
    norm.full_size = 8
    norm.weight = torch.randn(4).to(torch.bfloat16)
    norm._comm = _AddRemoteSquareSum(remote_square_sum)

    norm.forward_inplace(local)

    inv_rms = torch.rsqrt(full.float().square().mean(dim=-1, keepdim=True) + norm.eps)
    expected = (full[:, :4].float() * inv_rms * norm.weight.float()).to(torch.bfloat16)
    legacy = (full[:, :4].float() * inv_rms).to(torch.bfloat16) * norm.weight
    assert torch.equal(local, expected)
    assert not torch.equal(legacy, expected)


def test_olmo3_qk_norm_weights_are_sharded_without_affecting_qwen():
    weight = torch.arange(8)
    key = "model.layers.0.self_attn.q_norm.weight"

    rank0 = _shard_tensor(key, weight, 0, 2, 2, "olmo3")
    rank1 = _shard_tensor(key, weight, 1, 2, 2, "olmo3")
    qwen = _shard_tensor(key, weight, 1, 2, 2, "qwen3")

    assert torch.equal(rank0, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(rank1, torch.tensor([4, 5, 6, 7]))
    assert torch.equal(torch.cat((rank0, rank1)), weight)
    assert qwen is weight


def test_olmo3_post_norm_weights_remain_replicated():
    weight = torch.arange(8)
    key = "model.layers.0.post_attention_layernorm.weight"

    sharded = _shard_tensor(key, weight, 1, 2, 2, "olmo3")

    assert sharded is weight
