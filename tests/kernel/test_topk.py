import torch
from minisgl.kernel import fast_topk, fast_topk_transform
from minisgl.utils import call_if_main


def _torch_topk(
    score: torch.Tensor,
    clip: int,
):
    topk = min(2048, clip)
    answer = torch.topk(score[:, :clip], topk, dim=-1, sorted=False).indices
    if clip < 2048:
        pad_size = 2048 - clip
        pad = torch.full((score.size(0), pad_size), -1, dtype=torch.int32, device=score.device)
        answer = torch.cat([answer, pad], dim=-1)
    return answer


@call_if_main(__name__)
def test_fast_topk():
    torch.manual_seed(0)
    B = 132 * 3 + 2
    clip = 8192 * 8
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(B, 1310172, dtype=torch.float32, device="cuda")
    indices = torch.full((B, 2048), -2, dtype=torch.int32, device="cuda")
    lengths = torch.full((B,), clip, dtype=torch.int32, device="cuda")
    fast_topk(score, indices, lengths)
    # sort indices by last dimension
    indices = indices.sort(dim=-1).values
    # find the pos where -2 is in indices
    answer = _torch_topk(score, clip).sort(dim=-1).values

    # check how many different in each row
    indice_cpu = indices.cpu().tolist()
    answer_cpu = answer.cpu().tolist()

    for i in range(B):
        more = set(indice_cpu[i]) - set(answer_cpu[i])
        less = set(answer_cpu[i]) - set(indice_cpu[i])
        if len(more) > 0 or len(less) > 0:
            print(f"Row {i} differs:")
            print(f"  more: {more}")
            print(f"  less: {less}")
            # print the more values
            source = score[i].cpu()
            print(f"  more values: {[source[j].item() for j in more if j >= 0]}")
            print(f"  less values: {[source[j].item() for j in less if j >= 0]}")

    # test performance
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    # use a large GEMM to warm up GPU
    def perf(f):
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        _ = a @ b
        tic.record()
        for _ in range(100):
            f()
        toc.record()
        torch.cuda.synchronize()
        return tic.elapsed_time(toc) / 100

    t0 = perf(lambda: fast_topk(score, indices, lengths))
    t1 = perf(lambda: torch.topk(score[:, :clip], 2048, dim=-1, sorted=False))
    print(f"fast_topk: {t0} ms, torch.topk: {t1} ms")


@call_if_main(__name__)
def test_fast_topk_transform():
    torch.manual_seed(0)
    B = 32
    clip = 50000
    INT_MAX = 2147483647
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(B, 100000, dtype=torch.float32, device="cuda")
    lengths = torch.full((B,), clip, dtype=torch.int32, device="cuda")
    src_page_table = torch.randint(0, INT_MAX, (B, clip), dtype=torch.int32, device="cuda")
    dst_page_table = torch.full((B, 2048), -1, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * clip
    fast_topk_transform(
        score=score,
        lengths=lengths,
        dst_page_table=dst_page_table,
        src_page_table=src_page_table,
        cu_seqlens=cu_seqlens,
    )
