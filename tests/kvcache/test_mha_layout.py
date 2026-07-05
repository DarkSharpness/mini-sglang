"""Gate 1.7f: hermetic tests for layout-aware MHAKVCache.

Most of the tests are purely source-level: importing the actual pool would
pull in torch + Ascend deps, which are unavailable on some dev hosts.
Instead we ``ast``-parse the three touched modules and assert:

1. ``MHAKVCache.__init__`` gets a ``layout`` kwarg defaulting to ``"nhd"``,
   validates it, and stores metadata.
2. The two allocation shapes appear textually in the two branches.
3. ``store_kv`` dispatches on ``self._layout``: the nhd branch keeps the
   ``store_cache`` CUDA path; the bnbsd branch scatters via pure PyTorch
   advanced indexing and does **not** import ``minisgl.kernel``.
4. Cross-page raw-slot conversion produces the expected page_ids/offsets.
5. ``create_kvcache_pool`` forwards an explicit ``layout`` kwarg (default
   ``"nhd"``) and never calls ``get_device_type``.
6. ``Engine.__init__`` picks the layout via a ternary on ``self.device_type``
   and passes it to ``create_kvcache_pool``.

Gate 1.7f-fix adds one **runtime** test that instantiates ``MHAKVCache`` on
CPU with ``layout="bnbsd"`` and drives real numerical scatter through
``store_kv``, with an ``__import__`` guard that fails immediately if the
bnbsd path attempts to touch ``minisgl.kernel``. It skips gracefully when
torch is not installed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MHA_PATH = _REPO_ROOT / "python" / "minisgl" / "kvcache" / "mha_pool.py"
_FACTORY_PATH = _REPO_ROOT / "python" / "minisgl" / "kvcache" / "__init__.py"
_ENGINE_PATH = _REPO_ROOT / "python" / "minisgl" / "engine" / "engine.py"


# ---------- source helpers ---------------------------------------------------


def _mha_source() -> str:
    return _MHA_PATH.read_text()


def _mha_tree() -> ast.Module:
    return ast.parse(_mha_source())


def _mha_cls() -> ast.ClassDef:
    return next(
        node for node in _mha_tree().body
        if isinstance(node, ast.ClassDef) and node.name == "MHAKVCache"
    )


def _mha_method(name: str) -> ast.FunctionDef:
    return next(
        node for node in _mha_cls().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _factory_source() -> str:
    return _FACTORY_PATH.read_text()


def _factory_tree() -> ast.Module:
    return ast.parse(_factory_source())


def _factory_fn() -> ast.FunctionDef:
    return next(
        node for node in _factory_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "create_kvcache_pool"
    )


def _engine_source() -> str:
    return _ENGINE_PATH.read_text()


def _engine_tree() -> ast.Module:
    return ast.parse(_engine_source())


def _engine_init() -> ast.FunctionDef:
    engine_cls = next(
        node for node in _engine_tree().body
        if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )
    return next(
        node for node in engine_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )


# ---------- MHAKVCache: layout kwarg & validation ---------------------------


def test_mha_kv_cache_init_has_layout_kwarg_defaulting_to_nhd():
    init = _mha_method("__init__")
    args = init.args
    kwargs_by_name = {a.arg: default for a, default in zip(args.args, [None] * (len(args.args) - len(args.defaults)) + list(args.defaults))}
    assert "layout" in kwargs_by_name, "MHAKVCache.__init__ must accept 'layout'"
    default = kwargs_by_name["layout"]
    assert isinstance(default, ast.Constant) and default.value == "nhd", \
        f"layout default must be the string 'nhd', got {ast.dump(default)!r}"


def test_mha_kv_cache_layout_annotation_is_literal_nhd_bnbsd():
    init = _mha_method("__init__")
    layout_arg = next(a for a in init.args.args if a.arg == "layout")
    assert layout_arg.annotation is not None, "layout must be annotated"
    text = ast.unparse(layout_arg.annotation)
    assert "Literal" in text and "'nhd'" in text and "'bnbsd'" in text, \
        f"layout annotation must be Literal['nhd', 'bnbsd'], got {text!r}"


def test_mha_kv_cache_raises_value_error_on_invalid_layout():
    init = _mha_method("__init__")
    raises = [
        node for node in ast.walk(init)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "ValueError"
    ]
    assert raises, "MHAKVCache.__init__ must raise ValueError on invalid layout"


def test_mha_kv_cache_nhd_allocation_shape_present():
    src = _mha_source()
    assert "(2, num_layers, num_pages, page_size, local_kv_heads, head_dim)" in src, \
        "nhd branch must allocate the 6D NHD shape"


def test_mha_kv_cache_bnbsd_allocation_shape_present():
    src = _mha_source()
    assert "(2, num_layers, num_pages, local_kv_heads, page_size, head_dim)" in src, \
        "bnbsd branch must allocate the 6D BnNBsD shape"


def test_mha_kv_cache_stores_layout_metadata():
    src = _mha_source()
    for attr in ("self._layout", "self._page_size", "self._local_kv_heads", "self._head_dim"):
        assert attr in src, f"MHAKVCache must persist {attr}"


# ---------- store_kv: layout dispatch ---------------------------------------


def test_store_kv_dispatches_on_self_layout():
    method = _mha_method("store_kv")
    text = ast.unparse(method)
    assert "self._layout" in text, "store_kv must branch on self._layout"


def test_store_kv_nhd_branch_still_calls_store_cache():
    method = _mha_method("store_kv")
    text = ast.unparse(method)
    assert "from minisgl.kernel import store_cache" in text, \
        "nhd branch must still import store_cache"
    assert "store_cache(" in text, "nhd branch must still call store_cache"


def _store_kv_top_level_if() -> ast.If:
    method = _mha_method("store_kv")
    for node in method.body:
        if isinstance(node, ast.If):
            return node
    raise AssertionError("store_kv must contain a top-level if branching on layout")


def test_store_kv_bnbsd_branch_does_not_import_or_call_store_cache():
    if_node = _store_kv_top_level_if()
    else_text = "\n".join(ast.unparse(n) for n in if_node.orelse)
    assert "store_cache" not in else_text, "bnbsd branch must NOT reference store_cache"
    assert "minisgl.kernel" not in else_text, "bnbsd branch must NOT import minisgl.kernel"


def test_store_kv_bnbsd_branch_uses_advanced_indexing_scatter():
    if_node = _store_kv_top_level_if()
    else_text = "\n".join(ast.unparse(n) for n in if_node.orelse)
    assert "out_loc // self._page_size" in else_text, \
        "bnbsd branch must derive page_ids as out_loc // self._page_size"
    assert "out_loc % self._page_size" in else_text, \
        "bnbsd branch must derive offsets as out_loc % self._page_size"
    assert "self._k_buffer[layer_id][page_ids, :, offsets, :]" in else_text, \
        "bnbsd branch must scatter K via advanced indexing on [page_ids, :, offsets, :]"
    assert "self._v_buffer[layer_id][page_ids, :, offsets, :]" in else_text, \
        "bnbsd branch must scatter V via advanced indexing on [page_ids, :, offsets, :]"


def test_store_kv_bnbsd_branch_reshapes_kv_to_3d():
    if_node = _store_kv_top_level_if()
    else_text = "\n".join(ast.unparse(n) for n in if_node.orelse)
    assert "k.view(-1, self._local_kv_heads, self._head_dim)" in else_text, \
        "bnbsd branch must reshape k to (-1, HKV, D)"
    assert "v.view(-1, self._local_kv_heads, self._head_dim)" in else_text, \
        "bnbsd branch must reshape v to (-1, HKV, D)"


# ---------- raw-slot arithmetic (pure Python, no torch) ---------------------


def test_bnbsd_raw_slot_to_page_id_and_offset_cross_page():
    """Spec case: raw_slots=[15, 16, 34] with page_size=16.

    Page IDs = raw // 16 → [0, 1, 2]; offsets = raw % 16 → [15, 0, 2]. The
    bnbsd branch performs exactly this arithmetic, so verify the maths
    matches the pattern present in the source without needing torch.
    """
    page_size = 16
    raw_slots = [15, 16, 34]
    page_ids = [s // page_size for s in raw_slots]
    offsets = [s % page_size for s in raw_slots]
    assert page_ids == [0, 1, 2]
    assert offsets == [15, 0, 2]
    # Sanity check the source literally uses `// self._page_size` and
    # `% self._page_size` (guards against a future refactor to something
    # semantically different but easy to miss).
    src = _mha_source()
    assert "out_loc // self._page_size" in src
    assert "out_loc % self._page_size" in src


def test_bnbsd_untouched_slots_are_left_to_torch_empty_semantics():
    """The bnbsd branch must not blanket-clear _k_buffer / _v_buffer — only
    the (page_ids, offsets) coordinates are written. Any assignment target
    referencing _k_buffer or _v_buffer in the bnbsd branch must be the
    advanced-index scatter form using layer_id, page_ids, and offsets.
    """
    if_node = _store_kv_top_level_if()
    for node in if_node.orelse:
        for assign in ast.walk(node):
            if isinstance(assign, ast.Assign):
                for tgt in assign.targets:
                    tgt_text = ast.unparse(tgt)
                    if "_k_buffer" in tgt_text or "_v_buffer" in tgt_text:
                        assert "layer_id" in tgt_text \
                            and "page_ids" in tgt_text \
                            and "offsets" in tgt_text, \
                            f"bnbsd write must be scoped to [layer_id][page_ids, :, offsets, :], got {tgt_text!r}"


# ---------- factory: layout param & no device inference ---------------------


def test_create_kvcache_pool_has_layout_kwarg_defaulting_to_nhd():
    fn = _factory_fn()
    args = fn.args
    kwargs_by_name = {a.arg: default for a, default in zip(args.args, [None] * (len(args.args) - len(args.defaults)) + list(args.defaults))}
    assert "layout" in kwargs_by_name, "create_kvcache_pool must accept 'layout'"
    default = kwargs_by_name["layout"]
    assert isinstance(default, ast.Constant) and default.value == "nhd", \
        f"factory layout default must be 'nhd', got {ast.dump(default)!r}"


def test_create_kvcache_pool_forwards_layout_to_mha_kv_cache():
    fn = _factory_fn()
    calls = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MHAKVCache"
    ]
    assert calls, "factory must instantiate MHAKVCache"
    kw_names = {kw.arg for kw in calls[0].keywords}
    assert "layout" in kw_names, "MHAKVCache must be called with layout=..."
    # And the argument value must be the local layout variable, not a
    # hard-coded constant, so device selection genuinely flows through.
    layout_kw = next(kw for kw in calls[0].keywords if kw.arg == "layout")
    assert isinstance(layout_kw.value, ast.Name) and layout_kw.value.id == "layout", \
        f"factory must forward layout=layout, got {ast.unparse(layout_kw.value)!r}"


def test_create_kvcache_pool_does_not_call_get_device_type():
    src = ast.unparse(_factory_fn())
    assert "get_device_type" not in src, \
        "factory must NOT infer layout via get_device_type()"


# ---------- Engine: layout selection & wiring -------------------------------


def _engine_layout_ifexp() -> ast.IfExp:
    """Find the ternary that selects 'bnbsd' vs 'nhd' in Engine.__init__."""
    init = _engine_init()
    for node in ast.walk(init):
        if isinstance(node, ast.IfExp) \
                and isinstance(node.body, ast.Constant) and node.body.value == "bnbsd" \
                and isinstance(node.orelse, ast.Constant) and node.orelse.value == "nhd":
            return node
    raise AssertionError(
        "Engine.__init__ must contain a ternary "
        "'bnbsd' if self.device_type == 'npu' else 'nhd'"
    )


def test_engine_selects_layout_via_device_type_ternary():
    ifexp = _engine_layout_ifexp()
    test_text = ast.unparse(ifexp.test)
    # Accept either ordering: `self.device_type == "npu"` or `"npu" == self.device_type`.
    assert "self.device_type" in test_text and '"npu"' in test_text.replace("'", '"'), \
        f"ternary test must compare self.device_type to 'npu', got {test_text!r}"


def test_engine_passes_layout_kwarg_to_create_kvcache_pool():
    init = _engine_init()
    calls = [
        node for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_kvcache_pool"
    ]
    assert calls, "Engine.__init__ must call create_kvcache_pool"
    kw_names = {kw.arg for kw in calls[0].keywords}
    assert "layout" in kw_names, \
        "create_kvcache_pool must be called with layout=..."


def test_engine_layout_kwarg_is_variable_not_literal():
    """Guards against a regression like ``layout="nhd"`` — the Engine must
    always forward the ternary result so npu → bnbsd, cuda/cpu → nhd."""
    init = _engine_init()
    call = next(
        node for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_kvcache_pool"
    )
    layout_kw = next(kw for kw in call.keywords if kw.arg == "layout")
    assert isinstance(layout_kw.value, ast.Name), \
        f"Engine must forward a variable to layout, got {ast.unparse(layout_kw.value)!r}"


# =========================================================================
# Gate 1.7f-fix: real CPU numerical test of the bnbsd store_kv path.
#
# Skipped when torch is not installed so the AST-only tests above still run
# on hosts without torch.
# =========================================================================


@pytest.fixture(scope="module")
def _tp_info_set():
    """Ensure DistributedInfo is initialised exactly once for the module.

    Skips gracefully when ``minisgl`` (i.e. the source tree isn't on
    ``PYTHONPATH``) or its transitive dependencies cannot be imported.
    """
    try:
        from minisgl.distributed import set_tp_info, try_get_tp_info
    except ImportError as exc:
        pytest.skip(f"minisgl not importable: {exc}")

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    yield


def test_bnbsd_store_kv_real_scatter_on_cpu(_tp_info_set):
    """Instantiate MHAKVCache(layout='bnbsd') on CPU and drive real scatter
    through store_kv. Verify:

    * three cross-page writes land at the correct (page, offset) coordinates;
    * sentinel-filled untouched slots survive;
    * per-layer view is contiguous and has BnNBsD shape ``(P, HKV, S, D)``;
    * the bnbsd branch does not import ``minisgl.kernel`` — enforced with an
      active ``__import__`` guard, not just AST inspection.
    """
    torch = pytest.importorskip("torch")

    try:
        from minisgl.kvcache.mha_pool import MHAKVCache
    except ImportError as exc:
        pytest.skip(f"minisgl.kvcache.mha_pool not importable: {exc}")

    num_pages, page_size, kv_heads, head_dim = 4, 16, 2, 4
    cache = MHAKVCache(
        num_kv_heads=kv_heads,
        num_layers=2,
        head_dim=head_dim,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
        layout="bnbsd",
    )

    # Fill entire KV buffer with a distinctive sentinel so we can detect
    # exactly which coordinates the scatter touches.
    cache._kv_buffer.fill_(-999.0)

    # Given test vectors from the spec.
    out_loc = torch.tensor([15, 16, 34], dtype=torch.int64)
    k = torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4)
    v = (1000 + torch.arange(3 * 2 * 4, dtype=torch.float32)).view(3, 8)

    # ---- Active import guard: fail fast if bnbsd store_kv touches minisgl.kernel.
    import builtins

    _orig_import = builtins.__import__
    forbidden_names = []

    def _forbid_kernel_import(name, *args, **kwargs):
        if name == "minisgl.kernel" or name.startswith("minisgl.kernel."):
            forbidden_names.append(name)
            raise AssertionError(
                f"bnbsd store_kv must NOT import {name!r}; the CUDA kernel "
                "path is nhd-only."
            )
        return _orig_import(name, *args, **kwargs)

    builtins.__import__ = _forbid_kernel_import
    try:
        cache.store_kv(k, v, out_loc, layer_id=1)
    finally:
        builtins.__import__ = _orig_import
    assert forbidden_names == [], \
        f"bnbsd path illegally imported: {forbidden_names!r}"

    # ---- Layer-1 view shape / contiguity.
    k_view_1 = cache.k_cache(1)
    v_view_1 = cache.v_cache(1)
    assert tuple(k_view_1.shape) == (num_pages, kv_heads, page_size, head_dim), \
        f"k_cache(1) shape mismatch: {tuple(k_view_1.shape)}"
    assert tuple(v_view_1.shape) == (num_pages, kv_heads, page_size, head_dim), \
        f"v_cache(1) shape mismatch: {tuple(v_view_1.shape)}"
    assert k_view_1.is_contiguous(), "k_cache(1) must be contiguous"
    assert v_view_1.is_contiguous(), "v_cache(1) must be contiguous"

    # ---- Written slots match input tensors.
    # loc 15 → page 0, offset 15
    # loc 16 → page 1, offset 0
    # loc 34 → page 2, offset 2
    expected = [(0, 15, 0), (1, 0, 1), (2, 2, 2)]
    for page, offset, token in expected:
        written_k = k_view_1[page, :, offset, :]
        written_v = v_view_1[page, :, offset, :]
        assert torch.equal(written_k, k[token]), (
            f"k mismatch at page={page}, offset={offset}: "
            f"got {written_k.tolist()} vs expected {k[token].tolist()}"
        )
        assert torch.equal(written_v, v[token].view(kv_heads, head_dim)), (
            f"v mismatch at page={page}, offset={offset}: "
            f"got {written_v.tolist()} vs expected "
            f"{v[token].view(kv_heads, head_dim).tolist()}"
        )
        # And the written values must NOT be the sentinel.
        assert not torch.any(written_k == -999.0)
        assert not torch.any(written_v == -999.0)

    # ---- Sentinel-untouched positions on layer 1 stay at -999.0.
    sentinel = torch.full((kv_heads, head_dim), -999.0, dtype=torch.float32)
    for page, offset in [(0, 0), (1, 1), (2, 1)]:
        assert torch.equal(k_view_1[page, :, offset, :], sentinel), \
            f"layer 1 page {page} offset {offset} K unexpectedly touched"
        assert torch.equal(v_view_1[page, :, offset, :], sentinel), \
            f"layer 1 page {page} offset {offset} V unexpectedly touched"

    # Entire page 3 on layer 1 must remain untouched.
    layer1_page3_sentinel = torch.full(
        (kv_heads, page_size, head_dim), -999.0, dtype=torch.float32
    )
    assert torch.equal(k_view_1[3], layer1_page3_sentinel), \
        "layer 1 page 3 K was unexpectedly written to"
    assert torch.equal(v_view_1[3], layer1_page3_sentinel), \
        "layer 1 page 3 V was unexpectedly written to"

    # ---- Layer 0 must be completely untouched — every K and V slot is -999.0.
    layer0_sentinel = torch.full(
        (num_pages, kv_heads, page_size, head_dim),
        -999.0,
        dtype=torch.float32,
    )
    assert torch.equal(cache.k_cache(0), layer0_sentinel), \
        "layer 0 K cache was unexpectedly written to"
    assert torch.equal(cache.v_cache(0), layer0_sentinel), \
        "layer 0 V cache was unexpectedly written to"
