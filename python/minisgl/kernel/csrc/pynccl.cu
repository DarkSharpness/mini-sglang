#include "nccl227.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/from_blob.h>
#include <algorithm>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string_view>
#include <torch/python.h>
#include <unordered_map>
#include <vector>

namespace {

using std::uint8_t;
using namespace std::string_view_literals;

using ncclIDWrapper = std::vector<uint8_t>;

auto ncclID2Wrapper(ncclUniqueId id) -> ncclIDWrapper {
  return ncclIDWrapper(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

auto wrapper2NcclID(ncclIDWrapper wrapper) -> ncclUniqueId {
  TORCH_CHECK(wrapper.size() == NCCL_UNIQUE_ID_BYTES,
              "Invalid ncclIDWrapper size: expected ", NCCL_UNIQUE_ID_BYTES,
              ", got ", wrapper.size());
  ncclUniqueId id;
  std::copy(wrapper.begin(), wrapper.end(), id.internal);
  return id;
}

const auto Nccl_name_map = std::unordered_map{
    std::pair{"sum"sv, ncclSum},
    {"prod", ncclProd},
    {"max", ncclMax},
    {"min", ncclMin},
};

const auto Nccl_dtype_map = std::unordered_map{
    std::pair{torch::kFloat32, ncclFloat32},
    {torch::kFloat16, ncclFloat16},
    {torch::kBFloat16, ncclBfloat16},
    {torch::kInt32, ncclInt32},
    {torch::kInt64, ncclInt64},
};

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ::ncclResult_t result = cmd;                                               \
    TORCH_CHECK(result == ::ncclSuccess,                                       \
                "NCCL error: ", ::ncclGetErrorString(result));                 \
  } while (0)

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    ::cudaError_t error = cmd;                                                 \
    TORCH_CHECK(error == ::cudaSuccess,                                        \
                "CUDA error: ", ::cudaGetErrorString(error));                  \
  } while (0)

template <typename T>
using shared_ref = std::shared_ptr<std::remove_pointer_t<T>>;

struct NCCLWrapper {
public:
  NCCLWrapper(int rank, int N, const size_t max_bytes, ncclIDWrapper uid,
              c10::Device device, bool fallback = false, size_t buf_count = 1)
      : m_comm(nullptr), m_buf(nullptr), m_win(nullptr),
        // const members
        m_rank(rank), m_world_size(N), m_device(device), m_fallback(fallback),
        m_buf_count(buf_count), m_max_bytes(max_bytes) {
    TORCH_CHECK(0 <= rank && rank < N, "Invalid rank: ", rank,
                ", expected in range [0, ", N, ")");
    TORCH_CHECK(buf_count >= 1, "Buffer count should be non-zero");

    // initialize NCCL communicator
    ncclComm_t comm;
    NCCL_CHECK(::ncclCommInitRank(&comm, N, wrapper2NcclID(uid), rank));
    m_comm = shared_ref<ncclComm_t>(comm, ::ncclCommDestroy);

    // allocate NCCL symmetric buffer
    const auto buf_size = max_bytes * buf_count;
    void *buf;
    NCCL_CHECK(::ncclMemAlloc(&buf, buf_size));
    m_buf = std::shared_ptr<void>(buf, ::ncclMemFree);

    // register NCCL window
    ncclWindow_t win;
    NCCL_CHECK(::ncclCommWindowRegister(comm, buf, buf_size, &win,
                                        NCCL_WIN_COLL_SYMMETRIC));
    m_win = shared_ref<ncclWindow_t>(win, [comm = m_comm](ncclWindow_t p) {
      // hold a reference of comm to ensure comm is still alive when
      ::ncclCommWindowDeregister(comm.get(), p);
    });

    // initialize buffer usage tracker
    m_used = std::shared_ptr<bool[]>{new bool[buf_count]{},
                                     std::default_delete<bool[]>()};
  }

  ~NCCLWrapper() noexcept = default;

  NCCLWrapper(const NCCLWrapper &) = delete;
  NCCLWrapper &operator=(const NCCLWrapper &) = delete;
  NCCLWrapper &operator=(NCCLWrapper &&) = delete;
  NCCLWrapper(NCCLWrapper &&other) noexcept
      : m_comm(std::exchange(other.m_comm, nullptr)),
        m_buf(std::exchange(other.m_buf, nullptr)),
        m_win(std::exchange(other.m_win, nullptr)),
        m_used(std::exchange(other.m_used, nullptr)),
        // const members
        m_rank(other.m_rank), m_world_size(other.m_world_size),
        m_device(other.m_device), m_fallback(other.m_fallback),
        m_buf_count(other.m_buf_count), m_max_bytes(other.m_max_bytes) {}

private:
  static auto _s_check_tensor(const torch::Tensor &input, const char *where)
      -> void {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), where,
                ": Input tensor must be a contiguous CUDA tensor");
  }

  static auto _s_get_dtype(const torch::Tensor &input) -> ncclDataType_t {
    const auto dtype = input.scalar_type();
    const auto iter_dtype = Nccl_dtype_map.find(dtype);
    TORCH_CHECK(iter_dtype != Nccl_dtype_map.end(),
                "Unsupported data type for NCCL operation: ", dtype);
    return iter_dtype->second;
  }

  static auto _s_get_op(std::string_view op_str) -> ncclRedOp_t {
    const auto iter_op = Nccl_name_map.find(op_str);
    TORCH_CHECK(iter_op != Nccl_name_map.end(),
                "Unsupported reduction operation: ", op_str);
    return iter_op->second;
  }

  static auto _s_size_bytes(torch::Tensor tensor) -> size_t {
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
  }

  auto _m_can_use_buffer(std::size_t nbytes) const -> bool {
    return nbytes <= m_max_bytes * m_buf_count;
  }

  auto _m_check_fallback(const char *where, std::size_t nbytes) const -> void {
    TORCH_CHECK(m_fallback, where,
                ": Input tensor is too large for the internal buffer (", nbytes,
                " > ", m_max_bytes,
                "). Set fallback=True to allow in-place "
                "operations on the input tensor.");
  }

  auto _m_need_memcpy(const std::byte *src) const -> bool {
    const auto buf = static_cast<std::byte *>(m_buf.get());
    return src < buf || src >= buf + m_max_bytes * m_buf_count;
  }

  auto _m_allocate_buffer(torch::Tensor shape) -> torch::Tensor {
    TORCH_CHECK(_m_can_use_buffer(_s_size_bytes(shape)),
                "_m_allocate_buffer: requested size exceeds buffer capacity");
    for (const auto i : c10::irange(m_buf_count)) {
      if (!m_used[i]) {
        m_used[i] = true;
        return torch::from_blob(
            static_cast<std::byte *>(m_buf.get()) + i * m_max_bytes,
            shape.sizes(), [used = m_used, i](void *p) { used[i] = false; },
            shape.options().device(m_device));
      }
    }
    TORCH_CHECK(false, "Failed to allocate NCCL buffer");
  }

public:
  auto all_reduce(torch::Tensor input, const std::string &op_str = "sum")
      -> torch::Tensor {
    _s_check_tensor(input, "NCCLWrapper::all_reduce");

    const auto op = _s_get_op(op_str);
    const auto dtype = _s_get_dtype(input);
    const auto numel = static_cast<size_t>(input.numel());
    const auto nbytes = numel * input.element_size();
    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const auto src = static_cast<std::byte *>(input.data_ptr());
    const auto comm = m_comm.get();

    // fallback when 1. input is too large or 2. input is not in buffer
    // we print error for fallback if either condition is true
    if (!_m_can_use_buffer(nbytes) || _m_need_memcpy(src))
      _m_check_fallback("NCCLWrapper::all_reduce", nbytes);

    NCCL_CHECK(::ncclAllReduce(src, src, numel, dtype, op, comm, stream));
    return input;
  }

  auto all_gather(const torch::Tensor input) -> torch::Tensor {
    _s_check_tensor(input, "NCCLWrapper::all_gather");

    const auto dtype = _s_get_dtype(input);
    const auto numel = static_cast<size_t>(input.numel());
    const auto nbytes = numel * input.element_size();
    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const auto buf = static_cast<std::byte *>(m_buf.get());
    const auto src = static_cast<std::byte *>(input.data_ptr());
    const auto comm = m_comm.get();

    auto output_sizes = input.sizes().vec();
    output_sizes[0] *= m_world_size;
    auto output = torch::empty(output_sizes, input.options());
    const auto dst = output.data_ptr();
    NCCL_CHECK(::ncclAllGather(src, dst, numel, dtype, comm, stream));
    return output;
  }

  auto get_buffer(torch::Tensor template_tensor) -> torch::Tensor {
    return _m_allocate_buffer(template_tensor);
  }

private:
  shared_ref<ncclComm_t> m_comm;
  std::shared_ptr<void> m_buf;
  shared_ref<ncclWindow_t> m_win;
  std::shared_ptr<bool[]> m_used;
  const int m_rank;
  const int m_world_size;
  const c10::Device m_device;
  const bool m_fallback;
  const size_t m_buf_count;
  const size_t m_max_bytes;
};

auto get_nccl_unique_id() -> ncclIDWrapper {
  ncclUniqueId id;
  NCCL_CHECK(::ncclGetUniqueId(&id));
  return ncclID2Wrapper(id);
}

} // namespace

PYBIND11_MODULE(pynccl_wrapper, m) {
  py::class_<NCCLWrapper>(m, "NCCLWrapper")
      .def(py::init<int, int, size_t, ncclIDWrapper, c10::Device, bool,
                    size_t>())
      .def("all_reduce", &NCCLWrapper::all_reduce)
      .def("all_gather", &NCCLWrapper::all_gather)
      .def("get_buffer", &NCCLWrapper::get_buffer);
  m.def("get_nccl_unique_id", &get_nccl_unique_id,
        "Get the unique ID for NCCL initialization");
}
