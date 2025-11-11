#pragma once

#include <concepts>
#include <ostream>
#include <source_location>
#include <sstream>
#include <utility>

namespace host {

struct PanicError : public std::runtime_error {
  PanicError(std::string message)
      : runtime_error(message), message(std::move(message)) {}
  auto detail() const -> std::string_view {
    const auto sv = std::string_view{message};
    const auto pos = sv.find(": ");
    return pos == std::string_view::npos ? sv : sv.substr(pos + 2);
  }
  std::string message;
};

template <typename... Args>
[[noreturn]]
inline auto panic(std::source_location location, Args &&...args) -> void {
  std::ostringstream os;
  os << "Runtime check failed at " << location.file_name() << ":"
     << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  } else {
    os << " in " << location.function_name();
  }
  throw PanicError(std::move(os).str());
}

template <typename... Args> struct RuntimeCheck {
  template <std::convertible_to<bool> T>
  explicit RuntimeCheck(
      T &&condition, Args &&...args,
      std::source_location location = std::source_location::current()) {
    if (!condition) {
      [[unlikely]];
      ::host::panic(location, std::forward<Args>(args)...);
    }
  }
};

template <typename T, typename... Args>
explicit RuntimeCheck(T &&, Args &&...) -> RuntimeCheck<Args...>;

namespace pointer {

template <std::same_as<void> T, std::integral... U>
auto offset(T *ptr, U... offset) -> void * {
  return static_cast<char *>(ptr) + (... + offset);
}

template <std::same_as<void> T, std::integral... U>
auto offset(const T *ptr, U... offset) -> const void * {
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

} // namespace host

namespace math {

template <std::integral T, std::integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

} // namespace math
