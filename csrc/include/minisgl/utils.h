#pragma once

#include <concepts>
#include <source_location>
#include <sstream>
#include <utility>

namespace host {

template <typename... Args>
[[noreturn]]
inline auto panic(std::source_location location, Args &&...args) -> void {
  std::stringstream ss;
  ss << "Runtime check failed at " << location.file_name() << ":"
     << location.line() << " in function " << location.function_name();
  if constexpr (sizeof...(args) > 0) {
    (((ss << ": ") << std::forward<Args>(args)), ...);
  }
  throw std::runtime_error(std::move(ss).str());
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

} // namespace host

namespace math {

template <std::integral T, std::integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

} // namespace math
