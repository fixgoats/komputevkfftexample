#pragma once
#include <complex>
#include <cstddef>
#include <cstdint>

typedef std::complex<float> c32;
typedef std::complex<double> c64;
typedef float f32;
typedef double f64;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;

struct cvec2 {
  c32 x;
  c32 y;

  constexpr cvec2 operator*(const cvec2 b) const { return {x * b.x, y * b.y}; }
  template <class T> constexpr cvec2 operator*(const T b) const {
    return {b * x, b * y};
  }
  constexpr cvec2 operator+(const cvec2 b) const { return {x + b.x, y + b.y}; }
  template <class T>
  friend constexpr cvec2 operator*(const T a, const cvec2 b) {
    return {a * b.x, a * b.y};
  }
};
