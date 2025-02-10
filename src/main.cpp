#include "kompute/Kompute.hpp"
#include "vkFFT.h"
#include "vkhelpers.h"
#include <bit>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using std::bit_cast;

template <typename T>
constexpr u32 euclid_mod(T a, u32 b) {
  assert(b != 0);
  return a % b;
}

constexpr u32 fftshiftidx(u32 i, u32 n) {
  return euclid_mod(i + (n + 1) / 2, n);
}

template <class T>
constexpr auto numfmt(T x) {
  if constexpr (std::is_same_v<T, c32> or std::is_same_v<T, c64>) {
    return std::format("({}+{}j)", x.real(), x.imag());
  } else {
    return std::format("{}", x);
  }
}

template <typename T>
void leftRotate(std::vector<T>& arr, u32 d) {
  auto n = arr.size();
  d = d % n; // To handle case when d >= n

  // Reverse the first d elements
  std::reverse(arr.begin(), arr.begin() + d);

  // Reverse the remaining elements
  std::reverse(arr.begin() + d, arr.end());

  // Reverse the whole array
  std::reverse(arr.begin(), arr.end());
}

template <typename T>
void fftshift(std::vector<T>& arr) {
  auto n = arr.size();
  u32 d = (n + 1) % 2;
  leftRotate(arr, d);
}

static std::vector<u32> compileSource(const std::string& source) {
  std::ofstream fileOut("tmp_kp_shader.comp");
  fileOut << source;
  fileOut.close();
  if (system(std::string("glslangValidator -V tmp_kp_shader.comp -o "
                         "tmp_kp_shader.comp.spv")
                 .c_str()))
    throw std::runtime_error("Error running glslangValidator command");
  std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
  std::vector<u8> buffer;
  buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
  return {(u32*)buffer.data(), (u32*)(buffer.data() + buffer.size())};
}

class FFT : public kp::OpBase {
public:
  FFT(VkFFTApplication* app, i64 direction, VkFFTLaunchParams* lParams)
      : app{app}, lParams{lParams}, direction{direction} {}

  void record(const vk::CommandBuffer& commandBuffer) override {
    lParams->commandBuffer = bit_cast<VkCommandBuffer*>(&commandBuffer);
    VkFFTAppend(app, direction, lParams);
  }
  virtual void preEval(const vk::CommandBuffer& commandBuffer) override {};
  virtual void postEval(const vk::CommandBuffer& commandBuffer) override {};
  virtual ~FFT() override {};
  VkFFTApplication* app;
  VkFFTLaunchParams* lParams;
  i64 direction;
};

constexpr f32 hbar = 6.582e-1;

int main() {
  VulkanApp bleh{};

  const f32 E = 1.3;
  const f32 tstart = 0;
  const f32 tend = 10;
  const u32 samples = 1024;
  std::vector<f32> buff(samples * 2);
  const f32 dt = (tend - tstart) / (f32)samples;
  for (u32 i = 0; i < samples; i++) {
    f32 t = tstart + i * dt;
    buff[2 * i] = std::cos(-E * t / hbar);
    buff[2 * i + 1] = std::sin(-E * t / hbar);
    /*buff[2 * i] = std::cos(-2 * x) * std::cos(M_PI * x) -
                  std::sin(-2 * x) * std::sin(M_PI * x);
    buff[2 * i + 1] = std::sin(-2 * x) * std::cos(M_PI * x) +
                      std::cos(-2 * x) * std::sin(M_PI * x);*/
  }
  auto tensor = std::make_shared<kp::TensorT<f32>>(
      std::shared_ptr<vk::PhysicalDevice>(&bleh.pDevice,
                                          [](vk::PhysicalDevice*) {}),
      std::shared_ptr<vk::Device>(&bleh.device, [](vk::Device*) {}), buff);
  auto seq = std::make_shared<kp::Sequence>(
      std::shared_ptr<vk::PhysicalDevice>(&bleh.pDevice,
                                          [](vk::PhysicalDevice*) {}),
      std::shared_ptr<vk::Device>(&bleh.device, [](vk::Device*) {}),
      std::shared_ptr<vk::Queue>(&bleh.queue, [](vk::Queue*) {}),
      bleh.getComputeQueueFamilyIndex());
  u64 bufferSize = samples * 8;
  VkFFTConfiguration conf{};
  conf.device = bit_cast<VkDevice*>(&bleh.device);
  conf.queue = bit_cast<VkQueue*>(&bleh.queue);
  conf.FFTdim = 1;
  conf.size[0] = samples;
  conf.fence = bit_cast<VkFence*>(&bleh.fence);
  conf.commandPool = bit_cast<VkCommandPool*>(&bleh.commandPool);
  conf.physicalDevice = bit_cast<VkPhysicalDevice*>(&bleh.pDevice);
  conf.buffer = bit_cast<VkBuffer*>(tensor->getPrimaryBuffer().get());
  conf.bufferSize = &bufferSize;
  VkFFTApplication app{};
  initializeVkFFT(&app, conf);
  VkFFTLaunchParams lp{};
  std::shared_ptr<FFT> forward{new FFT(&app, -1, &lp)};
  std::shared_ptr<FFT> backward{new FFT(&app, 1, &lp)};

  seq->record<kp::OpSyncDevice>({tensor})
      ->record(forward)
      ->record<kp::OpSyncLocal>({tensor})
      ->eval();
  deleteVkFFT(&app);

  std::ofstream values("testsign.csv");
  buff = tensor->vector();
  std::vector<c32> arg(buff.begin(), buff.end());
  // fftshift(arg);
  for (u32 i = 0; i < samples; i++) {
    values << numfmt(arg[i]) << ' ';
  }
  values.close();
}
