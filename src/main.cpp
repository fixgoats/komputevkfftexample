#include "kompute/Kompute.hpp"
#include "vkFFT.h"
#include "vkhelpers.h"
#include <bit>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using std::bit_cast;

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

int main() {
  auto bleh = VulkanApp();
  // Empty deleters so VulkanApp definitely handles clean up (maybe make
  // VulkanApp a shared object)

  std::vector<f32> buff(128 * 2);
  for (u32 i = 0; i < 128; i++) {
    f32 x = 2 * M_PI * (f32)i / 128.;
    buff[2 * i] = std::sin(x);
    buff[2 * i + 1] = 0.;
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
  u64 bufferSize = 128 * 8;
  VkFFTConfiguration conf{};
  conf.device = bit_cast<VkDevice*>(&bleh.device);
  conf.queue = bit_cast<VkQueue*>(&bleh.queue);
  conf.FFTdim = 1;
  conf.size[0] = 128;
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
      ->record(backward)
      ->record<kp::OpSyncLocal>({tensor})
      ->eval();
  deleteVkFFT(&app);

  for (const auto& e : tensor->vector()) {
    std::cout << e << ' ';
  }
  std::cout << '\n';
}
