#pragma once
#include "hack.h"
#include "typedefs.h"
#include <format>
#include <fstream>
#include <type_traits>

template <typename T> std::vector<T> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<T> buffer(fileSize / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
  file.close();
  return buffer;
}

struct VulkanApp {
  vk::Instance instance;
  vk::PhysicalDevice pDevice;
  vk::Device device;
  vk::Queue queue;
  vk::Fence fence;
  vk::CommandPool commandPool;

  VulkanApp();
  void copyBuffers(vk::Buffer &srcBuffer, vk::Buffer &dstBuffer,
                   uint32_t bufferSize);
  void copyInBatches(vk::Buffer &srcBuffer, vk::Buffer &dstBuffer,
                     uint32_t batchSize, uint32_t numBatches);

  uint32_t getComputeQueueFamilyIndex();
  ~VulkanApp();
};

template <class T> constexpr auto numfmt(T x) {
  if constexpr (std::is_same_v<T, c32> or std::is_same_v<T, c64>) {
    return std::format("({}+{}j)", x.real(), x.imag());
  } else {
    return std::format("{}", x);
  }
}

std::string tstamp();

void saveToFile(std::string fname, const char *buf, size_t size);

const vk::MemoryBarrier fullMemoryBarrier(vk::AccessFlagBits::eShaderRead |
                                              vk::AccessFlagBits::eMemoryWrite,
                                          vk::AccessFlagBits::eMemoryRead |
                                              vk::AccessFlagBits::eMemoryWrite);

std::vector<uint32_t> readFile(const std::string &filename);
vk::PhysicalDevice pickPhysicalDevice(const vk::Instance &instance,
                                      const int32_t desiredGPU = -1);
