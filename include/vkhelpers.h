#pragma once
#include "hack.h"
#include "typedefs.h"

struct VulkanApp {
  vk::Instance instance;
  vk::PhysicalDevice pDevice;
  vk::Device device;
  vk::Queue queue;
  vk::Fence fence;
  vk::CommandPool commandPool;

  VulkanApp();
  u32 getComputeQueueFamilyIndex();
  ~VulkanApp();
};

vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance,
                                      const int32_t desiredGPU = -1);
