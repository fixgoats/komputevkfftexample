#include "vkhelpers.h"
#include <iostream>

vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance,
                                      const int32_t desiredGPU) {
  // check if there are GPUs that support Vulkan and "intelligently" select
  // one. Prioritises discrete GPUs, and after that VRAM size.
  std::vector<vk::PhysicalDevice> pDevices =
      instance.enumeratePhysicalDevices();
  uint32_t nDevices = pDevices.size();

  // shortcut if there's only one device available.
  if (nDevices == 1) {
    return pDevices[0];
  }
  // Try to select desired GPU if specified.
  if (desiredGPU > -1) {
    if (desiredGPU < static_cast<int32_t>(nDevices)) {
      return pDevices[desiredGPU];
    } else {
      std::cout << "Device not available\n";
    }
  }

  std::vector<uint32_t> discrete; // the indices of the available discrete gpus
  std::vector<uint64_t> vram(nDevices);
  for (uint32_t i = 0; i < nDevices; i++) {
    if (pDevices[i].getProperties().deviceType ==
        vk::PhysicalDeviceType::eDiscreteGpu) {
      discrete.push_back(i);
    }

    auto heaps = pDevices[i].getMemoryProperties().memoryHeaps;
    for (const auto& heap : heaps) {
      if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
        vram[i] = heap.size;
      }
    }
  }

  // only consider discrete gpus if available:
  if (discrete.size() > 0) {
    if (discrete.size() == 1) {
      return pDevices[discrete[0]];
    } else {
      uint32_t max = 0;
      uint32_t selectedGPU = 0;
      for (const auto& index : discrete) {
        if (vram[index] > max) {
          max = vram[index];
          selectedGPU = index;
        }
      }
      return pDevices[selectedGPU];
    }
  } else {
    uint32_t max = 0;
    uint32_t selectedGPU = 0;
    for (uint32_t i = 0; i < nDevices; i++) {
      if (vram[i] > max) {
        max = vram[i];
        selectedGPU = i;
      }
    }
    return pDevices[selectedGPU];
  }
}

static const std::string appName{"Vulkan App"};

VulkanApp::VulkanApp() {
  vk::ApplicationInfo appInfo{appName.c_str(), 1, nullptr, 0,
                              VK_API_VERSION_1_3};
  const std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
  vk::InstanceCreateInfo iCI(vk::InstanceCreateFlags(), &appInfo, layers, {});
  instance = vk::createInstance(iCI);
  pDevice = pickPhysicalDevice(instance);
  uint32_t computeQueueFamilyIndex = getComputeQueueFamilyIndex();
  float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo dQCI(vk::DeviceQueueCreateFlags(),
                                 computeQueueFamilyIndex, 1, &queuePriority);
  vk::DeviceCreateInfo dCI(vk::DeviceCreateFlags(), dQCI);
  device = pDevice.createDevice(dCI);
  vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(),
                                                  computeQueueFamilyIndex);
  commandPool = device.createCommandPool(commandPoolCreateInfo);
  queue = device.getQueue(computeQueueFamilyIndex, 0);
  fence = device.createFence(vk::FenceCreateInfo());
}

uint32_t VulkanApp::getComputeQueueFamilyIndex() {
  auto queueFamilyProps = pDevice.getQueueFamilyProperties();
  auto propIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(),
                   [](const vk::QueueFamilyProperties& prop) {
                     return prop.queueFlags & vk::QueueFlagBits::eCompute;
                   });
  return std::distance(queueFamilyProps.begin(), propIt);
}

VulkanApp::~VulkanApp() {
  device.waitIdle();
  device.destroyFence(fence);
  device.destroyCommandPool(commandPool);
  device.destroy();
  instance.destroy();
}
