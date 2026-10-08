#include "ps5helixsr/ps5_helixsr.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(VkResult result, const char *call) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(call) + ": " + std::to_string(result));
}

static std::vector<uint8_t> read_file(const char *path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  const auto size = stream.tellg();
  if (!stream || size <= 0)
    throw std::runtime_error("model read failed");
  std::vector<uint8_t> result(static_cast<size_t>(size));
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(result.data()), size))
    throw std::runtime_error("model read failed");
  return result;
}

static uint32_t memory_type(const VkPhysicalDeviceMemoryProperties &memory,
                            uint32_t bits) {
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags &
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      return i;
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
    if (bits & (1u << i))
      return i;
  throw std::runtime_error("image memory unavailable");
}

struct Image {
  VkImage image{};
  VkImageView view{};
  VkDeviceMemory memory{};
};

static Image make_image(VkDevice device,
                        const VkPhysicalDeviceMemoryProperties &memory,
                        uint32_t width, uint32_t height, VkFormat format) {
  Image result;
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  check(vkCreateImage(device, &info, nullptr, &result.image), "vkCreateImage");
  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(device, result.image, &requirements);
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = memory_type(memory, requirements.memoryTypeBits);
  check(vkAllocateMemory(device, &allocation, nullptr, &result.memory),
        "vkAllocateMemory");
  check(vkBindImageMemory(device, result.image, result.memory, 0),
        "vkBindImageMemory");
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = result.image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = format;
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  check(vkCreateImageView(device, &view, nullptr, &result.view),
        "vkCreateImageView");
  return result;
}

static void destroy_image(VkDevice device, Image &image) {
  if (image.view)
    vkDestroyImageView(device, image.view, nullptr);
  if (image.image)
    vkDestroyImage(device, image.image, nullptr);
  if (image.memory)
    vkFreeMemory(device, image.memory, nullptr);
}

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: helixsr_host_api canonical-weights.bin");
    auto weights = read_file(argv[1]);
    VkInstance instance{};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    check(vkCreateInstance(&instance_info, nullptr, &instance),
          "vkCreateInstance");
    uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
          "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> physical_devices(device_count);
    check(vkEnumeratePhysicalDevices(instance, &device_count,
                                     physical_devices.data()),
          "vkEnumeratePhysicalDevices");
    VkPhysicalDevice physical{};
    uint32_t family = 0;
    for (auto candidate : physical_devices) {
      uint32_t count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
      std::vector<VkQueueFamilyProperties> families(count);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count,
                                               families.data());
      for (uint32_t i = 0; i < count; ++i)
        if (families[i].queueCount &&
            (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          physical = candidate;
          family = i;
          break;
        }
      if (physical)
        break;
    }
    if (!physical)
      throw std::runtime_error("compute device unavailable");
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features features12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.pNext = &derivatives;
    VkPhysicalDeviceVulkan11Features features11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    features11.pNext = &features12;
    VkPhysicalDeviceFeatures2 features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &features11;
    vkGetPhysicalDeviceFeatures2(physical, &features);
    if (!features.features.shaderInt16 || !features.features.shaderInt64 ||
        !features12.shaderFloat16 || !features11.storageBuffer16BitAccess ||
        !derivatives.computeDerivativeGroupLinear)
      throw std::runtime_error("required shader features unavailable");
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR enabled_derivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    enabled_derivatives.computeDerivativeGroupLinear = VK_TRUE;
    VkPhysicalDeviceVulkan12Features enabled12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enabled12.shaderFloat16 = VK_TRUE;
    enabled12.shaderInt8 = features12.shaderInt8;
    enabled12.storageBuffer8BitAccess = features12.storageBuffer8BitAccess;
    enabled12.pNext = &enabled_derivatives;
    VkPhysicalDeviceVulkan11Features enabled11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    enabled11.storageBuffer16BitAccess = VK_TRUE;
    enabled11.pNext = &enabled12;
    VkPhysicalDeviceFeatures enabled{};
    enabled.shaderInt16 = enabled.shaderInt64 = VK_TRUE;
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    const char *extension = VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &enabled11;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &enabled;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = &extension;
    VkDevice device{};
    check(vkCreateDevice(physical, &device_info, nullptr, &device),
          "vkCreateDevice");
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = family;
    VkCommandPool command_pool{};
    check(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool),
          "vkCreateCommandPool");

    const uint32_t cases[][4] = {
        {128, 96, 128, 96}, {192, 144, 128, 96}, {256, 192, 128, 96}};
    uint64_t total_device_bytes = 0, total_host_bytes = 0;
    for (const auto &shape : cases) {
      Image color = make_image(device, memory, shape[2], shape[3],
                               VK_FORMAT_R16G16B16A16_SFLOAT);
      Image depth =
          make_image(device, memory, shape[2], shape[3], VK_FORMAT_R32_SFLOAT);
      Image motion = make_image(device, memory, shape[2], shape[3],
                                VK_FORMAT_R16G16_SFLOAT);
      Image output = make_image(device, memory, shape[0], shape[1],
                                VK_FORMAT_R16G16B16A16_SFLOAT);
      ps5helixsr_context_desc context_desc{};
      context_desc.struct_size = sizeof(context_desc);
      context_desc.physical_device = physical;
      context_desc.device = device;
      context_desc.output_width = shape[0];
      context_desc.output_height = shape[1];
      context_desc.render_width = shape[2];
      context_desc.render_height = shape[3];
      context_desc.flags = PS5HELIXSR_FLAG_AUTO_EXPOSURE;
      context_desc.canonical_weights = weights.data();
      context_desc.canonical_weights_bytes = weights.size();
      ps5helixsr_memory_requirements requirements{};
      if (ps5helixsr_get_memory_requirements(&context_desc, &requirements) !=
          PS5HELIXSR_OK)
        throw std::runtime_error("memory requirements failed");
      ps5helixsr_context *context{};
      if (ps5helixsr_context_create(&context_desc, &context) != PS5HELIXSR_OK)
        throw std::runtime_error("context creation failed");
      VkCommandBufferAllocateInfo allocation{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      allocation.commandPool = command_pool;
      allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocation.commandBufferCount = 1;
      VkCommandBuffer command{};
      check(vkAllocateCommandBuffers(device, &allocation, &command),
            "vkAllocateCommandBuffers");
      VkCommandBufferBeginInfo begin{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
      VkImage images[] = {color.image, depth.image, motion.image, output.image};
      VkImageMemoryBarrier barriers[4]{};
      for (uint32_t i = 0; i < 4; ++i) {
        barriers[i] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barriers[i].dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].srcQueueFamilyIndex = barriers[i].dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        barriers[i].image = images[i];
        barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      }
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 4, barriers);
      ps5helixsr_dispatch_desc dispatch{};
      dispatch.struct_size = sizeof(dispatch);
      dispatch.command_buffer = command;
      dispatch.color = color.view;
      dispatch.depth = depth.view;
      dispatch.motion_vectors = motion.view;
      dispatch.output = output.view;
      dispatch.color_layout = dispatch.depth_layout =
          dispatch.motion_vectors_layout = VK_IMAGE_LAYOUT_GENERAL;
      dispatch.jitter_x = 0.25f;
      dispatch.jitter_y = -0.375f;
      dispatch.pre_exposure = 1.25f;
      dispatch.reset = 1;
      if (ps5helixsr_dispatch(context, &dispatch) != PS5HELIXSR_OK)
        throw std::runtime_error("dispatch recording failed");
      if (ps5helixsr_dispatch(context, &dispatch) != PS5HELIXSR_ERROR_IN_FLIGHT)
        throw std::runtime_error("in-flight reuse was accepted");
      check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
      if (ps5helixsr_context_notify_completed(context) != PS5HELIXSR_OK)
        throw std::runtime_error("completion notification failed");
      check(vkResetCommandPool(device, command_pool, 0), "vkResetCommandPool");
      ps5helixsr_context_destroy(context);
      destroy_image(device, output);
      destroy_image(device, motion);
      destroy_image(device, depth);
      destroy_image(device, color);
      total_device_bytes += requirements.device_bytes;
      total_host_bytes += requirements.host_visible_bytes;
    }
    vkDestroyCommandPool(device, command_pool, nullptr);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::cout << "{\"result\":\"success\",\"device\":\""
              << properties.deviceName << "\",\"contexts\":3,"
              << "\"model_bytes\":" << weights.size()
              << ",\"total_device_bytes\":" << total_device_bytes
              << ",\"total_host_visible_bytes\":" << total_host_bytes
              << ",\"recorded\":true,\"submitted\":false,"
              << "\"in_flight_rejected\":true}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
