/* Copyright (C) 2026 BlackBearReloaded
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Writes the results of sampling, fetching, FP16 arithmetic and image stores done by
 * 512 invocations to a file. Run it once on a stock software driver and once on the
 * 32-lane one: the two files must be identical, or the wide driver cannot be trusted
 * as an execution host.   usage: host_driver_check SHADER.spv OUT.bin */
#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(VkResult result, const char *call) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(call) + " = " + std::to_string(int(result)));
}

static uint16_t to_half(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int exponent = int((bits >> 23) & 255) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0)
    return uint16_t(sign);
  return uint16_t(sign | (uint32_t(exponent) << 10) | (mantissa >> 13));
}

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: host_driver_check SHADER.spv OUT.bin");
    std::ifstream file(argv[1], std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(file)), {});
    VkInstance instance;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");
    uint32_t count = 1;
    VkPhysicalDevice physical;
    vkEnumeratePhysicalDevices(instance, &count, &physical);
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical, &properties);
    std::printf("device: %s subgroup=%u\n", properties.properties.deviceName, subgroup.subgroupSize);
    VkPhysicalDeviceShaderFloat16Int8Features float16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    float16.shaderFloat16 = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.features.shaderInt16 = VK_TRUE;
    features.pNext = &float16;
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device;
    check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice");
    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    const auto memory_type = [&](uint32_t bits, VkMemoryPropertyFlags flags) {
      for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags)
          return i;
      throw std::runtime_error("memory type");
    };
    const uint32_t width = 37, height = 23, invocations = 512;
    const VkDeviceSize texels = VkDeviceSize(width) * height * 8, results = VkDeviceSize(invocations) * 5 * 16;
    const VkDeviceSize target_bytes = 64 * 8 * 8;
    VkBuffer buffer;
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = texels + results + target_bytes;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory buffer_memory;
    check(vkAllocateMemory(device, &allocation, nullptr, &buffer_memory), "vkAllocateMemory");
    check(vkBindBufferMemory(device, buffer, buffer_memory, 0), "vkBindBufferMemory");
    uint8_t *mapped;
    check(vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&mapped)), "vkMapMemory");
    std::memset(mapped, 0, size_t(buffer_info.size));
    for (uint32_t y = 0; y < height; ++y)
      for (uint32_t x = 0; x < width; ++x) {
        const float value[4] = {0.1f + 0.021f * x + 0.5f * std::sin(0.9f * y), 0.3f + 0.037f * y,
                                float((x * 7 + y * 13) % 5) * 0.25f, 1.0f};
        for (int c = 0; c < 4; ++c) {
          const uint16_t half = to_half(value[c]);
          std::memcpy(mapped + (size_t(y) * width + x) * 8 + c * 2, &half, 2);
        }
      }
    const auto make_image = [&](uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImage &image, VkImageView &view) {
      VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      info.imageType = VK_IMAGE_TYPE_2D;
      info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      info.extent = {w, h, 1};
      info.mipLevels = info.arrayLayers = 1;
      info.samples = VK_SAMPLE_COUNT_1_BIT;
      info.usage = usage;
      check(vkCreateImage(device, &info, nullptr, &image), "vkCreateImage");
      VkMemoryRequirements needed;
      vkGetImageMemoryRequirements(device, image, &needed);
      VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocate.allocationSize = needed.size;
      allocate.memoryTypeIndex = memory_type(needed.memoryTypeBits, 0);
      VkDeviceMemory image_memory;
      check(vkAllocateMemory(device, &allocate, nullptr, &image_memory), "vkAllocateMemory");
      check(vkBindImageMemory(device, image, image_memory, 0), "vkBindImageMemory");
      VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view_info.image = image;
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = info.format;
      view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      check(vkCreateImageView(device, &view_info, nullptr, &view), "vkCreateImageView");
    };
    VkImage source, target;
    VkImageView source_view, target_view;
    make_image(width, height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, source, source_view);
    make_image(64, 8, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, target, target_view);
    VkSampler samplers[2];
    for (int i = 0; i < 2; ++i) {
      VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      info.magFilter = info.minFilter = i ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
      info.mipmapMode = i ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
      info.addressModeU = info.addressModeV = info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      check(vkCreateSampler(device, &info, nullptr, &samplers[i]), "vkCreateSampler");
    }
    const VkDescriptorType types[5] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_SAMPLER,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    VkDescriptorSetLayoutBinding bindings[5];
    for (uint32_t i = 0; i < 5; ++i)
      bindings[i] = {i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 5;
    layout_info.pBindings = bindings;
    VkDescriptorSetLayout set_layout;
    check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout), "vkCreateDescriptorSetLayout");
    VkDescriptorPoolSize sizes[4] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_SAMPLER, 2},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 4;
    pool_info.pPoolSizes = sizes;
    VkDescriptorPool pool;
    check(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &set_layout;
    VkDescriptorSet set;
    check(vkAllocateDescriptorSets(device, &set_info, &set), "vkAllocateDescriptorSets");
    VkDescriptorImageInfo images[4] = {{VK_NULL_HANDLE, source_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                       {samplers[0], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},
                                       {samplers[1], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},
                                       {VK_NULL_HANDLE, target_view, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo result_info{buffer, texels, results};
    VkWriteDescriptorSet writes[5];
    for (uint32_t i = 0; i < 5; ++i) {
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = types[i];
      if (i < 4)
        writes[i].pImageInfo = &images[i];
      else
        writes[i].pBufferInfo = &result_info;
    }
    vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    VkPipelineLayout pipeline_layout;
    check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size();
    module_info.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule module;
    check(vkCreateShaderModule(device, &module_info, nullptr, &module), "vkCreateShaderModule");
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.layout = pipeline_layout;
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    VkPipeline pipeline;
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "vkCreateComputePipelines");
    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandPool command_pool;
    check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command;
    check(vkAllocateCommandBuffers(device, &command_info, &command), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
    const auto barrier = [&](VkImage image, VkImageLayout before, VkImageLayout after) {
      VkImageMemoryBarrier item{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      item.srcAccessMask = item.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      item.oldLayout = before;
      item.newLayout = after;
      item.srcQueueFamilyIndex = item.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      item.image = image;
      item.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                           nullptr, 0, nullptr, 1, &item);
    };
    barrier(source, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(command, buffer, source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier(source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    barrier(target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(command, invocations / 128, 1, 1);
    barrier(target, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    region.bufferOffset = texels + results;
    region.imageExtent = {64, 8, 1};
    vkCmdCopyImageToBuffer(command, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char *>(mapped + texels), std::streamsize(results + target_bytes));
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
