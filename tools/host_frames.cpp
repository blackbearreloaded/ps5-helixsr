/* Copyright (C) 2026 BlackBearReloaded
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Runs the runtime over a frame-input directory on a host Vulkan device and
 * writes every output frame; the same directory layout as the reference probe:
 *   frames.bin, in/NN/{color.rgba16f,depth.r32f,motion.rg16f} -> out/NN.rgba16f
 * With --dump-frame N every resource of the context is written after each step
 * of that frame to dump/SS/NAME.bin (steps listed in dump/steps.txt).
 * The device must run 32-lane subgroups (see docs/VALIDATION.md).
 */
#include <vulkan/vulkan.h>

#include <ps5_helixsr_debug.h>
#include <ps5helixsr/ps5_helixsr.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void check(VkResult result, const char *call) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(call) + " = " + std::to_string(int(result)));
}

std::vector<uint8_t> read_file(const fs::path &path, size_t bytes = 0) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot open " + path.string());
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), {});
  if (bytes && data.size() != bytes)
    throw std::runtime_error("unexpected size of " + path.string());
  return data;
}

void write_file(const fs::path &path, const void *data, size_t bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char *>(data), std::streamsize(bytes));
  if (!file)
    throw std::runtime_error("cannot write " + path.string());
}

struct Image {
  VkImage image{};
  VkDeviceMemory memory{};
  VkImageView view{};
  uint32_t width{}, height{};
};

struct App {
  VkInstance instance{};
  VkPhysicalDevice physical{};
  VkDevice device{};
  VkQueue queue{};
  VkPhysicalDeviceMemoryProperties memory{};
  VkCommandPool pool{};
  VkCommandBuffer command{};
  VkFence fence{};
  Image color, depth, motion, output;
  VkBuffer staging{};
  VkDeviceMemory staging_memory{};
  uint8_t *mapped{};
  VkDeviceSize staging_bytes{};
};

uint32_t memory_type(const App &app, uint32_t bits, VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < app.memory.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (app.memory.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  throw std::runtime_error("memory type unavailable");
}

Image make_image(App &app, VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usage) {
  Image result{};
  result.width = width;
  result.height = height;
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.usage = usage;
  check(vkCreateImage(app.device, &info, nullptr, &result.image), "vkCreateImage");
  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(app.device, result.image, &requirements);
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = memory_type(app, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  check(vkAllocateMemory(app.device, &allocation, nullptr, &result.memory), "vkAllocateMemory");
  check(vkBindImageMemory(app.device, result.image, result.memory, 0), "vkBindImageMemory");
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = result.image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = format;
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  check(vkCreateImageView(app.device, &view, nullptr, &result.view), "vkCreateImageView");
  return result;
}

void image_barrier(VkCommandBuffer command, VkImage image, VkImageLayout before, VkImageLayout after) {
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcAccessMask = barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.oldLayout = before;
  barrier.newLayout = after;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);
}

void everything_barrier(VkCommandBuffer command) {
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
      VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                       &barrier, 0, nullptr, 0, nullptr);
}

void initialize(App &app) {
  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance.pApplicationInfo = &application;
  check(vkCreateInstance(&instance, nullptr, &app.instance), "vkCreateInstance");
  uint32_t count = 1;
  VkResult found = vkEnumeratePhysicalDevices(app.instance, &count, &app.physical);
  if ((found != VK_SUCCESS && found != VK_INCOMPLETE) || !count)
    throw std::runtime_error("no Vulkan device");
  VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &subgroup;
  vkGetPhysicalDeviceProperties2(app.physical, &properties);
  std::printf("device: %s subgroup=%u\n", properties.properties.deviceName, subgroup.subgroupSize);
  if (subgroup.subgroupSize != 32)
    throw std::runtime_error("the device does not run 32-lane subgroups");

  VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivative{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
  derivative.computeDerivativeGroupLinear = VK_TRUE;
  VkPhysicalDevice16BitStorageFeatures storage16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  storage16.storageBuffer16BitAccess = VK_TRUE;
  storage16.pNext = &derivative;
  VkPhysicalDeviceShaderFloat16Int8Features float16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  float16.shaderFloat16 = VK_TRUE;
  float16.pNext = &storage16;
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.features.shaderInt16 = VK_TRUE;
  features.features.shaderInt64 = VK_TRUE;
  features.features.shaderStorageImageExtendedFormats = VK_TRUE;
  features.pNext = &float16;
  const char *extensions[] = {VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME};
  float priority = 1;
  VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device.pNext = &features;
  device.queueCreateInfoCount = 1;
  device.pQueueCreateInfos = &queue;
  device.enabledExtensionCount = 1;
  device.ppEnabledExtensionNames = extensions;
  check(vkCreateDevice(app.physical, &device, nullptr, &app.device), "vkCreateDevice");
  vkGetDeviceQueue(app.device, 0, 0, &app.queue);
  vkGetPhysicalDeviceMemoryProperties(app.physical, &app.memory);
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  check(vkCreateCommandPool(app.device, &pool, nullptr, &app.pool), "vkCreateCommandPool");
  VkCommandBufferAllocateInfo command{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command.commandPool = app.pool;
  command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command.commandBufferCount = 1;
  check(vkAllocateCommandBuffers(app.device, &command, &app.command), "vkAllocateCommandBuffers");
  VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  check(vkCreateFence(app.device, &fence, nullptr, &app.fence), "vkCreateFence");
}

void make_staging(App &app, VkDeviceSize bytes) {
  app.staging_bytes = bytes;
  VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer.size = bytes;
  buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  check(vkCreateBuffer(app.device, &buffer, nullptr, &app.staging), "vkCreateBuffer");
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(app.device, app.staging, &requirements);
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = memory_type(app, requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  check(vkAllocateMemory(app.device, &allocation, nullptr, &app.staging_memory), "vkAllocateMemory");
  check(vkBindBufferMemory(app.device, app.staging, app.staging_memory, 0), "vkBindBufferMemory");
  check(vkMapMemory(app.device, app.staging_memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&app.mapped)),
        "vkMapMemory");
}

void submit(App &app) {
  check(vkEndCommandBuffer(app.command), "vkEndCommandBuffer");
  VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  info.commandBufferCount = 1;
  info.pCommandBuffers = &app.command;
  check(vkQueueSubmit(app.queue, 1, &info, app.fence), "vkQueueSubmit");
  check(vkWaitForFences(app.device, 1, &app.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  check(vkResetFences(app.device, 1, &app.fence), "vkResetFences");
}

struct FrameInfo {
  float jitter_x, jitter_y;
  uint32_t reset;
  float pre_exposure;
};

void begin(App &app) {
  check(vkResetCommandBuffer(app.command, 0), "vkResetCommandBuffer");
  VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(app.command, &info), "vkBeginCommandBuffer");
}

int run(int argc, char **argv) {
  if (argc < 9) {
    std::fprintf(stderr, "usage: %s FRAMES_DIR WEIGHTS RENDER_W RENDER_H OUTPUT_W OUTPUT_H FRAMES OUT_DIR "
                         "[--auto-exposure] [--dump-frame N]\n", argv[0]);
    return 2;
  }
  const fs::path frames_dir = argv[1], out_dir = argv[8];
  const uint32_t rw = std::atoi(argv[3]), rh = std::atoi(argv[4]), ow = std::atoi(argv[5]), oh = std::atoi(argv[6]);
  const uint32_t frames = std::atoi(argv[7]);
  bool auto_exposure = false;
  int dump_frame = -1;
  for (int i = 9; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--auto-exposure"))
      auto_exposure = true;
    else if (!std::strcmp(argv[i], "--dump-frame") && i + 1 < argc)
      dump_frame = std::atoi(argv[++i]);
  }
  const auto weights = read_file(argv[2]);
  const auto info = read_file(frames_dir / "frames.bin");
  if (info.size() < size_t(frames) * sizeof(FrameInfo))
    throw std::runtime_error("frames.bin holds fewer frames than requested");

  App app;
  initialize(app);
  const VkImageUsageFlags input = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  app.color = make_image(app, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh, input);
  app.depth = make_image(app, VK_FORMAT_R32_SFLOAT, rw, rh, input);
  app.motion = make_image(app, VK_FORMAT_R16G16_SFLOAT, rw, rh, input);
  app.output = make_image(app, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh,
                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  ps5helixsr_context_desc description{};
  description.struct_size = sizeof(description);
  description.physical_device = app.physical;
  description.device = app.device;
  description.output_width = ow;
  description.output_height = oh;
  description.render_width = rw;
  description.render_height = rh;
  description.flags = auto_exposure ? PS5HELIXSR_FLAG_AUTO_EXPOSURE : 0u;
  description.canonical_weights = weights.data();
  description.canonical_weights_bytes = weights.size();
  ps5helixsr_context *context = nullptr;
  auto start = std::chrono::steady_clock::now();
  const ps5helixsr_result created = ps5helixsr_context_create(&description, &context);
  std::printf("context: result=%d ms=%.0f\n", created,
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  if (created != PS5HELIXSR_OK)
    return 1;

  std::vector<ps5helixsr_debug_resource> resources(64);
  resources.resize(ps5helixsr_debug_resources(context, resources.data(), 64));
  VkDeviceSize dump_bytes = 0;
  std::vector<VkDeviceSize> offsets;
  for (const auto &resource : resources) {
    offsets.push_back(dump_bytes);
    dump_bytes += (resource.bytes + 15) & ~VkDeviceSize(15);
  }
  const VkDeviceSize color_bytes = VkDeviceSize(rw) * rh * 8, small_bytes = VkDeviceSize(rw) * rh * 4;
  const VkDeviceSize output_bytes = VkDeviceSize(ow) * oh * 8;
  const VkDeviceSize depth_at = color_bytes, motion_at = depth_at + small_bytes, output_at = motion_at + small_bytes;
  const VkDeviceSize dump_at = output_at + output_bytes;
  make_staging(app, dump_at + (dump_frame >= 0 ? dump_bytes : 0));

  for (uint32_t frame = 0; frame < frames; ++frame) {
    FrameInfo item;
    std::memcpy(&item, info.data() + frame * sizeof(item), sizeof(item));
    char name[32];
    std::snprintf(name, sizeof(name), "%02u", frame);
    const fs::path in = frames_dir / "in" / name;
    std::memcpy(app.mapped, read_file(in / "color.rgba16f", color_bytes).data(), color_bytes);
    std::memcpy(app.mapped + depth_at, read_file(in / "depth.r32f", small_bytes).data(), small_bytes);
    std::memcpy(app.mapped + motion_at, read_file(in / "motion.rg16f", small_bytes).data(), small_bytes);

    ps5helixsr_dispatch_desc dispatch{};
    dispatch.struct_size = sizeof(dispatch);
    dispatch.command_buffer = app.command;
    dispatch.color = app.color.view;
    dispatch.depth = app.depth.view;
    dispatch.motion_vectors = app.motion.view;
    dispatch.output = app.output.view;
    dispatch.color_layout = dispatch.depth_layout = dispatch.motion_vectors_layout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dispatch.jitter_x = item.jitter_x;
    dispatch.jitter_y = item.jitter_y;
    dispatch.pre_exposure = item.pre_exposure;
    dispatch.reset = item.reset;

    std::vector<ps5helixsr_debug_step> steps(128);
    steps.resize(ps5helixsr_debug_steps(context, item.reset, steps.data(), 128));
    const bool dump = int(frame) == dump_frame;
    const uint32_t passes = dump ? uint32_t(steps.size()) : 1;
    std::ofstream step_list;
    if (dump) {
      fs::create_directories(out_dir / "dump");
      step_list.open(out_dir / "dump" / "steps.txt");
    }
    start = std::chrono::steady_clock::now();
    for (uint32_t pass = 0; pass < passes; ++pass) {
      begin(app);
      if (pass == 0) {
        const VkImageLayout before = frame ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        const Image *inputs[] = {&app.color, &app.depth, &app.motion};
        const VkDeviceSize at[] = {0, depth_at, motion_at};
        for (int i = 0; i < 3; ++i) {
          image_barrier(app.command, inputs[i]->image, before, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
          VkBufferImageCopy region{};
          region.bufferOffset = at[i];
          region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
          region.imageExtent = {rw, rh, 1};
          vkCmdCopyBufferToImage(app.command, app.staging, inputs[i]->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
          image_barrier(app.command, inputs[i]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        if (!frame)
          image_barrier(app.command, app.output.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
      }
      ps5helixsr_debug_set_range(context, dump ? pass : 0, dump ? 1 : UINT32_MAX);
      const ps5helixsr_result result = ps5helixsr_dispatch(context, &dispatch);
      if (result != PS5HELIXSR_OK)
        throw std::runtime_error("ps5helixsr_dispatch = " + std::to_string(result));
      everything_barrier(app.command);
      if (pass + 1 == passes) {
        VkBufferImageCopy region{};
        region.bufferOffset = output_at;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {ow, oh, 1};
        vkCmdCopyImageToBuffer(app.command, app.output.image, VK_IMAGE_LAYOUT_GENERAL, app.staging, 1, &region);
      }
      if (dump)
        for (size_t i = 0; i < resources.size(); ++i) {
          if (resources[i].buffer) {
            VkBufferCopy region{0, dump_at + offsets[i], resources[i].bytes};
            vkCmdCopyBuffer(app.command, resources[i].buffer, app.staging, 1, &region);
          } else {
            VkBufferImageCopy region{};
            region.bufferOffset = dump_at + offsets[i];
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {resources[i].width, resources[i].height, 1};
            vkCmdCopyImageToBuffer(app.command, resources[i].image, VK_IMAGE_LAYOUT_GENERAL, app.staging, 1,
                                   &region);
          }
        }
      everything_barrier(app.command);
      submit(app);
      if (dump) {
        char step_name[16];
        std::snprintf(step_name, sizeof(step_name), "%02u", pass);
        step_list << pass << ' ' << steps[pass].launch << ' ' << steps[pass].name << ' ' << steps[pass].grid[0]
                  << ' ' << steps[pass].grid[1] << ' ' << steps[pass].grid[2] << ' ' << steps[pass].uniform_offset
                  << ' ' << steps[pass].uniform_bytes << '\n';
        for (size_t i = 0; i < resources.size(); ++i)
          write_file(out_dir / "dump" / step_name / (std::string(resources[i].name) + ".bin"),
                     app.mapped + dump_at + offsets[i], resources[i].bytes);
      }
    }
    check(ps5helixsr_context_notify_completed(context) == PS5HELIXSR_OK ? VK_SUCCESS : VK_ERROR_UNKNOWN, "notify");
    write_file(out_dir / "out" / (std::string(name) + ".rgba16f"), app.mapped + output_at, output_bytes);
    std::printf("frame %u: steps=%zu ms=%.0f\n", frame, steps.size(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    std::fflush(stdout);
  }
  ps5helixsr_context_destroy(context);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
