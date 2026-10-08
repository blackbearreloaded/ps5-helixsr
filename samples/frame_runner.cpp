/* Copyright (C) 2026 BlackBearReloaded
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * PS5 (and host) frame runner: the console half of the validation.
 * Reads a frame-input directory (tools/reference_inputs.py layout), runs the
 * runtime over it through the public API, saves every output frame and times
 * the upscaler on its own. The same inputs go through the original HelixSR
 * (tools/reference_probe.py) and through the host runner (tools/host_frames.cpp).
 *
 * assets/cases.txt names the cases, one per line; each is a folder assets/NAME with
 * run.txt:  render W H / output W H / frames N / auto 0|1 / save 0|1 / timing K / profile K / stages 0|1
 * One launch runs them all (console launches are the scarce resource).
 * "stages 1" records the first frame one step at a time and saves every buffer of the network
 * a step has changed (NAME-stage-SS-BUFFER.bin), for a stage by stage comparison with another run.
 * Markers: HELIXSR_CASE, HELIXSR_FRAME ... crc32=, HELIXSR_STAGE, HELIXSR_TIMING ..., HELIXSR_END result=
 */
#ifdef HELIXSR_HOST
#include <vulkan/vulkan.h>
#else
#include <ps5vk/ps5vk.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <ps5helixsr/ps5_helixsr.h>
#include <ps5_helixsr_debug.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

namespace {

FILE *log_file;
#ifdef HELIXSR_HOST
const char *asset_root() {
  const char *value = std::getenv("HELIXSR_ASSET_DIR");
  return value ? value : "assets";
}
const char *output_root() {
  const char *value = std::getenv("HELIXSR_OUTPUT_DIR");
  return value ? value : ".";
}
int debug_text(int, const char *text) { return std::fputs(text, stdout); }
#else
extern "C" int sceKernelDebugOutText(int, const char *);
extern "C" int sceSystemServiceLoadExec(const char *, char *const[]);
extern "C" int helixsr_native_heap_init(void);
const char *asset_root() { return "/app0/assets"; }
int debug_text(int level, const char *text) { return sceKernelDebugOutText(level, text); }
const char *output_root() {
  static const char *chosen;
  /* The title folder outlives the title, so results can be fetched after it has ended. */
  static const char *const candidates[] = {"/app0/results", "/data/helixsr-results", "/download0"};
  for (const char *path : candidates) {
    if (chosen)
      break;
    mkdir(path, 0777);
    char probe[256];
    std::snprintf(probe, sizeof(probe), "%s/.writable", path);
    FILE *file = std::fopen(probe, "wb");
    if (file && std::fputc('1', file) != EOF && !std::fclose(file))
      chosen = path;
    else if (file)
      std::fclose(file);
  }
  return chosen ? chosen : "/download0";
}
#endif

void report(const char *format, ...) {
  char message[2048];
  va_list args;
  va_start(args, format);
  std::vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  debug_text(0, message);
  if (log_file) {
    std::fputs(message, log_file);
    std::fflush(log_file);
  }
}

void check(VkResult result, const char *call) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(call) + " = " + std::to_string(int(result)));
}

double now_ms() {
  timespec value{};
  clock_gettime(CLOCK_MONOTONIC, &value);
  return value.tv_sec * 1000.0 + value.tv_nsec / 1000000.0;
}

std::vector<uint8_t> load(const std::string &name, size_t bytes = 0) {
  const std::string path = std::string(asset_root()) + "/" + name;
  FILE *file = std::fopen(path.c_str(), "rb");
  if (!file)
    throw std::runtime_error("cannot open asset " + path);
  std::fseek(file, 0, SEEK_END);
  const long size = std::ftell(file);
  std::fseek(file, 0, SEEK_SET);
  std::vector<uint8_t> result(size > 0 ? size_t(size) : 0);
  const size_t got = result.empty() ? 0 : std::fread(result.data(), 1, result.size(), file);
  std::fclose(file);
  if (got != result.size() || (bytes && got != bytes))
    throw std::runtime_error("asset size mismatch " + path);
  return result;
}

void save(const std::string &name, const void *data, size_t bytes) {
  const std::string path = std::string(output_root()) + "/" + name;
  FILE *file = std::fopen(path.c_str(), "wb");
  if (!file)
    throw std::runtime_error("cannot save " + path);
  const bool failed = std::fwrite(data, 1, bytes, file) != bytes;
  if (std::fclose(file) != 0 || failed)
    throw std::runtime_error("cannot save " + path);
}

uint32_t crc32(const uint8_t *data, size_t bytes) {
  static uint32_t table[256];
  if (!table[1])
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t value = i;
      for (uint32_t bit = 0; bit < 8; ++bit)
        value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1u)));
      table[i] = value;
    }
  uint32_t value = ~0u;
  for (size_t i = 0; i < bytes; ++i)
    value = table[(value ^ data[i]) & 255u] ^ (value >> 8);
  return ~value;
}

struct Config {
  uint32_t render_width{}, render_height{}, output_width{}, output_height{};
  uint32_t frames{}, automatic{}, save_frames{1}, timing{}, profile{}, stages{};
};

Config read_config(const std::string &name) {
  const auto text = load(name + "/run.txt");
  Config config;
  const std::string all(text.begin(), text.end());
  const auto find = [&](const char *key, uint32_t *a, uint32_t *b) {
    const size_t at = all.find(key);
    if (at == std::string::npos)
      return;
    unsigned x = 0, y = 0;
    const int count = std::sscanf(all.c_str() + at + std::strlen(key), "%u %u", &x, &y);
    if (count >= 1)
      *a = x;
    if (count >= 2 && b)
      *b = y;
  };
  find("render", &config.render_width, &config.render_height);
  find("output", &config.output_width, &config.output_height);
  find("frames", &config.frames, nullptr);
  find("auto", &config.automatic, nullptr);
  find("save", &config.save_frames, nullptr);
  find("timing", &config.timing, nullptr);
  find("profile", &config.profile, nullptr);
  find("stages", &config.stages, nullptr);
  if (!config.render_width || !config.render_height || !config.output_width || !config.output_height ||
      !config.frames)
    throw std::runtime_error("run.txt is incomplete");
  return config;
}

struct FrameInfo {
  float jitter_x, jitter_y;
  uint32_t reset;
  float pre_exposure;
};

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
  uint32_t queue_family{};
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkCommandPool pool{};
  VkCommandBuffer command{};
  VkFence fence{};
  VkPipelineCache pipeline_cache{};
  Image color, depth, motion, output;
  VkBuffer staging{};
  VkDeviceMemory staging_memory{};
  uint8_t *mapped{};
  VkDeviceSize depth_at{}, motion_at{}, output_at{};
};

uint32_t memory_type(const App &app, uint32_t bits, VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < app.memory_properties.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (app.memory_properties.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  throw std::runtime_error("required memory type unavailable");
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
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateImage(app.device, &info, nullptr, &result.image), "vkCreateImage");
  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(app.device, result.image, &requirements);
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex =
      memory_type(app, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  check(vkAllocateMemory(app.device, &allocation, nullptr, &result.memory), "vkAllocateMemory(image)");
  check(vkBindImageMemory(app.device, result.image, result.memory, 0), "vkBindImageMemory");
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = result.image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = format;
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  check(vkCreateImageView(app.device, &view, nullptr, &result.view), "vkCreateImageView");
  return result;
}

void image_barrier(VkCommandBuffer command, VkImage image, VkImageLayout before, VkImageLayout after,
                   VkAccessFlags source, VkAccessFlags destination, VkPipelineStageFlags source_stage,
                   VkPipelineStageFlags destination_stage) {
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcAccessMask = source;
  barrier.dstAccessMask = destination;
  barrier.oldLayout = before;
  barrier.newLayout = after;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(command, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void initialize(App &app) {
  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.pApplicationName = "PS5 HelixSR frame runner";
  application.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance.pApplicationInfo = &application;
  check(vkCreateInstance(&instance, nullptr, &app.instance), "vkCreateInstance");
  uint32_t count = 1;
  const VkResult found = vkEnumeratePhysicalDevices(app.instance, &count, &app.physical);
  if ((found != VK_SUCCESS && found != VK_INCOMPLETE) || !count)
    throw std::runtime_error("no Vulkan physical device");
  uint32_t families = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(app.physical, &families, nullptr);
  std::vector<VkQueueFamilyProperties> properties(families);
  vkGetPhysicalDeviceQueueFamilyProperties(app.physical, &families, properties.data());
  const auto family = std::find_if(properties.begin(), properties.end(),
                                   [](const auto &item) { return item.queueFlags & VK_QUEUE_COMPUTE_BIT; });
  if (family == properties.end())
    throw std::runtime_error("no compute queue family");
  app.queue_family = uint32_t(family - properties.begin());

  VkPhysicalDevice16BitStorageFeatures storage16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  available.pNext = &storage16;
  vkGetPhysicalDeviceFeatures2(app.physical, &available);
  VkPhysicalDeviceProperties device_properties{};
  vkGetPhysicalDeviceProperties(app.physical, &device_properties);
  report("HELIXSR_DEVICE name=%s int16=%u storage16=%u\n", device_properties.deviceName,
         available.features.shaderInt16, storage16.storageBuffer16BitAccess);
  if (!available.features.shaderInt16 || !storage16.storageBuffer16BitAccess)
    throw std::runtime_error("the device lacks 16-bit integers or 16-bit storage-buffer access");

  VkPhysicalDevice16BitStorageFeatures enable16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  enable16.storageBuffer16BitAccess = VK_TRUE;
#ifdef HELIXSR_HOST
  VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivative{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
  derivative.computeDerivativeGroupLinear = VK_TRUE;
  VkPhysicalDeviceShaderFloat16Int8Features float16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  float16.shaderFloat16 = VK_TRUE;
  float16.pNext = &derivative;
  enable16.pNext = &float16;
  const char *extensions[] = {VK_KHR_16BIT_STORAGE_EXTENSION_NAME, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME};
#else
  /* ps5vk takes the 16-bit storage feature only with its extension named. */
  const char *extensions[] = {VK_KHR_16BIT_STORAGE_EXTENSION_NAME};
#endif
  VkPhysicalDeviceFeatures2 enabled{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  enabled.features.shaderInt16 = VK_TRUE;
#ifdef HELIXSR_HOST
  enabled.features.shaderInt64 = VK_TRUE;
  enabled.features.shaderStorageImageExtendedFormats = VK_TRUE;
#endif
  enabled.pNext = &enable16;
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue.queueFamilyIndex = app.queue_family;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device.pNext = &enabled;
  device.queueCreateInfoCount = 1;
  device.pQueueCreateInfos = &queue;
  device.enabledExtensionCount = uint32_t(sizeof(extensions) / sizeof(extensions[0]));
  device.ppEnabledExtensionNames = extensions;
  check(vkCreateDevice(app.physical, &device, nullptr, &app.device), "vkCreateDevice");
  vkGetDeviceQueue(app.device, app.queue_family, 0, &app.queue);
  vkGetPhysicalDeviceMemoryProperties(app.physical, &app.memory_properties);
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool.queueFamilyIndex = app.queue_family;
  check(vkCreateCommandPool(app.device, &pool, nullptr, &app.pool), "vkCreateCommandPool");
  VkCommandBufferAllocateInfo command{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  command.commandPool = app.pool;
  command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command.commandBufferCount = 1;
  check(vkAllocateCommandBuffers(app.device, &command, &app.command), "vkAllocateCommandBuffers");
  VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  check(vkCreateFence(app.device, &fence, nullptr, &app.fence), "vkCreateFence");
  VkPipelineCacheCreateInfo cache{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  check(vkCreatePipelineCache(app.device, &cache, nullptr, &app.pipeline_cache), "vkCreatePipelineCache");
}

void begin(App &app) {
  check(vkResetCommandBuffer(app.command, 0), "vkResetCommandBuffer");
  VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(app.command, &info), "vkBeginCommandBuffer");
}

double submit(App &app) {
  check(vkEndCommandBuffer(app.command), "vkEndCommandBuffer");
  VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  info.commandBufferCount = 1;
  info.pCommandBuffers = &app.command;
  const double start = now_ms();
  check(vkQueueSubmit(app.queue, 1, &info, app.fence), "vkQueueSubmit");
  check(vkWaitForFences(app.device, 1, &app.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  const double elapsed = now_ms() - start;
  check(vkResetFences(app.device, 1, &app.fence), "vkResetFences");
  return elapsed;
}

void make_resources(App &app, const Config &config) {
  const VkImageUsageFlags input = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  app.color = make_image(app, VK_FORMAT_R16G16B16A16_SFLOAT, config.render_width, config.render_height, input);
  app.depth = make_image(app, VK_FORMAT_R32_SFLOAT, config.render_width, config.render_height, input);
  app.motion = make_image(app, VK_FORMAT_R16G16_SFLOAT, config.render_width, config.render_height, input);
  app.output = make_image(app, VK_FORMAT_R16G16B16A16_SFLOAT, config.output_width, config.output_height,
                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  const VkDeviceSize pixels = VkDeviceSize(config.render_width) * config.render_height;
  app.depth_at = pixels * 8;
  app.motion_at = app.depth_at + pixels * 4;
  app.output_at = app.motion_at + pixels * 4;
  VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer.size = app.output_at + VkDeviceSize(config.output_width) * config.output_height * 8;
  buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(app.device, &buffer, nullptr, &app.staging), "vkCreateBuffer(staging)");
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(app.device, app.staging, &requirements);
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = memory_type(app, requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  check(vkAllocateMemory(app.device, &allocation, nullptr, &app.staging_memory), "vkAllocateMemory(staging)");
  check(vkBindBufferMemory(app.device, app.staging, app.staging_memory, 0), "vkBindBufferMemory(staging)");
  check(vkMapMemory(app.device, app.staging_memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void **>(&app.mapped)),
        "vkMapMemory(staging)");
}

void upload_inputs(App &app, const std::string &name, const Config &config, uint32_t frame, bool first) {
  char prefix[160];
  std::snprintf(prefix, sizeof(prefix), "%s/in/%02u", name.c_str(), frame);
  const size_t pixels = size_t(config.render_width) * config.render_height;
  const auto color = load(std::string(prefix) + "/color.rgba16f", pixels * 8);
  const auto depth = load(std::string(prefix) + "/depth.r32f", pixels * 4);
  const auto motion = load(std::string(prefix) + "/motion.rg16f", pixels * 4);
  std::memcpy(app.mapped, color.data(), color.size());
  std::memcpy(app.mapped + app.depth_at, depth.data(), depth.size());
  std::memcpy(app.mapped + app.motion_at, motion.data(), motion.size());
  begin(app);
  const Image *inputs[] = {&app.color, &app.depth, &app.motion};
  const VkDeviceSize at[] = {0, app.depth_at, app.motion_at};
  for (int i = 0; i < 3; ++i) {
    image_barrier(app.command, inputs[i]->image,
                  first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, first ? 0 : VK_ACCESS_SHADER_READ_BIT,
                  VK_ACCESS_TRANSFER_WRITE_BIT,
                  first ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy region{};
    region.bufferOffset = at[i];
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {config.render_width, config.render_height, 1};
    vkCmdCopyBufferToImage(app.command, app.staging, inputs[i]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);
    image_barrier(app.command, inputs[i]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }
  submit(app);
}

double upscale(App &app, ps5helixsr_context *context, const FrameInfo &info, bool first, bool partial = false) {
  begin(app);
  image_barrier(app.command, app.output.image,
                first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                first ? 0 : VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                first ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  ps5helixsr_dispatch_desc dispatch{};
  dispatch.struct_size = sizeof(dispatch);
  dispatch.command_buffer = app.command;
  dispatch.color = app.color.view;
  dispatch.depth = app.depth.view;
  dispatch.motion_vectors = app.motion.view;
  dispatch.output = app.output.view;
  dispatch.color_layout = dispatch.depth_layout = dispatch.motion_vectors_layout =
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  dispatch.jitter_x = info.jitter_x;
  dispatch.jitter_y = info.jitter_y;
  dispatch.pre_exposure = info.pre_exposure;
  dispatch.reset = info.reset;
  const ps5helixsr_result result = ps5helixsr_dispatch(context, &dispatch);
  if (result != PS5HELIXSR_OK)
    throw std::runtime_error("ps5helixsr_dispatch = " + std::to_string(result));
  image_barrier(app.command, app.output.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
  const double elapsed = submit(app);
  if (ps5helixsr_context_notify_completed(context) != PS5HELIXSR_OK && !partial)
    throw std::runtime_error("completion notification rejected");
  return elapsed;
}

void read_output(App &app, const Config &config) {
  begin(app);
  VkBufferImageCopy region{};
  region.bufferOffset = app.output_at;
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {config.output_width, config.output_height, 1};
  vkCmdCopyImageToBuffer(app.command, app.output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, app.staging, 1,
                         &region);
  VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  ready.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(app.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &ready, 0,
                       nullptr, 0, nullptr);
  submit(app);
}

/* Every buffer of the network whose bytes changed since it was last saved. */
void save_stage(App &app, ps5helixsr_context *context, const std::string &name, uint32_t step, const char *step_name,
                VkDeviceSize staging_bytes, std::map<std::string, uint32_t> &saved) {
  std::vector<ps5helixsr_debug_resource> resources(64);
  resources.resize(ps5helixsr_debug_resources(context, resources.data(), 64));
  for (const auto &resource : resources) {
    if (!resource.buffer || !resource.bytes || resource.bytes > staging_bytes ||
        !std::strcmp(resource.name, "WEIGHTS") || !std::strcmp(resource.name, "UNIFORMS"))
      continue;
    begin(app);
    VkBufferCopy region{0, 0, resource.bytes};
    vkCmdCopyBuffer(app.command, resource.buffer, app.staging, 1, &region);
    VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ready.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(app.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &ready, 0,
                         nullptr, 0, nullptr);
    submit(app);
    const uint32_t crc = crc32(app.mapped, size_t(resource.bytes));
    const auto last = saved.find(resource.name);
    if (last != saved.end() && last->second == crc)
      continue;
    saved[resource.name] = crc;
    char file[240];
    std::snprintf(file, sizeof(file), "%s-stage-%02u-%s.bin", name.c_str(), step, resource.name);
    save(file, app.mapped, size_t(resource.bytes));
    report("HELIXSR_STAGE step=%u name=%s buffer=%s bytes=%llu crc32=%08x\n", step, step_name, resource.name,
           static_cast<unsigned long long>(resource.bytes), crc);
  }
}

void destroy_image(App &app, Image &image) {
  if (image.view)
    vkDestroyImageView(app.device, image.view, nullptr);
  if (image.image)
    vkDestroyImage(app.device, image.image, nullptr);
  if (image.memory)
    vkFreeMemory(app.device, image.memory, nullptr);
  image = {};
}

void destroy_resources(App &app) {
  vkDeviceWaitIdle(app.device);
  for (Image *image : {&app.color, &app.depth, &app.motion, &app.output})
    destroy_image(app, *image);
  if (app.mapped)
    vkUnmapMemory(app.device, app.staging_memory);
  if (app.staging)
    vkDestroyBuffer(app.device, app.staging, nullptr);
  if (app.staging_memory)
    vkFreeMemory(app.device, app.staging_memory, nullptr);
  app.mapped = nullptr;
  app.staging = VK_NULL_HANDLE;
  app.staging_memory = VK_NULL_HANDLE;
}

void run_case(App &app, const std::string &name, const std::vector<uint8_t> &model) {
    const Config config = read_config(name);
    report("HELIXSR_CASE name=%s render=%ux%u output=%ux%u frames=%u auto=%u timing=%u\n", name.c_str(),
           config.render_width, config.render_height, config.output_width, config.output_height, config.frames,
           config.automatic, config.timing);
    const auto frames = load(name + "/frames.bin");
    if (frames.size() < size_t(config.frames) * sizeof(FrameInfo))
      throw std::runtime_error("frames.bin holds fewer frames than requested");
    make_resources(app, config);

    ps5helixsr_context_desc description{};
    description.struct_size = sizeof(description);
    description.physical_device = app.physical;
    description.device = app.device;
    description.output_width = config.output_width;
    description.output_height = config.output_height;
    description.render_width = config.render_width;
    description.render_height = config.render_height;
    description.flags = config.automatic ? PS5HELIXSR_FLAG_AUTO_EXPOSURE : 0u;
    description.canonical_weights = model.data();
    description.canonical_weights_bytes = model.size();
    description.pipeline_cache = app.pipeline_cache;
    ps5helixsr_memory_requirements memory{};
    if (ps5helixsr_get_memory_requirements(&description, &memory) != PS5HELIXSR_OK)
      throw std::runtime_error("memory requirements rejected");
    ps5helixsr_context *context = nullptr;
    const double start = now_ms();
    const ps5helixsr_result created = ps5helixsr_context_create(&description, &context);
    report("HELIXSR_CONTEXT result=%d ms=%.1f device_bytes=%llu pipelines=%u\n", created, now_ms() - start,
           static_cast<unsigned long long>(memory.device_bytes), memory.pipeline_count);
    if (created != PS5HELIXSR_OK)
      throw std::runtime_error("context creation failed");

    const size_t output_bytes = size_t(config.output_width) * config.output_height * 8;
    FrameInfo info{};
    for (uint32_t frame = 0; frame < config.frames; ++frame) {
      std::memcpy(&info, frames.data() + frame * sizeof(info), sizeof(info));
      upload_inputs(app, name, config, frame, frame == 0);
      double elapsed = 0;
      if (config.stages && frame == 0) {
        std::vector<ps5helixsr_debug_step> steps(128);
        steps.resize(ps5helixsr_debug_steps(context, 1, steps.data(), 128));
        std::map<std::string, uint32_t> saved;
        const VkDeviceSize staging_bytes = app.output_at + output_bytes;
        for (uint32_t step = 0; step < steps.size(); ++step) {
          ps5helixsr_debug_set_range(context, step, 1);
          elapsed += upscale(app, context, info, step == 0, true);
          save_stage(app, context, name, step, steps[step].name, staging_bytes, saved);
        }
        ps5helixsr_debug_set_range(context, 0, UINT32_MAX);
      } else {
        elapsed = upscale(app, context, info, frame == 0);
      }
      read_output(app, config);
      const uint8_t *pixels = app.mapped + app.output_at;
      report("HELIXSR_FRAME index=%u reset=%u ms=%.3f crc32=%08x\n", frame, info.reset, elapsed,
             crc32(pixels, output_bytes));
      if (config.save_frames) {
        char file[200];
        std::snprintf(file, sizeof(file), "%s-%02u.rgba16f", name.c_str(), frame);
        save(file, pixels, output_bytes);
      }
    }
    if (config.timing) {
      /* The last frame's inputs again and again: the upscaler alone, submit to completion. */
      info.reset = 0;
      double total = 0, best = 1e9, worst = 0;
      for (uint32_t i = 0; i < config.timing; ++i) {
        const double elapsed = upscale(app, context, info, false);
        total += elapsed;
        best = std::min(best, elapsed);
        worst = std::max(worst, elapsed);
      }
      report("HELIXSR_TIMING frames=%u mean_ms=%.3f min_ms=%.3f max_ms=%.3f\n", config.timing,
             total / config.timing, best, worst);
    }
    if (config.profile) {
      /* Each step on its own, submit to completion; "none" is the cost of an empty submission. */
      info.reset = 0;
      std::vector<ps5helixsr_debug_step> steps(128);
      steps.resize(ps5helixsr_debug_steps(context, 0, steps.data(), 128));
      for (uint32_t step = 0; step <= steps.size(); ++step) {
        double total = 0, best = 1e9;
        for (uint32_t i = 0; i < config.profile; ++i) {
          ps5helixsr_debug_set_range(context, step, step < steps.size() ? 1 : 0);
          const double elapsed = upscale(app, context, info, false, true);
          total += elapsed;
          best = std::min(best, elapsed);
        }
        if (step < steps.size())
          report("HELIXSR_STEP index=%u launch=%u name=%s grid=%u,%u,%u mean_ms=%.3f min_ms=%.3f\n", step,
                 steps[step].launch, steps[step].name, steps[step].grid[0], steps[step].grid[1],
                 steps[step].grid[2], total / config.profile, best);
        else
          report("HELIXSR_STEP index=%u name=none mean_ms=%.3f min_ms=%.3f\n", step, total / config.profile, best);
      }
      ps5helixsr_debug_set_range(context, 0, UINT32_MAX);
    }
    vkDeviceWaitIdle(app.device);
    ps5helixsr_context_destroy(context);
    destroy_resources(app);
}

int run() {
#ifndef HELIXSR_HOST
  const int heap = helixsr_native_heap_init();
  if (heap) {
    report("HELIXSR_ERROR native heap = %d\n", heap);
    return 1;
  }
#endif
  char log_path[256];
  std::snprintf(log_path, sizeof(log_path), "%s/helixsr-runner.log", output_root());
  log_file = std::fopen(log_path, "wb");
  App app;
  try {
    report("HELIXSR_BEGIN out=%s\n", output_root());
    const auto list = load("cases.txt");
    const auto model = load("model.bin");
    initialize(app);
    std::string name;
    for (size_t i = 0; i <= list.size(); ++i) {
      const char c = i < list.size() ? char(list[i]) : '\n';
      if (c == '\n' || c == '\r') {
        if (!name.empty())
          run_case(app, name, model);
        name.clear();
      } else {
        name += c;
      }
    }
    report("HELIXSR_END result=0\n");
    return 0;
  } catch (const std::exception &error) {
    report("HELIXSR_ERROR %s\n", error.what());
    report("HELIXSR_END result=1\n");
    return 1;
  }
}

} // namespace

int main() {
  const int result = run();
  if (log_file)
    std::fclose(log_file);
#ifndef HELIXSR_HOST
  /* Leave by ourselves so that no forced close is needed; the log stays in the title folder. */
  sleep(15);
  sceSystemServiceLoadExec("exit", nullptr);
  for (;;)
    sleep(1);
#endif
  return result;
}
