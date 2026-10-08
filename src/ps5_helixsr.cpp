#include "ps5helixsr/ps5_helixsr.h"
#include "ps5_helixsr_debug.h"
#include "runtime_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

using ps5helixsr::Descriptor;
using ps5helixsr::ResourceDomain;
using ps5helixsr::ResourceRef;
using ps5helixsr::RuntimePlan;
using ps5helixsr::Step;
using ps5helixsr::StepKind;

namespace {

struct VulkanError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void vk_check(VkResult result, const char *call) {
  if (result != VK_SUCCESS)
    throw VulkanError(std::string(call) + ": " + std::to_string(result));
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

VkFormat format(DXGI_FORMAT value) {
  switch (value) {
  case DXGI_FORMAT_R16G16B16A16_FLOAT:
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  case DXGI_FORMAT_R16G16_FLOAT:
    return VK_FORMAT_R16G16_SFLOAT;
  case DXGI_FORMAT_R32_FLOAT:
    return VK_FORMAT_R32_SFLOAT;
  case DXGI_FORMAT_R16_FLOAT:
    return VK_FORMAT_R16_SFLOAT;
  default:
    throw std::runtime_error("unsupported image format");
  }
}

uint32_t format_bytes(DXGI_FORMAT value) {
  switch (value) {
  case DXGI_FORMAT_R16G16B16A16_FLOAT:
    return 8;
  case DXGI_FORMAT_R16G16_FLOAT:
    return 4;
  case DXGI_FORMAT_R32_FLOAT:
    return 4;
  case DXGI_FORMAT_R16_FLOAT:
    return 2;
  default:
    throw std::runtime_error("unsupported image format");
  }
}

VkDescriptorType descriptor_type(char kind) {
  switch (kind) {
  case 'U':
    return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  case 'B':
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  case 'I':
    return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  case 'W':
    return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  case 'S':
    return VK_DESCRIPTOR_TYPE_SAMPLER;
  default:
    throw std::runtime_error("unsupported descriptor kind");
  }
}

struct Buffer {
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  VkDeviceSize bytes{};
  void *mapped{};
};

struct Image {
  VkImage image{};
  VkImageView view{};
  VkDeviceMemory memory{};
  uint32_t width{}, height{};
  bool persistent{};
};

struct Pipeline {
  const ps5helixsr::assets::Shader *asset{};
  VkDescriptorSetLayout set_layout{};
  VkPipelineLayout layout{};
  VkShaderModule module{};
  VkPipeline pipeline{};
};

struct Slot {
  std::string key;
  Pipeline *pipeline{};
  VkDescriptorSet set{};
  VkDeviceSize uniform_offset{};
  size_t uniform_capacity{};
};

std::pair<uint32_t, uint32_t> image_size(const bcm::ImageRole &role,
                                         const bcm::Params &params) {
  switch (role.size) {
  case bcm::SizeOne:
    return {1, 1};
  case bcm::SizeRender:
    return {params.Wr, params.Hr};
  case bcm::SizeOutput:
    return {params.Wo, params.Ho};
  case bcm::SizeOutputEvenH:
    return {params.Wo, (params.Ho + 1) & ~1u};
  }
  throw std::runtime_error("invalid image size policy");
}

bool external_image(const bcm::ImageRole &role) {
  return role.game || !std::strcmp(role.name, "OUTPUT");
}

std::string step_key(const Step &step, uint32_t ordinal) {
  return std::to_string(step.launch_index) + ":" + step.shader->name + ":" +
         std::to_string(ordinal);
}

std::vector<std::pair<const Step *, std::string>>
dispatches(const RuntimePlan &plan) {
  std::map<uint32_t, uint32_t> ordinals;
  std::vector<std::pair<const Step *, std::string>> result;
  for (const auto &step : plan.steps)
    if (step.kind == StepKind::Dispatch) {
      auto ordinal = ordinals[step.launch_index]++;
      result.emplace_back(&step, step_key(step, ordinal));
    }
  return result;
}

std::map<std::string, Step> topology(const bcm::Params &base,
                                     const bcm::PlanOptions &options) {
  std::map<std::string, Step> definitions;
  for (uint32_t frame = 0; frame < 64; ++frame) {
    auto params = base;
    params.hasPrev = frame != 0;
    params.reset = frame == 0;
    const auto plan = ps5helixsr::prepare_runtime_plan(params, frame, options);
    for (const auto &[step, key] : dispatches(plan)) {
      auto inserted = definitions.emplace(key, *step);
      if (!inserted.second) {
        auto &old = inserted.first->second;
        if (old.shader != step->shader ||
            old.descriptors.size() != step->descriptors.size())
          throw std::runtime_error("dispatch topology changed");
        if (old.uniform.size() < step->uniform.size())
          old.uniform.resize(step->uniform.size());
      }
    }
  }
  return definitions;
}

} // namespace

struct ps5helixsr_context {
  ps5helixsr_context_desc desc{};
  bcm::Params base{};
  bcm::PlanOptions options{};
  const bcm::NetTables *tables{};
  VkPhysicalDeviceMemoryProperties memory_properties{};
  std::vector<Buffer> buffers;
  Buffer weights, transient_a, transient_b, zero, uniforms, exposure;
  std::vector<Image> images;
  std::vector<Pipeline> pipelines;
  std::map<std::string, Slot> slots;
  VkDescriptorPool descriptor_pool{};
  VkSampler point{}, linear{};
  uint32_t frame{};
  float previous_jitter_x{}, previous_jitter_y{};
  bool initialized{}, in_flight{};
  uint32_t debug_first{0}, debug_count{UINT32_MAX};
};

namespace {

uint32_t memory_type(const ps5helixsr_context &context, uint32_t bits,
                     VkMemoryPropertyFlags wanted) {
  for (uint32_t i = 0; i < context.memory_properties.memoryTypeCount; ++i)
    if ((bits & (1u << i)) &&
        (context.memory_properties.memoryTypes[i].propertyFlags & wanted) ==
            wanted)
      return i;
  throw std::runtime_error("required memory type unavailable");
}

void make_buffer(ps5helixsr_context &context, VkDeviceSize bytes,
                 VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                 bool map, Buffer &out) {
  out.bytes = bytes;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = bytes;
  info.usage = usage;
  vk_check(vkCreateBuffer(context.desc.device, &info, context.desc.allocator,
                          &out.buffer),
           "vkCreateBuffer");
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(context.desc.device, out.buffer, &requirements);
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex =
      memory_type(context, requirements.memoryTypeBits, properties);
  vk_check(vkAllocateMemory(context.desc.device, &allocation,
                            context.desc.allocator, &out.memory),
           "vkAllocateMemory(buffer)");
  vk_check(vkBindBufferMemory(context.desc.device, out.buffer, out.memory, 0),
           "vkBindBufferMemory");
  if (map)
    vk_check(vkMapMemory(context.desc.device, out.memory, 0, VK_WHOLE_SIZE, 0,
                         &out.mapped),
             "vkMapMemory");
}

void make_image(ps5helixsr_context &context, const bcm::ImageRole &role,
                Image &out) {
  const auto [width, height] = image_size(role, context.base);
  out.width = width;
  out.height = height;
  out.persistent = role.persistent;
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format(role.format);
  info.extent = {width, height, 1};
  info.mipLevels = info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  vk_check(vkCreateImage(context.desc.device, &info, context.desc.allocator,
                         &out.image),
           "vkCreateImage");
  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(context.desc.device, out.image, &requirements);
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = requirements.size;
  allocation.memoryTypeIndex = memory_type(context, requirements.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vk_check(vkAllocateMemory(context.desc.device, &allocation,
                            context.desc.allocator, &out.memory),
           "vkAllocateMemory(image)");
  vk_check(vkBindImageMemory(context.desc.device, out.image, out.memory, 0),
           "vkBindImageMemory");
  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view.image = out.image;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = info.format;
  view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vk_check(vkCreateImageView(context.desc.device, &view, context.desc.allocator,
                             &out.view),
           "vkCreateImageView");
}

/* SHA-256 (FIPS 180-4), for the identity of a kernel that comes from a file. */
std::string sha256_hex(const uint8_t *data, size_t bytes) {
  static const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  const uint64_t bits = uint64_t(bytes) * 8;
  const size_t padded = (bytes + 9 + 63) / 64 * 64;
  for (size_t block = 0; block < padded; block += 64) {
    uint8_t chunk[64];
    for (size_t i = 0; i < 64; ++i) {
      const size_t at = block + i;
      chunk[i] = at < bytes ? data[at] : at == bytes ? 0x80 : at >= padded - 8 ? uint8_t(bits >> (8 * (padded - 1 - at))) : 0;
    }
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = uint32_t(chunk[4 * i]) << 24 | uint32_t(chunk[4 * i + 1]) << 16 | uint32_t(chunk[4 * i + 2]) << 8 | chunk[4 * i + 3];
    for (int i = 16; i < 64; ++i)
      w[i] = w[i - 16] + (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
             (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t v[8];
    std::memcpy(v, h, sizeof(v));
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = v[7] + (rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25)) + ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
      const uint32_t t2 = (rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22)) + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
      v[7] = v[6]; v[6] = v[5]; v[5] = v[4]; v[4] = v[3] + t1;
      v[3] = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = t1 + t2;
    }
    for (int i = 0; i < 8; ++i)
      h[i] += v[i];
  }
  char text[65];
  for (int i = 0; i < 8; ++i)
    std::snprintf(text + 8 * i, 9, "%08x", h[i]);
  return text;
}

/* A kernel in the application's file (tools/pack_kernels.py): "HXKP", version, count,
   entries of name[48], offset, bytes. Returns nullptr unless it is there, whole, and the
   module this build pins. */
const uint32_t *packed_kernel(const ps5helixsr_context_desc &desc,
                              const ps5helixsr::assets::Shader &asset, size_t *words) {
  const auto *file = static_cast<const uint8_t *>(desc.kernels);
  uint32_t head[3];
  if (!file || desc.kernels_bytes < sizeof(head))
    return nullptr;
  std::memcpy(head, file, sizeof(head));
  if (head[0] != 0x504b5848u || head[1] != 1 || head[2] > 256 ||
      desc.kernels_bytes < sizeof(head) + size_t(head[2]) * 56)
    return nullptr;
  for (uint32_t i = 0; i < head[2]; ++i) {
    const uint8_t *entry = file + sizeof(head) + size_t(i) * 56;
    uint32_t span[2];
    std::memcpy(span, entry + 48, sizeof(span));
    if (entry[47] || std::strcmp(reinterpret_cast<const char *>(entry), asset.name))
      continue;
    if (span[1] < 20 || span[1] % 4 || span[0] % 4 || span[0] > desc.kernels_bytes ||
        span[1] > desc.kernels_bytes - span[0] ||
        sha256_hex(file + span[0], span[1]) != asset.sha256)
      return nullptr;
    *words = span[1] / 4;
    return reinterpret_cast<const uint32_t *>(file + span[0]);
  }
  return nullptr;
}

/* Every kernel this build does not carry must come with the description. */
bool kernels_present(const ps5helixsr_context_desc &desc) {
  for (size_t i = 0; i < ps5helixsr::assets::size(); ++i) {
    const auto &asset = ps5helixsr::assets::data()[i];
    size_t words = 0;
    if (!asset.words && !packed_kernel(desc, asset, &words))
      return false;
  }
  return true;
}

Pipeline *find_pipeline(ps5helixsr_context &context,
                        const ps5helixsr::assets::Shader *asset) {
  for (auto &pipeline : context.pipelines)
    if (pipeline.asset == asset)
      return &pipeline;
  throw std::runtime_error("pipeline missing");
}

void create_pipelines(ps5helixsr_context &context) {
  context.pipelines.resize(ps5helixsr::assets::size());
  for (size_t i = 0; i < context.pipelines.size(); ++i) {
    auto &pipeline = context.pipelines[i];
    pipeline.asset = &ps5helixsr::assets::data()[i];
    std::vector<VkDescriptorSetLayoutBinding> bindings(
        pipeline.asset->descriptor_count);
    for (uint32_t j = 0; j < bindings.size(); ++j)
      bindings[j] = {j, descriptor_type(pipeline.asset->descriptors[j].kind), 1,
                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo set{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set.bindingCount = uint32_t(bindings.size());
    set.pBindings = bindings.data();
    vk_check(vkCreateDescriptorSetLayout(context.desc.device, &set,
                                         context.desc.allocator,
                                         &pipeline.set_layout),
             "vkCreateDescriptorSetLayout");
    VkPipelineLayoutCreateInfo layout{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &pipeline.set_layout;
    vk_check(vkCreatePipelineLayout(context.desc.device, &layout,
                                    context.desc.allocator, &pipeline.layout),
             "vkCreatePipelineLayout");
    VkShaderModuleCreateInfo module{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module.codeSize = pipeline.asset->words * 4;
    module.pCode = pipeline.asset->code;
    if (!pipeline.asset->words) {
      size_t words = 0;
      module.pCode = packed_kernel(context.desc, *pipeline.asset, &words);
      module.codeSize = words * 4;
      if (!module.pCode)
        throw std::runtime_error("kernel missing");
    }
#ifdef PS5HELIXSR_DEBUG_SHADERS
    /* Host diagnostics only: an instrumented module replaces the packaged one. */
    std::vector<uint32_t> replacement;
    if (const char *directory = std::getenv("PS5HELIXSR_DEBUG_SHADER_DIR")) {
      const std::string path =
          std::string(directory) + "/" + pipeline.asset->name + ".spv";
      if (FILE *file = std::fopen(path.c_str(), "rb")) {
        std::fseek(file, 0, SEEK_END);
        replacement.resize(size_t(std::ftell(file)) / 4);
        std::fseek(file, 0, SEEK_SET);
        if (std::fread(replacement.data(), 4, replacement.size(), file) ==
            replacement.size()) {
          module.codeSize = replacement.size() * 4;
          module.pCode = replacement.data();
          std::fprintf(stderr, "debug shader: %s\n", path.c_str());
        }
        std::fclose(file);
      }
    }
#endif
    vk_check(vkCreateShaderModule(context.desc.device, &module,
                                  context.desc.allocator, &pipeline.module),
             "vkCreateShaderModule");
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    subgroup.requiredSubgroupSize = 32;
    VkComputePipelineCreateInfo info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.layout = pipeline.layout;
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = pipeline.module;
    info.stage.pName = "main";
    if (context.desc.flags & PS5HELIXSR_FLAG_SUBGROUP_SIZE_CONTROL)
      info.stage.pNext = &subgroup;
    vk_check(vkCreateComputePipelines(
                 context.desc.device, context.desc.pipeline_cache, 1, &info,
                 context.desc.allocator, &pipeline.pipeline),
             "vkCreateComputePipelines");
  }
}

void create_sampler(ps5helixsr_context &context, VkFilter filter,
                    VkSampler &sampler) {
  VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  info.magFilter = info.minFilter = filter;
  info.mipmapMode = filter == VK_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                               : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  info.addressModeU = info.addressModeV = info.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  info.minLod = info.maxLod = 0;
  vk_check(vkCreateSampler(context.desc.device, &info, context.desc.allocator,
                           &sampler),
           "vkCreateSampler");
}

void collect_slots(ps5helixsr_context &context) {
  const auto definitions = topology(context.base, context.options);
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.desc.physical_device, &properties);
  const uint64_t alignment =
      std::max<uint64_t>(1, properties.limits.minUniformBufferOffsetAlignment);
  uint64_t uniform_bytes = 0;
  std::map<VkDescriptorType, uint32_t> counts;
  for (const auto &[key, step] : definitions) {
    Slot slot;
    slot.key = key;
    slot.pipeline = find_pipeline(context, step.shader);
    slot.uniform_offset = align_up(uniform_bytes, alignment);
    slot.uniform_capacity = step.uniform.size();
    uniform_bytes = slot.uniform_offset + slot.uniform_capacity;
    for (const auto &descriptor : step.descriptors)
      ++counts[descriptor_type(descriptor.kind)];
    context.slots.emplace(key, std::move(slot));
  }
  uniform_bytes = align_up(uniform_bytes, alignment);
  make_buffer(context, uniform_bytes,
              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              true, context.uniforms);
  std::vector<VkDescriptorPoolSize> sizes;
  for (auto [type, count] : counts)
    sizes.push_back({type, count});
  VkDescriptorPoolCreateInfo pool{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool.maxSets = uint32_t(context.slots.size());
  pool.poolSizeCount = uint32_t(sizes.size());
  pool.pPoolSizes = sizes.data();
  vk_check(vkCreateDescriptorPool(context.desc.device, &pool,
                                  context.desc.allocator,
                                  &context.descriptor_pool),
           "vkCreateDescriptorPool");
  for (auto &[key, slot] : context.slots) {
    VkDescriptorSetAllocateInfo allocation{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = context.descriptor_pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &slot.pipeline->set_layout;
    vk_check(
        vkAllocateDescriptorSets(context.desc.device, &allocation, &slot.set),
        "vkAllocateDescriptorSets");
  }
}

void destroy_buffer(ps5helixsr_context &context, Buffer &buffer) {
  if (buffer.mapped)
    vkUnmapMemory(context.desc.device, buffer.memory);
  if (buffer.buffer)
    vkDestroyBuffer(context.desc.device, buffer.buffer, context.desc.allocator);
  if (buffer.memory)
    vkFreeMemory(context.desc.device, buffer.memory, context.desc.allocator);
  buffer = {};
}

void cleanup(ps5helixsr_context &context) {
  if (!context.desc.device)
    return;
  if (context.descriptor_pool)
    vkDestroyDescriptorPool(context.desc.device, context.descriptor_pool,
                            context.desc.allocator);
  if (context.point)
    vkDestroySampler(context.desc.device, context.point,
                     context.desc.allocator);
  if (context.linear)
    vkDestroySampler(context.desc.device, context.linear,
                     context.desc.allocator);
  for (auto &pipeline : context.pipelines) {
    if (pipeline.pipeline)
      vkDestroyPipeline(context.desc.device, pipeline.pipeline,
                        context.desc.allocator);
    if (pipeline.module)
      vkDestroyShaderModule(context.desc.device, pipeline.module,
                            context.desc.allocator);
    if (pipeline.layout)
      vkDestroyPipelineLayout(context.desc.device, pipeline.layout,
                              context.desc.allocator);
    if (pipeline.set_layout)
      vkDestroyDescriptorSetLayout(context.desc.device, pipeline.set_layout,
                                   context.desc.allocator);
  }
  for (auto &image : context.images) {
    if (image.view)
      vkDestroyImageView(context.desc.device, image.view,
                         context.desc.allocator);
    if (image.image)
      vkDestroyImage(context.desc.device, image.image, context.desc.allocator);
    if (image.memory)
      vkFreeMemory(context.desc.device, image.memory, context.desc.allocator);
  }
  for (auto &buffer : context.buffers)
    destroy_buffer(context, buffer);
  destroy_buffer(context, context.weights);
  destroy_buffer(context, context.transient_a);
  destroy_buffer(context, context.transient_b);
  destroy_buffer(context, context.zero);
  destroy_buffer(context, context.exposure);
  destroy_buffer(context, context.uniforms);
}

bool valid_context_desc(const ps5helixsr_context_desc *desc,
                        bool require_weights) {
  if (!desc || desc->struct_size != sizeof(*desc) || !desc->physical_device ||
      !desc->device || !desc->output_width || !desc->output_height ||
      !desc->render_width || !desc->render_height ||
      desc->render_width > desc->output_width ||
      desc->render_height > desc->output_height || desc->output_width > 4096 ||
      desc->output_height > 4096 ||
      (require_weights &&
       (!desc->canonical_weights ||
        desc->canonical_weights_bytes != bcm::canonical_bytes())) ||
      (desc->flags & ~(PS5HELIXSR_FLAG_AUTO_EXPOSURE |
                       PS5HELIXSR_FLAG_SUBGROUP_SIZE_CONTROL)))
    return false;
  /* The main network at every ratio from 1x to 3x, as HelixSR runs by default
     (NVIDIA's own choice above 2.5x is a second network, which is not carried).
     A render size is a truncated quotient: 2560 / 3 is 853, a hair over 3x. */
  return desc->output_width <= 3 * (desc->render_width + 1) &&
         desc->output_height <= 3 * (desc->render_height + 1);
}

ps5helixsr_result translate_exception() {
  try {
    throw;
  } catch (const std::bad_alloc &) {
    return PS5HELIXSR_ERROR_OUT_OF_MEMORY;
  } catch (const VulkanError &) {
    return PS5HELIXSR_ERROR_VULKAN;
  } catch (...) {
    return PS5HELIXSR_ERROR_UNSUPPORTED;
  }
}

struct BoundImage {
  VkImage image{};
  VkImageView view{};
  VkImageLayout layout{VK_IMAGE_LAYOUT_GENERAL};
};

BoundImage resolve_image(ps5helixsr_context &context, uint32_t role,
                         const ps5helixsr_dispatch_desc &dispatch) {
  if (role >= context.tables->imageCount)
    throw std::runtime_error("image role outside table");
  const auto &spec = context.tables->images[role];
  if (!std::strcmp(spec.name, "GAME_COLOR"))
    return {{}, dispatch.color, dispatch.color_layout};
  if (!std::strcmp(spec.name, "GAME_DEPTH"))
    return {{}, dispatch.depth, dispatch.depth_layout};
  if (!std::strcmp(spec.name, "GAME_MV"))
    return {{}, dispatch.motion_vectors, dispatch.motion_vectors_layout};
  if (!std::strcmp(spec.name, "OUTPUT"))
    return {{}, dispatch.output, VK_IMAGE_LAYOUT_GENERAL};
  uint32_t physical = role;
  const int read = bcm::image_index(context.base.net, "MV_HIST_READ");
  const int write = bcm::image_index(context.base.net, "MV_HIST_WRITE");
  if (int(role) == read)
    physical = context.frame & 1 ? uint32_t(write) : uint32_t(read);
  else if (int(role) == write)
    physical = context.frame & 1 ? uint32_t(read) : uint32_t(write);
  auto &image = context.images[physical];
  if (!image.view)
    throw std::runtime_error("application image role unsupported");
  return {image.image, image.view, VK_IMAGE_LAYOUT_GENERAL};
}

Buffer &resolve_buffer(ps5helixsr_context &context, ResourceRef resource) {
  switch (resource.domain) {
  case ResourceDomain::Weights:
    return context.weights;
  case ResourceDomain::Buffer:
    if (resource.index >= context.buffers.size())
      throw std::runtime_error("buffer role outside table");
    return context.buffers[resource.index];
  case ResourceDomain::TransientA:
    return context.transient_a;
  case ResourceDomain::TransientB:
    return context.transient_b;
  case ResourceDomain::Uniform:
    return context.uniforms;
  default:
    throw std::runtime_error("resource is not a buffer");
  }
}

void transition_initial_images(ps5helixsr_context &context,
                               VkCommandBuffer command) {
  std::vector<VkImageMemoryBarrier> barriers;
  for (const auto &image : context.images)
    if (image.image) {
      VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.dstAccessMask = image.persistent ? VK_ACCESS_TRANSFER_WRITE_BIT
                                               : VK_ACCESS_SHADER_READ_BIT |
                                                     VK_ACCESS_SHADER_WRITE_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex =
          VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image.image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      barriers.push_back(barrier);
    }
  if (!barriers.empty())
    vkCmdPipelineBarrier(
        command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, uint32_t(barriers.size()), barriers.data());
}

void clear_history(ps5helixsr_context &context, VkCommandBuffer command) {
  if (context.initialized) {
    std::vector<VkImageMemoryBarrier> barriers;
    for (const auto &image : context.images)
      if (image.image && image.persistent) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(barrier);
      }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, uint32_t(barriers.size()), barriers.data());
  }
  vkCmdFillBuffer(command, context.zero.buffer, 0, VK_WHOLE_SIZE, 0);
  VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  filled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  filled.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &filled, 0,
                       nullptr, 0, nullptr);
  for (const auto &image : context.images)
    if (image.image && image.persistent) {
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {image.width, image.height, 1};
      vkCmdCopyBufferToImage(command, context.zero.buffer, image.image,
                             VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }
  if (!context.options.autoExposure) {
    /* Without automatic exposure the input stage reads the caller's exposure
       (1.0) from EXPOSURE_CUR, as the original does from its own 1x1 texture. */
    const int current = bcm::image_index(context.base.net, "EXPOSURE_CUR");
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {1, 1, 1};
    vkCmdCopyBufferToImage(command, context.exposure.buffer,
                           context.images[current].image,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  }
  VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  done.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &done, 0,
                       nullptr, 0, nullptr);
}

void update_and_record(ps5helixsr_context &context, const Step &step,
                       const std::string &key,
                       const ps5helixsr_dispatch_desc &dispatch) {
  auto found = context.slots.find(key);
  if (found == context.slots.end() ||
      step.uniform.size() > found->second.uniform_capacity)
    throw std::runtime_error("dispatch slot missing");
  auto &slot = found->second;
  std::memcpy(static_cast<uint8_t *>(context.uniforms.mapped) +
                  slot.uniform_offset,
              step.uniform.data(), step.uniform.size());
  std::vector<VkWriteDescriptorSet> writes(step.descriptors.size());
  std::vector<VkDescriptorBufferInfo> buffer_info(step.descriptors.size());
  std::vector<VkDescriptorImageInfo> image_info(step.descriptors.size());
  for (uint32_t i = 0; i < step.descriptors.size(); ++i) {
    const Descriptor &descriptor = step.descriptors[i];
    auto &write = writes[i];
    write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = slot.set;
    write.dstBinding = i;
    write.descriptorCount = 1;
    write.descriptorType = descriptor_type(descriptor.kind);
    if (descriptor.kind == 'S') {
      image_info[i].sampler =
          descriptor.linear_filter ? context.linear : context.point;
      write.pImageInfo = &image_info[i];
    } else if (descriptor.kind == 'I' || descriptor.kind == 'W') {
      const auto image =
          resolve_image(context, descriptor.resource.index, dispatch);
      image_info[i].imageView = image.view;
      image_info[i].imageLayout = image.layout;
      write.pImageInfo = &image_info[i];
    } else {
      Buffer *buffer = nullptr;
      VkDeviceSize offset = descriptor.offset;
      if (descriptor.kind == 'U') {
        buffer = &context.uniforms;
        offset = slot.uniform_offset;
      } else {
        buffer = &resolve_buffer(context, descriptor.resource);
      }
      if (offset + descriptor.range > buffer->bytes)
        throw std::runtime_error("descriptor buffer range invalid");
      buffer_info[i] = {buffer->buffer, offset, descriptor.range};
      write.pBufferInfo = &buffer_info[i];
    }
  }
  vkUpdateDescriptorSets(context.desc.device, uint32_t(writes.size()),
                         writes.data(), 0, nullptr);
  vkCmdBindPipeline(dispatch.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    slot.pipeline->pipeline);
  vkCmdBindDescriptorSets(dispatch.command_buffer,
                          VK_PIPELINE_BIND_POINT_COMPUTE, slot.pipeline->layout,
                          0, 1, &slot.set, 0, nullptr);
  vkCmdDispatch(dispatch.command_buffer, step.grid[0], step.grid[1],
                step.grid[2]);
}

} // namespace

extern "C" ps5helixsr_result ps5helixsr_get_memory_requirements(
    const ps5helixsr_context_desc *desc,
    ps5helixsr_memory_requirements *requirements) {
  if (!requirements || !valid_context_desc(desc, false))
    return PS5HELIXSR_ERROR_INVALID_ARGUMENT;
  try {
    bcm::Params params;
    params.Wo = desc->output_width;
    params.Ho = desc->output_height;
    params.Wr = desc->render_width;
    params.Hr = desc->render_height;
    bcm::finalize(params);
    const auto &tables = bcm::net_tables(params.net);
    uint64_t device = tables.weightBlobBytes;
    for (uint32_t i = 0; i < tables.bufferCount; ++i)
      device += tables.buffers[i].bytes(params);
    uint64_t zero = 0;
    for (uint32_t i = 0; i < tables.imageCount; ++i)
      if (!external_image(tables.images[i])) {
        const auto [width, height] = image_size(tables.images[i], params);
        const uint64_t bytes =
            uint64_t(width) * height * format_bytes(tables.images[i].format);
        device += bytes;
        zero = std::max(zero, bytes);
      }
    const uint64_t pixels = uint64_t(params.Wp) * params.Hp;
    device += 5 * pixels + 5 * pixels / 2 + zero;
    bcm::PlanOptions options;
    options.autoExposure = (desc->flags & PS5HELIXSR_FLAG_AUTO_EXPOSURE) != 0;
    options.displayResMv = false;
    const auto definitions = topology(params, options);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(desc->physical_device, &properties);
    const uint64_t alignment = std::max<uint64_t>(
        1, properties.limits.minUniformBufferOffsetAlignment);
    uint64_t uniforms = 0;
    for (const auto &[key, step] : definitions)
      uniforms = align_up(uniforms, alignment) + step.uniform.size();
    uniforms = align_up(uniforms, alignment);
    *requirements = {device, tables.weightBlobBytes + uniforms,
                     uint32_t(ps5helixsr::assets::size()),
                     uint32_t(definitions.size())};
    return PS5HELIXSR_OK;
  } catch (...) {
    return translate_exception();
  }
}

extern "C" ps5helixsr_result
ps5helixsr_context_create(const ps5helixsr_context_desc *desc,
                          ps5helixsr_context **output) {
  if (!output || !valid_context_desc(desc, true) || !kernels_present(*desc))
    return PS5HELIXSR_ERROR_INVALID_ARGUMENT;
  *output = nullptr;
  auto *context = new (std::nothrow) ps5helixsr_context;
  if (!context)
    return PS5HELIXSR_ERROR_OUT_OF_MEMORY;
  try {
    context->desc = *desc;
    context->base.Wo = desc->output_width;
    context->base.Ho = desc->output_height;
    context->base.Wr = desc->render_width;
    context->base.Hr = desc->render_height;
    bcm::finalize(context->base);
    context->tables = &bcm::net_tables(context->base.net);
    context->options.autoExposure =
        (desc->flags & PS5HELIXSR_FLAG_AUTO_EXPOSURE) != 0;
    context->options.displayResMv = false;
    context->options.hdr = true;
    context->options.depthInverted = true;
    vkGetPhysicalDeviceMemoryProperties(desc->physical_device,
                                        &context->memory_properties);
    std::vector<uint8_t> weights;
    if (!bcm::build_weight_blob(
            context->base.net,
            static_cast<const uint8_t *>(desc->canonical_weights),
            desc->canonical_weights_bytes, weights))
      throw std::runtime_error("canonical model rejected");
    ps5helixsr::reorder_convolution_weights(context->base, context->options,
                                            weights);
    make_buffer(*context, weights.size(),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                true, context->weights);
    std::memcpy(context->weights.mapped, weights.data(), weights.size());
    vkUnmapMemory(desc->device, context->weights.memory);
    context->weights.mapped = nullptr;
    context->buffers.resize(context->tables->bufferCount);
    for (uint32_t i = 0; i < context->tables->bufferCount; ++i)
      make_buffer(*context, context->tables->buffers[i].bytes(context->base),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false,
                  context->buffers[i]);
    const uint64_t pixels = uint64_t(context->base.Wp) * context->base.Hp;
    make_buffer(*context, 5 * pixels,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false,
                context->transient_a);
    make_buffer(*context, 5 * pixels / 2,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false,
                context->transient_b);
    context->images.resize(context->tables->imageCount);
    uint64_t zero_bytes = 0;
    for (uint32_t i = 0; i < context->tables->imageCount; ++i)
      if (!external_image(context->tables->images[i])) {
        make_image(*context, context->tables->images[i], context->images[i]);
        zero_bytes = std::max(
            zero_bytes, uint64_t(context->images[i].width) *
                            context->images[i].height *
                            format_bytes(context->tables->images[i].format));
      }
    make_buffer(*context, zero_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, context->zero);
    make_buffer(*context, 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                true, context->exposure);
    const uint16_t one = 0x3c00; /* half 1.0 */
    std::memset(context->exposure.mapped, 0, 16);
    std::memcpy(context->exposure.mapped, &one, sizeof(one));
    vkUnmapMemory(desc->device, context->exposure.memory);
    context->exposure.mapped = nullptr;
    create_sampler(*context, VK_FILTER_NEAREST, context->point);
    create_sampler(*context, VK_FILTER_LINEAR, context->linear);
    create_pipelines(*context);
    collect_slots(*context);
    *output = context;
    return PS5HELIXSR_OK;
  } catch (...) {
    const auto result = translate_exception();
    cleanup(*context);
    delete context;
    return result;
  }
}

extern "C" void ps5helixsr_context_destroy(ps5helixsr_context *context) {
  if (!context)
    return;
  cleanup(*context);
  delete context;
}

extern "C" ps5helixsr_result
ps5helixsr_dispatch(ps5helixsr_context *context,
                    const ps5helixsr_dispatch_desc *desc) {
  if (!context || !desc || desc->struct_size != sizeof(*desc) ||
      !desc->command_buffer || !desc->color || !desc->depth ||
      !desc->motion_vectors || !desc->output || !(desc->pre_exposure > 0) ||
      !std::isfinite(desc->pre_exposure) || !std::isfinite(desc->jitter_x) ||
      !std::isfinite(desc->jitter_y) || desc->reset > 1 ||
      desc->output == desc->color ||
      (desc->color_layout != VK_IMAGE_LAYOUT_GENERAL &&
       desc->color_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ||
      (desc->depth_layout != VK_IMAGE_LAYOUT_GENERAL &&
       desc->depth_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ||
      (desc->motion_vectors_layout != VK_IMAGE_LAYOUT_GENERAL &&
       desc->motion_vectors_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
    return PS5HELIXSR_ERROR_INVALID_ARGUMENT;
  if (context->in_flight)
    return PS5HELIXSR_ERROR_IN_FLIGHT;
  try {
    auto params = context->base;
    const bool reset = desc->reset || !context->initialized;
    params.jx = desc->jitter_x;
    params.jy = desc->jitter_y;
    params.pjx = context->previous_jitter_x;
    params.pjy = context->previous_jitter_y;
    params.hasPrev = context->initialized && !reset;
    params.pre = desc->pre_exposure;
    params.reset = reset;
    const auto plan = ps5helixsr::prepare_runtime_plan(params, context->frame,
                                                       context->options);
    const uint32_t first = context->debug_first;
    const uint64_t last = uint64_t(first) + context->debug_count;
    if (!context->initialized && first == 0)
      transition_initial_images(*context, desc->command_buffer);
    if (reset && first == 0)
      clear_history(*context, desc->command_buffer);
    const VkMemoryBarrier between{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    std::map<uint32_t, uint32_t> ordinals;
    bool dispatched = first != 0;
    uint32_t index = 0;
    for (const auto &step : plan.steps) {
      if (step.kind == StepKind::Barrier)
        continue;
      const uint32_t current = index++;
      if (current < first || current >= last) {
        if (step.kind == StepKind::Dispatch)
          ++ordinals[step.launch_index];
        continue;
      }
      if (step.kind == StepKind::FillBuffer) {
        if (dispatched) {
          VkMemoryBarrier to_fill{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                  VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_ACCESS_TRANSFER_WRITE_BIT};
          vkCmdPipelineBarrier(desc->command_buffer,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &to_fill,
                               0, nullptr, 0, nullptr);
        }
        auto &buffer = resolve_buffer(*context, step.resource);
        vkCmdFillBuffer(desc->command_buffer, buffer.buffer, step.offset,
                        step.size, step.value);
        VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                               VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(desc->command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &filled, 0, nullptr, 0, nullptr);
        dispatched = false;
        continue;
      }
      if (dispatched)
        vkCmdPipelineBarrier(desc->command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &between, 0, nullptr, 0, nullptr);
      const auto key = step_key(step, ordinals[step.launch_index]++);
      update_and_record(*context, step, key, *desc);
      dispatched = true;
    }
    if (last < index)
      return PS5HELIXSR_OK; /* diagnostic partial frame */
    context->previous_jitter_x = desc->jitter_x;
    context->previous_jitter_y = desc->jitter_y;
    context->initialized = true;
    context->in_flight = true;
    return PS5HELIXSR_OK;
  } catch (...) {
    return translate_exception();
  }
}

extern "C" ps5helixsr_result
ps5helixsr_context_notify_completed(ps5helixsr_context *context) {
  if (!context)
    return PS5HELIXSR_ERROR_INVALID_ARGUMENT;
  if (!context->in_flight)
    return PS5HELIXSR_ERROR_INVALID_ARGUMENT;
  context->in_flight = false;
  ++context->frame;
  return PS5HELIXSR_OK;
}

extern "C" uint32_t
ps5helixsr_debug_resources(ps5helixsr_context *context,
                           ps5helixsr_debug_resource *resources,
                           uint32_t capacity) {
  uint32_t count = 0;
  const auto add_buffer = [&](const char *name, const Buffer &buffer) {
    if (!buffer.buffer || count >= capacity)
      return;
    auto &out = resources[count++];
    out = {};
    std::snprintf(out.name, sizeof(out.name), "%s", name);
    out.buffer = buffer.buffer;
    out.bytes = buffer.bytes;
  };
  for (uint32_t i = 0; i < context->buffers.size(); ++i)
    add_buffer(context->tables->buffers[i].name, context->buffers[i]);
  add_buffer("UNIFORMS", context->uniforms);
  add_buffer("WEIGHTS", context->weights);
  add_buffer("TRANSIENT_A", context->transient_a);
  add_buffer("TRANSIENT_B", context->transient_b);
  for (uint32_t i = 0; i < context->images.size(); ++i) {
    const auto &image = context->images[i];
    if (!image.image || count >= capacity)
      continue;
    const auto &role = context->tables->images[i];
    auto &out = resources[count++];
    out = {};
    std::snprintf(out.name, sizeof(out.name), "%s", role.name);
    out.image = image.image;
    out.format = format(role.format);
    out.width = image.width;
    out.height = image.height;
    out.bytes = uint64_t(image.width) * image.height * format_bytes(role.format);
  }
  return count;
}

extern "C" uint32_t ps5helixsr_debug_steps(ps5helixsr_context *context,
                                           uint32_t reset,
                                           ps5helixsr_debug_step *steps,
                                           uint32_t capacity) {
  auto params = context->base;
  params.reset = reset || !context->initialized;
  params.hasPrev = context->initialized && !params.reset;
  const auto plan = ps5helixsr::prepare_runtime_plan(params, context->frame,
                                                     context->options);
  uint32_t count = 0;
  std::map<uint32_t, uint32_t> ordinals;
  for (const auto &step : plan.steps) {
    if (step.kind == StepKind::Barrier)
      continue;
    const std::string key = step.kind == StepKind::Dispatch
        ? step_key(step, ordinals[step.launch_index]++) : std::string();
    if (count < capacity) {
      auto &out = steps[count];
      out = {};
      const auto slot = context->slots.find(key);
      if (slot != context->slots.end()) {
        out.uniform_offset = slot->second.uniform_offset;
        out.uniform_bytes = step.uniform.size();
      }
      std::snprintf(out.name, sizeof(out.name), "%s",
                    step.kind == StepKind::FillBuffer ? "fill" : step.shader->name);
      out.launch = step.launch_index;
      std::memcpy(out.grid, step.grid, sizeof(out.grid));
    }
    ++count;
  }
  return count;
}

extern "C" void ps5helixsr_debug_set_range(ps5helixsr_context *context,
                                           uint32_t first, uint32_t count) {
  context->debug_first = first;
  context->debug_count = count;
}
