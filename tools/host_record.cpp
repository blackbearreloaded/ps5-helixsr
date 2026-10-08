// Create every Vulkan object and encode one HelixSR frame without submitting
// it.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

static void check(VkResult result, const char *call) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(call) + ": " + std::to_string(result));
}
static std::vector<uint8_t> bytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  auto n = f.tellg();
  if (!f || n <= 0 || n > 16 * 1024 * 1024 || n % 4)
    throw std::runtime_error("invalid SPIR-V: " + path);
  std::vector<uint8_t> result((size_t)n);
  f.seekg(0);
  if (!f.read((char *)result.data(), n))
    throw std::runtime_error("SPIR-V read failed");
  if (result.size() < 20 ||
      *reinterpret_cast<const uint32_t *>(result.data()) != 0x07230203)
    throw std::runtime_error("bad SPIR-V header");
  return result;
}
static std::vector<uint8_t> unhex(const std::string &text) {
  if (text.size() % 2)
    throw std::runtime_error("odd hex payload");
  std::vector<uint8_t> out(text.size() / 2);
  auto digit = [](char c) -> unsigned {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    throw std::runtime_error("bad hex");
  };
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = (uint8_t)((digit(text[2 * i]) << 4) | digit(text[2 * i + 1]));
  return out;
}
static VkFormat format(const std::string &name) {
  if (name == "r16f")
    return VK_FORMAT_R16_SFLOAT;
  if (name == "rg16f")
    return VK_FORMAT_R16G16_SFLOAT;
  if (name == "rgba16f")
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  if (name == "r32f")
    return VK_FORMAT_R32_SFLOAT;
  throw std::runtime_error("unknown format");
}
static VkDescriptorType descriptor(char kind) {
  if (kind == 'U')
    return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  if (kind == 'B')
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  if (kind == 'I')
    return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  if (kind == 'W')
    return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  if (kind == 'S')
    return VK_DESCRIPTOR_TYPE_SAMPLER;
  throw std::runtime_error("unknown descriptor kind");
}
struct ResourceSpec {
  char kind{};
  uint64_t bytes{};
  uint32_t usage{}, width{}, height{};
  VkFormat format{};
  std::string owner;
};
struct PipelineSpec {
  std::string name, path, kinds;
};
struct DescriptorSpec {
  uint32_t binding{}, resource{};
  char kind{};
  uint64_t offset{}, range{};
  std::string filter;
};
struct SetSpec {
  uint32_t pipeline{};
  std::vector<DescriptorSpec> descriptors;
};
struct Command {
  char kind{};
  std::vector<uint64_t> values;
};
struct Fixture {
  std::vector<ResourceSpec> resources;
  std::vector<PipelineSpec> pipelines;
  std::map<uint64_t, std::vector<uint8_t>> uniforms;
  std::vector<SetSpec> sets;
  std::vector<Command> commands;
};
static Fixture fixture(const char *path) {
  std::ifstream f(path);
  std::string magic;
  if (!(f >> magic) || magic != "HELIXSR_RECORD_V1")
    throw std::runtime_error("invalid fixture");
  Fixture x;
  std::string tag;
  while (f >> tag) {
    if (tag == "E")
      break;
    if (tag == "R") {
      uint32_t id;
      ResourceSpec r;
      f >> id >> r.kind;
      if (id != x.resources.size())
        throw std::runtime_error("non-dense resources");
      if (r.kind == 'B')
        f >> r.bytes >> r.usage >> r.owner;
      else {
        std::string fmt;
        f >> r.width >> r.height >> fmt >> r.owner;
        r.format = format(fmt);
      }
      x.resources.push_back(r);
    } else if (tag == "P") {
      uint32_t id, count;
      PipelineSpec p;
      f >> id >> p.name >> p.path >> count >> p.kinds;
      if (id != x.pipelines.size() || p.kinds.size() != count)
        throw std::runtime_error("invalid pipeline");
      x.pipelines.push_back(p);
    } else if (tag == "U") {
      uint64_t offset;
      std::string data;
      f >> offset >> data;
      if (!x.uniforms.emplace(offset, unhex(data)).second)
        throw std::runtime_error("duplicate uniform");
    } else if (tag == "S") {
      uint32_t id, count;
      SetSpec set;
      f >> id >> set.pipeline >> count;
      if (id != x.sets.size() || set.pipeline >= x.pipelines.size())
        throw std::runtime_error("invalid set");
      for (uint32_t i = 0; i < count; ++i) {
        std::string d;
        uint32_t sid;
        DescriptorSpec spec;
        f >> d >> sid >> spec.binding >> spec.kind;
        if (d != "D" || sid != id)
          throw std::runtime_error("invalid descriptor record");
        if (spec.kind == 'S')
          f >> spec.filter;
        else if (spec.kind == 'I' || spec.kind == 'W')
          f >> spec.resource;
        else
          f >> spec.resource >> spec.offset >> spec.range;
        set.descriptors.push_back(spec);
      }
      x.sets.push_back(std::move(set));
    } else if (tag == "C") {
      Command c;
      f >> c.kind;
      uint64_t count = 0;
      if (c.kind == 'T') {
        uint64_t src, dst, srcAccess, dstAccess;
        f >> src >> dst >> srcAccess >> dstAccess >> count;
        c.values = {src, dst, srcAccess, dstAccess, count};
        for (uint64_t i = 0, v; i < count; ++i) {
          f >> v;
          c.values.push_back(v);
        }
      } else if (c.kind == 'F') {
        c.values.resize(4);
        for (auto &v : c.values)
          f >> v;
      } else if (c.kind == 'C') {
        c.values.resize(4);
        for (auto &v : c.values)
          f >> v;
      } else if (c.kind == 'B') {
        f >> count;
        c.values = {count};
        for (uint64_t i = 0, v; i < count * 5; ++i) {
          f >> v;
          c.values.push_back(v);
        }
      } else if (c.kind == 'D') {
        c.values.resize(4);
        for (auto &v : c.values)
          f >> v;
      } else
        throw std::runtime_error("unknown command record");
      x.commands.push_back(std::move(c));
    } else
      throw std::runtime_error("unknown fixture record");
  }
  if (x.resources.empty() || x.pipelines.empty() || x.sets.empty() ||
      x.commands.empty())
    throw std::runtime_error("empty fixture");
  return x;
}
struct Resource {
  VkBuffer buffer{};
  VkImage image{};
  VkImageView view{};
  VkDeviceMemory memory{};
  void *mapped{};
};
struct Pipeline {
  VkDescriptorSetLayout setLayout{};
  VkPipelineLayout layout{};
  VkShaderModule module{};
  VkPipeline handle{};
};
struct Context {
  VkInstance instance{};
  VkDevice device{};
  VkDescriptorPool descriptorPool{};
  VkCommandPool commandPool{};
  VkSampler point{}, linear{};
  std::vector<Resource> resources;
  std::vector<Pipeline> pipelines;
  std::vector<VkDescriptorSet> sets;
  ~Context() {
    if (device) {
      vkDeviceWaitIdle(device);
      if (commandPool)
        vkDestroyCommandPool(device, commandPool, nullptr);
      if (descriptorPool)
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
      if (point)
        vkDestroySampler(device, point, nullptr);
      if (linear)
        vkDestroySampler(device, linear, nullptr);
      for (auto &p : pipelines) {
        if (p.handle)
          vkDestroyPipeline(device, p.handle, nullptr);
        if (p.module)
          vkDestroyShaderModule(device, p.module, nullptr);
        if (p.layout)
          vkDestroyPipelineLayout(device, p.layout, nullptr);
        if (p.setLayout)
          vkDestroyDescriptorSetLayout(device, p.setLayout, nullptr);
      }
      for (auto &r : resources) {
        if (r.mapped)
          vkUnmapMemory(device, r.memory);
        if (r.view)
          vkDestroyImageView(device, r.view, nullptr);
        if (r.image)
          vkDestroyImage(device, r.image, nullptr);
        if (r.buffer)
          vkDestroyBuffer(device, r.buffer, nullptr);
        if (r.memory)
          vkFreeMemory(device, r.memory, nullptr);
      }
      vkDestroyDevice(device, nullptr);
    }
    if (instance)
      vkDestroyInstance(instance, nullptr);
  }
};
static uint32_t memory_type(const VkPhysicalDeviceMemoryProperties &memory,
                            uint32_t bits, VkMemoryPropertyFlags wanted) {
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
    if ((bits & (1u << i)) &&
        (memory.memoryTypes[i].propertyFlags & wanted) == wanted)
      return i;
  throw std::runtime_error("required memory type unavailable");
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: helixsr_host_record fixture.txt");
    Fixture in = fixture(argv[1]);
    Context c;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    check(vkCreateInstance(&ici, nullptr, &c.instance), "vkCreateInstance");
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(c.instance, &count, nullptr),
          "device count");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(c.instance, &count, devices.data()),
          "devices");
    VkPhysicalDevice physical{};
    uint32_t family = 0;
    for (auto d : devices) {
      uint32_t n = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(d, &n, nullptr);
      std::vector<VkQueueFamilyProperties> q(n);
      vkGetPhysicalDeviceQueueFamilyProperties(d, &n, q.data());
      for (uint32_t i = 0; i < n; ++i)
        if (q[i].queueCount && (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          physical = d;
          family = i;
          break;
        }
      if (physical)
        break;
    }
    if (!physical)
      throw std::runtime_error("no compute device");
    uint32_t extCount = 0;
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extCount,
                                               nullptr),
          "extension count");
    std::vector<VkExtensionProperties> exts(extCount);
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extCount,
                                               exts.data()),
          "extensions");
    bool derivatives = false;
    for (const auto &e : exts)
      derivatives |= !std::strcmp(
          e.extensionName, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
    if (!derivatives)
      throw std::runtime_error("compute derivatives extension unavailable");
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features f12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.pNext = &derivative;
    VkPhysicalDeviceVulkan11Features f11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(physical, &features);
    if (!features.features.shaderInt16 || !features.features.shaderInt64 ||
        !f12.shaderFloat16 || !f11.storageBuffer16BitAccess ||
        !derivative.computeDerivativeGroupLinear)
      throw std::runtime_error(
          "required generated-shader features unavailable");
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR ed{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    ed.computeDerivativeGroupLinear = VK_TRUE;
    VkPhysicalDeviceVulkan12Features e12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    e12.shaderFloat16 = VK_TRUE;
    e12.shaderInt8 = f12.shaderInt8;
    e12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
    e12.pNext = &ed;
    VkPhysicalDeviceVulkan11Features e11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    e11.storageBuffer16BitAccess = VK_TRUE;
    e11.pNext = &e12;
    VkPhysicalDeviceFeatures enabled{};
    enabled.shaderInt16 = VK_TRUE;
    enabled.shaderInt64 = VK_TRUE;
    float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    const char *extension = VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &e11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &enabled;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &extension;
    check(vkCreateDevice(physical, &dci, nullptr, &c.device), "vkCreateDevice");
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    c.resources.resize(in.resources.size());
    for (size_t i = 0; i < in.resources.size(); ++i) {
      const auto &s = in.resources[i];
      auto &r = c.resources[i];
      if (s.kind == 'B') {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = s.bytes;
        bi.usage = (s.usage & 1 ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0) |
                   (s.usage & 2 ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : 0) |
                   (s.usage & 4 ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0) |
                   (s.usage & 8 ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0);
        check(vkCreateBuffer(c.device, &bi, nullptr, &r.buffer),
              "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(c.device, r.buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            memory_type(memory, req.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(c.device, &ai, nullptr, &r.memory),
              "vkAllocateMemory(buffer)");
        check(vkBindBufferMemory(c.device, r.buffer, r.memory, 0),
              "vkBindBufferMemory");
        check(vkMapMemory(c.device, r.memory, 0, VK_WHOLE_SIZE, 0, &r.mapped),
              "vkMapMemory");
        std::memset(r.mapped, 0, (size_t)s.bytes);
      } else {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = s.format;
        ii.extent = {s.width, s.height, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(c.device, &ii, nullptr, &r.image), "vkCreateImage");
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(c.device, r.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        try {
          ai.memoryTypeIndex = memory_type(memory, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        } catch (...) {
          ai.memoryTypeIndex = memory_type(memory, req.memoryTypeBits, 0);
        }
        check(vkAllocateMemory(c.device, &ai, nullptr, &r.memory),
              "vkAllocateMemory(image)");
        check(vkBindImageMemory(c.device, r.image, r.memory, 0),
              "vkBindImageMemory");
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = r.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = s.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(c.device, &vi, nullptr, &r.view),
              "vkCreateImageView");
      }
    }
    uint32_t uniformResource = in.resources.size();
    for (uint32_t i = 0; i < in.resources.size(); ++i)
      if (in.resources[i].kind == 'B' && (in.resources[i].usage & 2)) {
        uniformResource = i;
        break;
      }
    if (uniformResource == in.resources.size())
      throw std::runtime_error("uniform resource missing");
    for (const auto &item : in.uniforms) {
      if (item.first + item.second.size() > in.resources[uniformResource].bytes)
        throw std::runtime_error("uniform payload out of range");
      std::memcpy((uint8_t *)c.resources[uniformResource].mapped + item.first,
                  item.second.data(), item.second.size());
    }
    auto makeSampler = [&](VkFilter filter, VkSampler *out) {
      VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      si.magFilter = si.minFilter = filter;
      si.mipmapMode = filter == VK_FILTER_LINEAR
                          ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                          : VK_SAMPLER_MIPMAP_MODE_NEAREST;
      si.addressModeU = si.addressModeV = si.addressModeW =
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      check(vkCreateSampler(c.device, &si, nullptr, out), "vkCreateSampler");
    };
    makeSampler(VK_FILTER_NEAREST, &c.point);
    makeSampler(VK_FILTER_LINEAR, &c.linear);
    c.pipelines.resize(in.pipelines.size());
    for (size_t i = 0; i < in.pipelines.size(); ++i) {
      const auto &s = in.pipelines[i];
      auto &p = c.pipelines[i];
      std::vector<VkDescriptorSetLayoutBinding> bindings(s.kinds.size());
      for (uint32_t j = 0; j < bindings.size(); ++j)
        bindings[j] = {j, descriptor(s.kinds[j]), 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      VkDescriptorSetLayoutCreateInfo li{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      li.bindingCount = bindings.size();
      li.pBindings = bindings.data();
      check(vkCreateDescriptorSetLayout(c.device, &li, nullptr, &p.setLayout),
            "vkCreateDescriptorSetLayout");
      VkPipelineLayoutCreateInfo pli{
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pli.setLayoutCount = 1;
      pli.pSetLayouts = &p.setLayout;
      check(vkCreatePipelineLayout(c.device, &pli, nullptr, &p.layout),
            "vkCreatePipelineLayout");
      auto code = bytes(s.path);
      VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      mi.codeSize = code.size();
      mi.pCode = (const uint32_t *)code.data();
      check(vkCreateShaderModule(c.device, &mi, nullptr, &p.module),
            "vkCreateShaderModule");
      VkComputePipelineCreateInfo pi{
          VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pi.layout = p.layout;
      pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pi.stage.module = p.module;
      pi.stage.pName = "main";
      check(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &pi, nullptr,
                                     &p.handle),
            "vkCreateComputePipelines");
    }
    std::map<VkDescriptorType, uint32_t> descriptorCounts;
    for (const auto &s : in.sets)
      for (const auto &d : s.descriptors)
        descriptorCounts[descriptor(d.kind)]++;
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto [type, n] : descriptorCounts)
      poolSizes.push_back({type, n});
    VkDescriptorPoolCreateInfo dpi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = in.sets.size();
    dpi.poolSizeCount = poolSizes.size();
    dpi.pPoolSizes = poolSizes.data();
    check(vkCreateDescriptorPool(c.device, &dpi, nullptr, &c.descriptorPool),
          "vkCreateDescriptorPool");
    c.sets.resize(in.sets.size());
    for (size_t i = 0; i < in.sets.size(); ++i) {
      auto layout = c.pipelines[in.sets[i].pipeline].setLayout;
      VkDescriptorSetAllocateInfo ai{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      ai.descriptorPool = c.descriptorPool;
      ai.descriptorSetCount = 1;
      ai.pSetLayouts = &layout;
      check(vkAllocateDescriptorSets(c.device, &ai, &c.sets[i]),
            "vkAllocateDescriptorSets");
      for (const auto &d : in.sets[i].descriptors) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = c.sets[i];
        w.dstBinding = d.binding;
        w.descriptorCount = 1;
        w.descriptorType = descriptor(d.kind);
        VkDescriptorBufferInfo bi{};
        VkDescriptorImageInfo ii{};
        if (d.kind == 'S')
          ii.sampler = d.filter == "linear" ? c.linear : c.point;
        else if (d.kind == 'I' || d.kind == 'W') {
          if (d.resource >= c.resources.size())
            throw std::runtime_error("descriptor image missing");
          ii.imageView = c.resources[d.resource].view;
          ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        } else {
          if (d.resource >= c.resources.size() ||
              d.offset + d.range > in.resources[d.resource].bytes)
            throw std::runtime_error("descriptor buffer range invalid");
          bi = {c.resources[d.resource].buffer, d.offset, d.range};
        }
        if (d.kind == 'U' || d.kind == 'B')
          w.pBufferInfo = &bi;
        else
          w.pImageInfo = &ii;
        vkUpdateDescriptorSets(c.device, 1, &w, 0, nullptr);
      }
    }
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = family;
    check(vkCreateCommandPool(c.device, &cpi, nullptr, &c.commandPool),
          "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = c.commandPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd{};
    check(vkAllocateCommandBuffers(c.device, &cai, &cmd),
          "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
    std::vector<VkImageMemoryBarrier> external;
    for (uint32_t i = 0; i < in.resources.size(); ++i)
      if (in.resources[i].kind == 'I' &&
          in.resources[i].owner == "application") {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = c.resources[i].image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        external.push_back(b);
      }
    if (!external.empty())
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, external.size(), external.data());
    uint32_t dispatches = 0, barriers = 0;
    for (const auto &op : in.commands) {
      if (op.kind == 'T') {
        uint32_t n = (uint32_t)op.values[4];
        std::vector<VkImageMemoryBarrier> list(n);
        for (uint32_t i = 0; i < n; ++i) {
          uint32_t id = op.values[5 + i];
          auto &b = list[i];
          b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
          b.srcAccessMask = op.values[2];
          b.dstAccessMask = op.values[3];
          b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
          b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
          b.srcQueueFamilyIndex = b.dstQueueFamilyIndex =
              VK_QUEUE_FAMILY_IGNORED;
          b.image = c.resources[id].image;
          b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        vkCmdPipelineBarrier(cmd, op.values[0], op.values[1], 0, 0, nullptr, 0,
                             nullptr, list.size(), list.data());
        barriers++;
      } else if (op.kind == 'F')
        vkCmdFillBuffer(cmd, c.resources[op.values[0]].buffer, op.values[1],
                        op.values[2], (uint32_t)op.values[3]);
      else if (op.kind == 'C') {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {(uint32_t)op.values[2], (uint32_t)op.values[3],
                              1};
        vkCmdCopyBufferToImage(cmd, c.resources[op.values[0]].buffer,
                               c.resources[op.values[1]].image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &region);
      } else if (op.kind == 'B') {
        uint32_t n = op.values[0];
        std::vector<VkBufferMemoryBarrier> buffers;
        std::vector<VkImageMemoryBarrier> images;
        VkPipelineStageFlags src = 0, dst = 0;
        for (uint32_t i = 0; i < n; ++i) {
          uint64_t at = 1 + i * 5, id = op.values[at];
          src |= op.values[at + 1];
          dst |= op.values[at + 3];
          if (in.resources[id].kind == 'B') {
            VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            b.srcAccessMask = op.values[at + 2];
            b.dstAccessMask = op.values[at + 4];
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            b.buffer = c.resources[id].buffer;
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            buffers.push_back(b);
          } else {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = op.values[at + 2];
            b.dstAccessMask = op.values[at + 4];
            b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            b.image = c.resources[id].image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            images.push_back(b);
          }
        }
        vkCmdPipelineBarrier(cmd, src, dst, 0, 0, nullptr, buffers.size(),
                             buffers.data(), images.size(), images.data());
        barriers++;
      } else if (op.kind == 'D') {
        uint32_t set = op.values[0], pipeline = in.sets[set].pipeline;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          c.pipelines[pipeline].handle);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                c.pipelines[pipeline].layout, 0, 1,
                                &c.sets[set], 0, nullptr);
        vkCmdDispatch(cmd, op.values[1], op.values[2], op.values[3]);
        dispatches++;
      }
    }
    check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    std::cout << "{\"result\":\"success\",\"device\":\""
              << properties.deviceName
              << "\",\"resources\":" << in.resources.size()
              << ",\"pipelines\":" << in.pipelines.size()
              << ",\"descriptor_sets\":" << in.sets.size()
              << ",\"dispatches\":" << dispatches
              << ",\"barriers\":" << barriers
              << ",\"submitted\":false,\"execution_qualified\":false}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
