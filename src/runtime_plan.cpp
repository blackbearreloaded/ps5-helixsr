#include "runtime_plan.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ps5helixsr {
namespace {

constexpr const char *kFused =
    "dltss_nchw8_conv_3x3_pool_conv_1x1_pool_512_032_008_fp16_e5m3_kernel";
constexpr const char *kStage64 =
    "dltss_nchw8_conv_3x3_pool_128_064_008_e5m3_fp16_kernel";
constexpr const char *kStage128 =
    "dltss_nchw8_conv_3x3_pool_128_064_008_fp16_fp16_kernel";
constexpr const char *kStage256 =
    "dltss_nchw8_conv_3x3_pool_064_064_008_fp16_fp16_kernel";
constexpr const char *kClear = "cuda_clear_buffer_kernel";

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

uint32_t load32(const std::vector<uint8_t> &data, size_t offset) {
  require(offset + 4 <= data.size(), "argument read exceeds block");
  uint32_t value;
  std::memcpy(&value, data.data() + offset, sizeof(value));
  return value;
}

void store64(std::vector<uint8_t> &data, size_t offset, uint64_t value) {
  require(offset + 8 <= data.size(), "argument write exceeds block");
  std::memcpy(data.data() + offset, &value, sizeof(value));
}

void append32(std::vector<uint8_t> &data, uint32_t value) {
  const auto old = data.size();
  data.resize(old + 4);
  std::memcpy(data.data() + old, &value, sizeof(value));
}

const bcm::LaunchBinding &binding(const bcm::Launch &launch, uint16_t offset) {
  const bcm::LaunchBinding *found = nullptr;
  for (const auto &item : launch.bindings)
    if (item.argOffset == offset) {
      require(!found, "duplicate compact binding");
      found = &item;
    }
  require(found, "missing compact binding");
  return *found;
}

uint64_t buffer_bytes(const bcm::NetTables &tables, const bcm::Params &params,
                      uint32_t role) {
  require(role < tables.bufferCount, "buffer role outside table");
  return tables.buffers[role].bytes(params);
}

Descriptor uniform_descriptor(size_t bytes) {
  return {'U', {ResourceDomain::Uniform, 0}, 0, bytes, bcm::AccessRead, -1};
}

Descriptor buffer_descriptor(ResourceRef resource, uint64_t range,
                             bcm::Access access) {
  return {'B', resource, 0, range, access, -1};
}

std::vector<uint8_t> words(std::initializer_list<uint32_t> values) {
  std::vector<uint8_t> result;
  result.reserve(values.size() * 4);
  for (auto value : values)
    append32(result, value);
  return result;
}

Step barrier(std::initializer_list<ResourceRef> resources, uint32_t launch) {
  Step result;
  result.kind = StepKind::Barrier;
  result.barrier_resources.assign(resources);
  result.launch_index = launch;
  return result;
}

/* The fast convolutions (tools/conv_kernels.py): output channels per invocation, and
   the weight offset each kernel carries as a literal. */
constexpr uint32_t kOutputsPerInvocation = 32;
uint32_t literal_weight_offset(uint32_t channels, uint32_t outputs) {
  if (channels == 16)
    return 0;
  if (channels == 32)
    return 9216;
  if (channels == 64)
    return 13824;
  return outputs == 128 ? 161536 : 456704;
}

Step compact_conv(const char *name, uint32_t width, uint32_t height,
                  ResourceRef input, uint64_t input_range,
                  uint32_t input_offset, uint64_t weight_range,
                  uint32_t weight_offset, uint32_t bias_offset,
                  ResourceRef pool, uint64_t pool_range, uint32_t pool_offset,
                  ResourceRef skip, uint64_t skip_range, uint32_t skip_offset,
                  uint32_t channels, uint32_t outputs, uint32_t kernel,
                  bool input_e5, uint32_t launch_index) {
  require(width >= 2 && height >= 2 && !(width & 1) && !(height & 1),
          "invalid compact dimensions");
  const uint64_t input_size = uint64_t(channels) * width * height * (input_e5 ? 1 : 2);
  const uint64_t weight_size = uint64_t(channels) * outputs * kernel * kernel * 2;
  const uint64_t output_bytes = uint64_t(outputs) * (width / 2) * (height / 2) * 2;
  require(uint64_t(input_offset) + input_size <= input_range,
          "compact input exceeds resource");
  require(uint64_t(weight_offset) + weight_size <= weight_range &&
              uint64_t(bias_offset) + outputs * 2 <= weight_range,
          "compact weights exceed resource");
  require(uint64_t(pool_offset) + output_bytes <= pool_range &&
              uint64_t(skip_offset) + output_bytes * 4 <= skip_range,
          "compact output exceeds resource");
  Step step;
  step.kind = StepKind::Dispatch;
  step.shader = assets::find(name);
  require(step.shader, "compact shader missing");
  step.uniform = words({width, height, input_offset, weight_offset, bias_offset,
                        pool_offset, skip_offset, 0});
  step.descriptors = {uniform_descriptor(step.uniform.size()),
                      buffer_descriptor(input, input_range, bcm::AccessRead),
                      buffer_descriptor({ResourceDomain::Weights, 0}, weight_range,
                                        bcm::AccessRead),
                      buffer_descriptor(pool, pool_range, bcm::AccessWrite),
                      buffer_descriptor(skip, skip_range, bcm::AccessWrite)};
  step.grid[0] = ((width / 2) * (height / 2) + 63) / 64;
  require(weight_offset == literal_weight_offset(channels, outputs),
          "convolution weights are not where the kernel reads them");
  step.grid[1] = outputs / kOutputsPerInvocation;
  step.grid[2] = 1;
  step.launch_index = launch_index;
  return step;
}

Step compact_encode(ResourceRef source, uint64_t source_range,
                    uint32_t source_offset, ResourceRef target,
                    uint64_t target_range, uint32_t target_offset,
                    uint32_t scalars, uint32_t launch_index) {
  require(scalars && !(scalars & 3) &&
              uint64_t(source_offset) + uint64_t(scalars) * 2 <= source_range &&
              uint64_t(target_offset) + scalars <= target_range,
          "compact encoder exceeds resource");
  Step step;
  step.kind = StepKind::Dispatch;
  step.shader = assets::find("encode_e5m3_resident");
  require(step.shader, "compact encoder missing");
  step.uniform = words({source_offset, target_offset, scalars, 0});
  step.descriptors = {uniform_descriptor(step.uniform.size()),
                      buffer_descriptor(source, source_range, bcm::AccessRead),
                      buffer_descriptor(target, target_range, bcm::AccessWrite)};
  step.grid[0] = uint32_t(((scalars / 4) + 63) / 64);
  step.grid[1] = step.grid[2] = 1;
  step.launch_index = launch_index;
  return step;
}

ResourceRef graph_buffer(const bcm::LaunchBinding &item) {
  require(item.kind == bcm::BindBuffer, "expected graph buffer");
  return {ResourceDomain::Buffer, item.role};
}

uint32_t checked_offset(const bcm::LaunchBinding &item, bcm::Access access) {
  require(item.kind == bcm::BindBuffer && item.access == access &&
              item.offset <= std::numeric_limits<uint32_t>::max() &&
              !(item.offset & 3),
          "invalid compact buffer");
  return uint32_t(item.offset);
}

uint32_t checked_weight(const bcm::LaunchBinding &item, uint64_t bytes,
                        uint64_t total) {
  require(item.kind == bcm::BindWeight && item.access == bcm::AccessRead &&
              item.weightSize == bytes && item.offset + bytes <= total &&
              !(item.offset & 3),
          "invalid compact weight");
  return uint32_t(item.offset);
}

void lower_compact(RuntimePlan &out, const bcm::Launch &launch,
                   uint32_t launch_index, const bcm::NetTables &tables) {
  const uint32_t wp = out.params.Wp, hp = out.params.Hp;
  const uint64_t weight_bytes = tables.weightBlobBytes;
  if (!std::strcmp(launch.name, kFused)) {
    require(launch.block[0] == 256 && launch.block[1] == 1 && launch.block[2] == 1,
            "unexpected fused workgroup");
    const auto &src = binding(launch, 0), &w1b = binding(launch, 8),
               &b1b = binding(launch, 88), &w2b = binding(launch, 40),
               &b2b = binding(launch, 176), &out1 = binding(launch, 80),
               &pool2 = binding(launch, 160), &skip2 = binding(launch, 168);
    const auto src_off = checked_offset(src, bcm::AccessRead);
    const auto out1_off = checked_offset(out1, bcm::AccessWrite);
    const auto pool2_off = checked_offset(pool2, bcm::AccessWrite);
    const auto skip2_off = checked_offset(skip2, bcm::AccessWrite);
    require(pool2.role == skip2.role, "fused outputs must share a buffer");
    const auto w1 = checked_weight(w1b, 16u * 32 * 3 * 3 * 2, weight_bytes);
    const auto b1 = checked_weight(b1b, 32u * 2, weight_bytes);
    const auto w2 = checked_weight(w2b, 32u * 64 * 2, weight_bytes);
    const auto b2 = checked_weight(b2b, 64u * 2, weight_bytes);
    const uint64_t pixels = uint64_t(wp) * hp;
    out.transient_a_bytes = 5 * pixels;
    out.transient_b_bytes = 5 * pixels / 2;
    const ResourceRef a{ResourceDomain::TransientA, 0};
    const ResourceRef b{ResourceDomain::TransientB, 0};
    out.steps.push_back(compact_conv(
        "conv_16_32_k3_half", wp / 4, hp / 4, graph_buffer(src),
        buffer_bytes(tables, out.params, src.role), src_off, weight_bytes, w1,
        b1, a, out.transient_a_bytes, 0, a, out.transient_a_bytes,
        uint32_t(pixels), 16, 32, 3, false, launch_index));
    out.steps.push_back(barrier({a}, launch_index));
    out.steps.push_back(compact_encode(
        a, out.transient_a_bytes, uint32_t(pixels), graph_buffer(out1),
        buffer_bytes(tables, out.params, out1.role), out1_off,
        uint32_t(2 * pixels), launch_index));
    out.steps.push_back(compact_conv(
        "conv_32_64_k1_half", wp / 8, hp / 8, a, out.transient_a_bytes, 0,
        weight_bytes, w2, b2, b, out.transient_b_bytes, 0, b,
        out.transient_b_bytes, uint32_t(pixels / 2), 32, 64, 1, false,
        launch_index));
    out.steps.push_back(barrier({b}, launch_index));
    out.steps.push_back(compact_encode(
        b, out.transient_b_bytes, 0, graph_buffer(pool2),
        buffer_bytes(tables, out.params, pool2.role), pool2_off,
        uint32_t(pixels / 4), launch_index));
    out.steps.push_back(compact_encode(
        b, out.transient_b_bytes, uint32_t(pixels / 2), graph_buffer(skip2),
        buffer_bytes(tables, out.params, skip2.role), skip2_off,
        uint32_t(pixels), launch_index));
    out.steps.push_back(
        barrier({graph_buffer(out1), graph_buffer(pool2)}, launch_index));
    return;
  }

  const char *shader = nullptr;
  uint32_t channels = 0, outputs = 0, divisor = 0;
  bool e5 = false;
  if (!std::strcmp(launch.name, kStage64)) {
    shader = "conv_64_128_k3_e5"; channels = 64; outputs = 128; divisor = 16; e5 = true;
  } else if (!std::strcmp(launch.name, kStage128)) {
    shader = "conv_128_128_k3_half"; channels = outputs = 128; divisor = 32;
  } else if (!std::strcmp(launch.name, kStage256)) {
    shader = "conv_128_256_k3_half"; channels = 128; outputs = 256; divisor = 64;
  } else {
    throw std::runtime_error("unknown compact stage");
  }
  require(launch.block[0] == 128 && launch.block[1] == 1 && launch.block[2] == 1,
          "unexpected standalone workgroup");
  const auto &src = binding(launch, 0), &weight = binding(launch, 8),
             &pool = binding(launch, 24), &skip = binding(launch, 40),
             &bias = binding(launch, 56), &duplicate = binding(launch, 264);
  require(duplicate.kind == bcm::BindWeight && duplicate.offset == weight.offset &&
              duplicate.weightSize == weight.weightSize,
          "standalone duplicate weight differs");
  const auto src_off = checked_offset(src, bcm::AccessRead);
  const auto pool_off = checked_offset(pool, bcm::AccessWrite);
  const auto skip_off = checked_offset(skip, bcm::AccessWrite);
  const auto weight_off = checked_weight(
      weight, uint64_t(channels) * outputs * 3 * 3 * 2, weight_bytes);
  const auto bias_off = checked_weight(bias, outputs * 2, weight_bytes);
  out.steps.push_back(compact_conv(
      shader, wp / divisor, hp / divisor, graph_buffer(src),
      buffer_bytes(tables, out.params, src.role), src_off, weight_bytes,
      weight_off, bias_off, graph_buffer(pool),
      buffer_bytes(tables, out.params, pool.role), pool_off,
      graph_buffer(skip), buffer_bytes(tables, out.params, skip.role), skip_off,
      channels, outputs, 3, e5, launch_index));
  out.steps.push_back(
      barrier({graph_buffer(pool), graph_buffer(skip)}, launch_index));
}

Step generated(const bcm::Launch &launch, uint32_t launch_index,
               const bcm::Params &params, const bcm::NetTables &tables) {
  Step step;
  step.kind = StepKind::Dispatch;
  step.shader = assets::recipe_for_kernel(launch.name);
  require(step.shader, "generated recipe missing");
  require(step.shader->descriptor_count && step.shader->descriptors[0].kind == 'U',
          "generated uniform descriptor missing");
  step.uniform = launch.args;
  std::vector<bool> mapped(launch.bindings.size());
  step.descriptors.push_back(uniform_descriptor(0));
  for (size_t i = 1; i < step.shader->descriptor_count; ++i) {
    const auto &abi = step.shader->descriptors[i];
    if (abi.kind == 'S') {
      require(abi.linear_filter == 0 || abi.linear_filter == 1,
              "invalid sampler ABI");
      step.descriptors.push_back(
          {'S', {ResourceDomain::Sampler, uint32_t(abi.linear_filter)}, 0, 0,
           bcm::AccessRead, abi.linear_filter});
      continue;
    }
    const bcm::LaunchBinding *item = nullptr;
    size_t item_index = 0;
    for (size_t j = 0; j < launch.bindings.size(); ++j)
      if (launch.bindings[j].argOffset == abi.parameter_offset) {
        require(!item, "duplicate generated parameter");
        item = &launch.bindings[j]; item_index = j;
      }
    require(item && !mapped[item_index], "generated parameter missing");
    mapped[item_index] = true;
    if (abi.kind == 'B') {
      require(item->kind == bcm::BindBuffer || item->kind == bcm::BindWeight,
              "generated buffer kind mismatch");
      uint64_t range;
      ResourceRef resource;
      if (item->kind == bcm::BindWeight) {
        require(item->access == bcm::AccessRead && item->weightSize &&
                    item->offset + item->weightSize <= tables.weightBlobBytes,
                "generated weight span invalid");
        range = tables.weightBlobBytes;
        resource = {ResourceDomain::Weights, 0};
      } else {
        range = buffer_bytes(tables, params, item->role);
        resource = {ResourceDomain::Buffer, item->role};
      }
      require(abi.resource_slot >= 0 && item->offset < range && range <= UINT32_MAX,
              "generated buffer range invalid");
      store64(step.uniform, item->argOffset,
              (uint64_t(abi.resource_slot + 1) << 40) | item->offset);
      step.descriptors.push_back(
          buffer_descriptor(resource, range, item->access));
    } else {
      require((abi.kind == 'I' || abi.kind == 'W') &&
                  (item->kind == bcm::BindTexture || item->kind == bcm::BindSurface) &&
                  item->role < tables.imageCount && item->offset == 0,
              "generated image kind mismatch");
      step.descriptors.push_back(
          {abi.kind, {ResourceDomain::Image, item->role}, 0, 0, item->access, -1});
    }
  }
  for (size_t i = 0; i < launch.bindings.size(); ++i)
    if (!mapped[i])
      store64(step.uniform, launch.bindings[i].argOffset, 0);
  step.uniform.resize((step.uniform.size() + 15) & ~size_t(15));
  for (auto value : launch.grid)
    append32(step.uniform, value);
  append32(step.uniform, 0);
  step.descriptors[0].range = step.uniform.size();
  std::copy(std::begin(launch.grid), std::end(launch.grid), step.grid);
  step.launch_index = launch_index;
  return step;
}

} // namespace

RuntimePlan prepare_runtime_plan(bcm::Params params, uint32_t frame,
                                 const bcm::PlanOptions &options) {
  bcm::finalize(params);
  require(params.net == bcm::NetNCHW8, "only NCHW8 model is packaged");
  RuntimePlan result;
  result.params = params;
  const auto &tables = bcm::net_tables(params.net);
  const auto launches = bcm::plan(params, frame, options);
  for (uint32_t index = 0; index < launches.size(); ++index) {
    const auto &launch = launches[index];
    if (!std::strcmp(launch.name, kClear)) {
      require(launch.args.size() == 40 && launch.bindings.size() == 1,
              "unexpected clear ABI");
      const auto &item = launch.bindings[0];
      const auto bytes = buffer_bytes(tables, params, item.role);
      require(item.kind == bcm::BindBuffer && item.access == bcm::AccessWrite &&
                  item.argOffset == 32 && item.offset == 0 && load32(launch.args, 8) == 0 &&
                  load32(launch.args, 12) == 0 && load32(launch.args, 16) == load32(launch.args, 0) &&
                  load32(launch.args, 20) == load32(launch.args, 4) &&
                  load32(launch.args, 24) == 0 && load32(launch.args, 28) == 1 &&
                  uint64_t(load32(launch.args, 0)) * load32(launch.args, 4) == bytes && !(bytes & 3),
              "unsupported clear operation");
      Step step;
      step.kind = StepKind::FillBuffer;
      step.resource = {ResourceDomain::Buffer, item.role};
      step.size = bytes;
      step.launch_index = index;
      result.steps.push_back(std::move(step));
    } else if (!std::strcmp(launch.name, kFused) ||
               !std::strcmp(launch.name, kStage64) ||
               !std::strcmp(launch.name, kStage128) ||
               !std::strcmp(launch.name, kStage256)) {
      lower_compact(result, launch, index, tables);
    } else {
      result.steps.push_back(generated(launch, index, params, tables));
    }
  }
  require(!result.steps.empty(), "runtime plan is empty");
  return result;
}

void reorder_convolution_weights(const bcm::Params &base,
                                 const bcm::PlanOptions &options,
                                 std::vector<uint8_t> &blob) {
  bcm::Params params = base;
  bcm::finalize(params);
  params.reset = 1;
  std::vector<ConvolutionWeights> spans;
  for (const auto &launch : bcm::plan(params, 0, options)) {
    if (!std::strcmp(launch.name, kFused)) {
      spans.push_back({binding(launch, 8).offset, 16, 32, 3});
      spans.push_back({binding(launch, 40).offset, 32, 64, 1});
    } else if (!std::strcmp(launch.name, kStage64)) {
      spans.push_back({binding(launch, 8).offset, 64, 128, 3});
    } else if (!std::strcmp(launch.name, kStage128)) {
      spans.push_back({binding(launch, 8).offset, 128, 128, 3});
    } else if (!std::strcmp(launch.name, kStage256)) {
      spans.push_back({binding(launch, 8).offset, 128, 256, 3});
    }
  }
  require(spans.size() == 5, "early-stage convolutions missing from the plan");
  std::vector<uint8_t> scratch;
  for (const auto &span : spans) {
    const size_t groups = size_t(span.channels / 8) * span.kernel * span.kernel;
    const size_t block = size_t(span.outputs) * 8 * 2; /* bytes per tap group */
    require(span.offset + groups * block <= blob.size(),
            "convolution weights outside the blob");
    scratch.resize(block);
    for (size_t t = 0; t < groups; ++t) {
      uint8_t *at = blob.data() + span.offset + t * block;
      /* [output][c % 8] -> [c % 8][output] */
      for (uint32_t o = 0; o < span.outputs; ++o)
        for (uint32_t i = 0; i < 8; ++i)
          std::memcpy(scratch.data() + (size_t(i) * span.outputs + o) * 2,
                      at + (size_t(o) * 8 + i) * 2, 2);
      std::memcpy(at, scratch.data(), block);
    }
  }
}

} // namespace ps5helixsr
