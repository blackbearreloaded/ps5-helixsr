#pragma once

#include "helixsr_assets.h"
#include "model.h"
#include <cstdint>
#include <string>
#include <vector>

namespace ps5helixsr {

enum class ResourceDomain : uint8_t {
  Uniform,
  Weights,
  Buffer,
  Image,
  TransientA,
  TransientB,
  Sampler
};

struct ResourceRef {
  ResourceDomain domain{};
  uint32_t index{};
  bool operator==(const ResourceRef &other) const {
    return domain == other.domain && index == other.index;
  }
};

struct Descriptor {
  char kind{};
  ResourceRef resource{};
  uint64_t offset{};
  uint64_t range{};
  bcm::Access access{bcm::AccessNone};
  int32_t linear_filter{-1};
};

enum class StepKind : uint8_t { Dispatch, FillBuffer, Barrier };

struct Step {
  StepKind kind{};
  const assets::Shader *shader{};
  std::vector<uint8_t> uniform;
  std::vector<Descriptor> descriptors;
  uint32_t grid[3]{};
  ResourceRef resource{};
  uint64_t offset{}, size{};
  uint32_t value{};
  std::vector<ResourceRef> barrier_resources;
  uint32_t launch_index{};
};

struct RuntimePlan {
  bcm::Params params{};
  std::vector<Step> steps;
  uint64_t transient_a_bytes{};
  uint64_t transient_b_bytes{};
};

RuntimePlan prepare_runtime_plan(bcm::Params params, uint32_t frame,
                                 const bcm::PlanOptions &options);

/* Weights of one early-stage convolution inside the network's weight blob. */
struct ConvolutionWeights {
  uint64_t offset{};
  uint32_t channels{}, outputs{}, kernel{};
};
/* The fast convolutions read their weights ordered by input channel:
   half index ((tap_group * 8 + c % 8) * outputs + o). Reorders the blob's
   spans in place (they are read by nothing else). */
void reorder_convolution_weights(const bcm::Params &params,
                                 const bcm::PlanOptions &options,
                                 std::vector<uint8_t> &blob);

} // namespace ps5helixsr
