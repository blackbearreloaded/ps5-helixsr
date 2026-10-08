#include "runtime_plan.hpp"

#include <charconv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace ps5helixsr;

static uint32_t number(const char *text) {
  uint32_t result = 0;
  const auto end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end)
    throw std::runtime_error("invalid number");
  return result;
}

static double real(const char *text) {
  double result = 0;
  const auto end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, result);
  if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(result))
    throw std::runtime_error("invalid real number");
  return result;
}

static std::string resource(ResourceRef value) {
  switch (value.domain) {
  case ResourceDomain::Uniform: return "uniforms";
  case ResourceDomain::Weights: return "weights";
  case ResourceDomain::Buffer: return "buffer:" + std::to_string(value.index);
  case ResourceDomain::Image: return "image:" + std::to_string(value.index);
  case ResourceDomain::TransientA: return "transient:fused_fp16_a";
  case ResourceDomain::TransientB: return "transient:fused_fp16_b";
  case ResourceDomain::Sampler: return value.index ? "linear" : "point";
  }
  throw std::runtime_error("invalid resource");
}

static void hex(const std::vector<uint8_t> &data) {
  std::cout << std::hex << std::setfill('0');
  for (auto value : data)
    std::cout << std::setw(2) << unsigned(value);
  std::cout << std::dec;
}

int main(int argc, char **argv) {
  try {
    if (argc != 7 && argc != 14)
      throw std::runtime_error("usage: runtime_plan width height render_width render_height frame auto_exposure [jx jy pjx pjy pre_exposure reset has_previous]");
    bcm::Params params;
    params.Wo = number(argv[1]); params.Ho = number(argv[2]);
    params.Wr = number(argv[3]); params.Hr = number(argv[4]);
    const auto frame = number(argv[5]);
    params.hasPrev = frame != 0; params.reset = frame == 0;
    if (argc == 14) {
      params.jx = float(real(argv[7])); params.jy = float(real(argv[8]));
      params.pjx = float(real(argv[9])); params.pjy = float(real(argv[10]));
      params.pre = real(argv[11]); params.reset = number(argv[12]);
      params.hasPrev = number(argv[13]) != 0;
      if (!(params.pre > 0) || params.reset > 1 || number(argv[13]) > 1)
        throw std::runtime_error("invalid dynamic frame parameters");
    }
    bcm::PlanOptions options;
    options.autoExposure = number(argv[6]) != 0;
    options.displayResMv = false;
    const auto plan = prepare_runtime_plan(params, frame, options);
    std::cout << "HELIXSR_RUNTIME_PLAN_V1\n";
    std::cout << "T " << plan.transient_a_bytes << ' ' << plan.transient_b_bytes << '\n';
    for (const auto &step : plan.steps) {
      if (step.kind == StepKind::FillBuffer) {
        std::cout << "F " << step.launch_index << ' ' << resource(step.resource)
                  << ' ' << step.offset << ' ' << step.size << ' ' << step.value << '\n';
      } else if (step.kind == StepKind::Barrier) {
        std::cout << "B " << step.launch_index << ' ' << step.barrier_resources.size();
        for (auto item : step.barrier_resources)
          std::cout << ' ' << resource(item);
        std::cout << '\n';
      } else {
        std::cout << "D " << step.launch_index << ' ' << step.shader->name << ' '
                  << step.grid[0] << ' ' << step.grid[1] << ' ' << step.grid[2] << ' ';
        hex(step.uniform);
        std::cout << ' ' << step.descriptors.size() << '\n';
        for (uint32_t i = 0; i < step.descriptors.size(); ++i) {
          const auto &descriptor = step.descriptors[i];
          std::cout << "d " << i << ' ' << descriptor.kind << ' '
                    << resource(descriptor.resource) << ' ' << descriptor.offset << ' '
                    << descriptor.range << ' ' << unsigned(descriptor.access) << ' '
                    << descriptor.linear_filter << '\n';
        }
      }
    }
    std::cout << "E\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
