#include "helixsr_assets.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <string>

int main() {
  using namespace ps5helixsr::assets;
  if (size() != 19 || find(nullptr) || find("missing"))
    return 1;
  std::set<std::string> names;
  size_t bytes = 0;
  for (size_t i = 0; i < size(); ++i) {
    const auto &shader = data()[i];
    if (!shader.name || !shader.code || shader.words < 5 ||
        shader.code[0] != 0x07230203 || !shader.sha256 ||
        std::strlen(shader.sha256) != 64 || !shader.descriptor_kinds ||
        !*shader.descriptor_kinds || !shader.descriptors ||
        shader.descriptor_count != std::strlen(shader.descriptor_kinds) ||
        shader.required_subgroup_size != 32 ||
        find(shader.name) != &shader || !names.emplace(shader.name).second)
      return 2;
    for (size_t j = 0; j < shader.descriptor_count; ++j)
      if (shader.descriptors[j].kind != shader.descriptor_kinds[j])
        return 3;
    if (shader.kernel_name && recipe_for_kernel(shader.kernel_name) != &shader)
      return 4;
    bytes += shader.words * sizeof(uint32_t);
  }
  std::cout << "{\"result\":\"success\",\"assets\":" << size()
            << ",\"bytes\":" << bytes << "}\n";
}
