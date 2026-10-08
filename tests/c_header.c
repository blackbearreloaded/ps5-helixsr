#include "ps5helixsr/ps5_helixsr.h"

int main(void) {
  ps5helixsr_context_desc context = {0};
  ps5helixsr_dispatch_desc dispatch = {0};
  ps5helixsr_memory_requirements memory = {0};
  context.struct_size = sizeof(context);
  dispatch.struct_size = sizeof(dispatch);
  return context.struct_size == sizeof(ps5helixsr_context_desc) &&
                 dispatch.struct_size == sizeof(ps5helixsr_dispatch_desc) &&
                 sizeof(memory) != 0
             ? 0
             : 1;
}
