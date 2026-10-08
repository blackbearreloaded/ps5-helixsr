/* Copyright (C) 2026 BlackBearReloaded
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Diagnostic access for the validation tools. Not part of the public API:
 * a frame can be recorded one step at a time, and every resource the context
 * owns can be copied out between steps for comparison with the reference. */
#ifndef PS5HELIXSR_DEBUG_H
#define PS5HELIXSR_DEBUG_H

#include <ps5helixsr/ps5_helixsr.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ps5helixsr_debug_resource {
  char name[48];
  VkBuffer buffer; /* or */
  VkImage image;   /* always in VK_IMAGE_LAYOUT_GENERAL */
  VkFormat format;
  uint32_t width, height;
  uint64_t bytes;
} ps5helixsr_debug_resource;

typedef struct ps5helixsr_debug_step {
  char name[64];
  uint32_t launch; /* index of the planner launch this step belongs to */
  uint32_t grid[3];
  uint64_t uniform_offset, uniform_bytes; /* in the UNIFORMS resource */
} ps5helixsr_debug_step;

uint32_t ps5helixsr_debug_resources(ps5helixsr_context *context,
                                    ps5helixsr_debug_resource *resources,
                                    uint32_t capacity);
/* Steps of the frame the next dispatch would record. */
uint32_t ps5helixsr_debug_steps(ps5helixsr_context *context, uint32_t reset,
                                ps5helixsr_debug_step *steps,
                                uint32_t capacity);
/* The next dispatches record only steps [first, first + count). The frame is
 * complete, and must be notified, once a range reaches its last step. */
void ps5helixsr_debug_set_range(ps5helixsr_context *context, uint32_t first,
                                uint32_t count);

#ifdef __cplusplus
}
#endif
#endif
