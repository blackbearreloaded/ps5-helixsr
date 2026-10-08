/* Copyright (C) 2026 BlackBearReloaded
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Experimental native HelixSR Model E runtime for Vulkan/ps5vk.
 *
 * The application owns the Vulkan device, command buffer, input/output images,
 * submission and synchronization. The context owns the repacked model,
 * pipelines, descriptors, constants, intermediates and temporal history.
 * Dispatch records commands only. Do not reuse or destroy a context until the
 * recorded work has completed; call ps5helixsr_context_notify_completed then.
 *
 * Fixed formats: color/output RGBA16F, depth R32F, motion RG16F. Motion vectors
 * are current-to-previous in render-pixel units. Color is linear HDR and depth
 * is inverted. All images have one mip and one array layer.
 */
#ifndef PS5HELIXSR_PS5_HELIXSR_H
#define PS5HELIXSR_PS5_HELIXSR_H

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PS5HELIXSR_VERSION_MAJOR 0
#define PS5HELIXSR_VERSION_MINOR 1
#define PS5HELIXSR_VERSION_PATCH 0

typedef enum ps5helixsr_result {
  PS5HELIXSR_OK = 0,
  PS5HELIXSR_ERROR_INVALID_ARGUMENT = -1,
  PS5HELIXSR_ERROR_UNSUPPORTED = -2,
  PS5HELIXSR_ERROR_OUT_OF_MEMORY = -3,
  PS5HELIXSR_ERROR_VULKAN = -4,
  PS5HELIXSR_ERROR_IN_FLIGHT = -5
} ps5helixsr_result;

typedef enum ps5helixsr_flags {
  /* Compute exposure from input luminance. */
  PS5HELIXSR_FLAG_AUTO_EXPOSURE = 1u << 0,
  /* The device enabled subgroupSizeControl; request 32 lanes on every pass. */
  PS5HELIXSR_FLAG_SUBGROUP_SIZE_CONTROL = 1u << 1
} ps5helixsr_flags;

typedef struct ps5helixsr_context ps5helixsr_context;

typedef struct ps5helixsr_context_desc {
  uint32_t struct_size;
  VkPhysicalDevice physical_device;
  VkDevice device;
  uint32_t output_width, output_height;
  uint32_t render_width, render_height;
  uint32_t flags;
  const void *canonical_weights;
  size_t canonical_weights_bytes;
  const VkAllocationCallbacks *allocator;
  VkPipelineCache pipeline_cache;
  /* The network's kernels (tools/pack_kernels.py), for a runtime built without
   * them: such a build holds only their SHA-256 and refuses any other module.
   * NULL when the kernels are compiled in. Not needed after creation. */
  const void *kernels;
  size_t kernels_bytes;
} ps5helixsr_context_desc;

typedef struct ps5helixsr_dispatch_desc {
  uint32_t struct_size;
  VkCommandBuffer command_buffer;
  VkImageView color, depth, motion_vectors, output;
  VkImageLayout color_layout, depth_layout, motion_vectors_layout;
  float jitter_x, jitter_y;
  float pre_exposure;
  uint32_t reset;
} ps5helixsr_dispatch_desc;

typedef struct ps5helixsr_memory_requirements {
  VkDeviceSize device_bytes;
  VkDeviceSize host_visible_bytes;
  uint32_t pipeline_count;
  uint32_t descriptor_set_count;
} ps5helixsr_memory_requirements;

ps5helixsr_result ps5helixsr_get_memory_requirements(
    const ps5helixsr_context_desc *desc,
    ps5helixsr_memory_requirements *requirements);
ps5helixsr_result ps5helixsr_context_create(const ps5helixsr_context_desc *desc,
                                            ps5helixsr_context **context);
void ps5helixsr_context_destroy(ps5helixsr_context *context);
ps5helixsr_result ps5helixsr_dispatch(ps5helixsr_context *context,
                                      const ps5helixsr_dispatch_desc *desc);
ps5helixsr_result
ps5helixsr_context_notify_completed(ps5helixsr_context *context);

#ifdef __cplusplus
}
#endif
#endif
