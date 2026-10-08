cmake_minimum_required(VERSION 3.16)
project(ps5_helixsr LANGUAGES C CXX)
option(HELIXSR_SANITIZE "Instrument host checks" ON)
option(HELIXSR_HOST_VULKAN "Build the offline Vulkan buffer runner" OFF)
option(HELIXSR_QUALIFY_WAVE32 "Require the selected host driver to pass actual 32-lane operations" OFF)
set(HELIXSR_DXC "" CACHE FILEPATH "DXC executable for offline Vulkan probes")
set(HELIXSR_VULKAN_ICD "" CACHE FILEPATH "Optional host Vulkan ICD selection for tests")
set(HELIXSR_PTX_DIR "" CACHE PATH "External Model E sm_89 PTX for optional stage comparison")
set(HELIXSR_GENERATED_DIR "" CACHE PATH "External generated main shader directory for binding integration checks")
set(HELIXSR_GENERATED_INVENTORY "${CMAKE_CURRENT_SOURCE_DIR}/validation/generated-main.json" CACHE FILEPATH "Pinned generated shader inventory")
set(HELIXSR_COMPACT_DIR "" CACHE PATH "Compiled compact resident shader directory")
set(HELIXSR_COMPACT_INVENTORY "${CMAKE_CURRENT_SOURCE_DIR}/validation/compact-resident.json" CACHE FILEPATH "Pinned compact resident shader inventory")
set(HELIXSR_MODEL_WEIGHTS "" CACHE FILEPATH "External canonical Model E weights for host API validation")
set(HELIXSR_SETUP_DIR "" CACHE PATH "Published HelixSR setup directory for reference-capability audit")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(HELIXSR_HOST_VULKAN)
  find_package(Vulkan REQUIRED)
  add_executable(helixsr_host_compute tools/host_compute.cpp)
  target_compile_features(helixsr_host_compute PRIVATE cxx_std_17)
  target_link_libraries(helixsr_host_compute PRIVATE Vulkan::Vulkan)
  add_executable(helixsr_host_buffers tools/host_buffers.cpp)
  target_compile_features(helixsr_host_buffers PRIVATE cxx_std_17)
  target_link_libraries(helixsr_host_buffers PRIVATE Vulkan::Vulkan)
  add_executable(helixsr_host_capabilities tools/host_capabilities.cpp)
  target_compile_features(helixsr_host_capabilities PRIVATE cxx_std_17)
  target_link_libraries(helixsr_host_capabilities PRIVATE Vulkan::Vulkan)
  add_executable(helixsr_host_record tools/host_record.cpp)
  target_compile_features(helixsr_host_record PRIVATE cxx_std_17)
  target_link_libraries(helixsr_host_record PRIVATE Vulkan::Vulkan)
endif()
add_library(helixsr_planner STATIC third_party/helixsr/model/model.cpp)
target_include_directories(helixsr_planner PUBLIC third_party/helixsr/model)
target_compile_features(helixsr_planner PUBLIC cxx_std_17)
if(HELIXSR_GENERATED_DIR AND HELIXSR_COMPACT_DIR)
  set(HELIXSR_ASSET_HEADER ${CMAKE_CURRENT_BINARY_DIR}/runtime-assets/helixsr_assets.h)
  set(HELIXSR_ASSET_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/runtime-assets/helixsr_assets.cpp)
  set(HELIXSR_ASSET_RECEIPT ${CMAKE_CURRENT_BINARY_DIR}/runtime-assets/runtime-assets.json)
  add_custom_command(OUTPUT ${HELIXSR_ASSET_HEADER} ${HELIXSR_ASSET_SOURCE} ${HELIXSR_ASSET_RECEIPT}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/build_runtime_assets.py
      --inventory ${HELIXSR_GENERATED_INVENTORY} --generated ${HELIXSR_GENERATED_DIR}
      --compact-inventory ${HELIXSR_COMPACT_INVENTORY} --compact-dir ${HELIXSR_COMPACT_DIR}
      --header ${HELIXSR_ASSET_HEADER} --source ${HELIXSR_ASSET_SOURCE} --receipt ${HELIXSR_ASSET_RECEIPT}
    DEPENDS tools/build_runtime_assets.py ${HELIXSR_GENERATED_INVENTORY} ${HELIXSR_COMPACT_INVENTORY}
    VERBATIM)
  add_library(helixsr_runtime_assets STATIC ${HELIXSR_ASSET_SOURCE})
  target_include_directories(helixsr_runtime_assets PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/runtime-assets)
  target_compile_features(helixsr_runtime_assets PUBLIC cxx_std_17)
  add_executable(helixsr_runtime_assets_test tests/runtime_assets.cpp)
  target_link_libraries(helixsr_runtime_assets_test PRIVATE helixsr_runtime_assets)
  add_library(helixsr_runtime_plan STATIC src/runtime_plan.cpp)
  target_include_directories(helixsr_runtime_plan PUBLIC src)
  target_link_libraries(helixsr_runtime_plan PUBLIC helixsr_runtime_assets helixsr_planner)
  target_compile_features(helixsr_runtime_plan PUBLIC cxx_std_17)
  add_executable(helixsr_runtime_plan_test tools/runtime_plan.cpp)
  target_link_libraries(helixsr_runtime_plan_test PRIVATE helixsr_runtime_plan)
  add_test(NAME runtime_assets COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_assets.py
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/build_runtime_assets.py $<TARGET_FILE:helixsr_runtime_assets_test>
    ${HELIXSR_GENERATED_INVENTORY} ${HELIXSR_GENERATED_DIR} ${HELIXSR_COMPACT_INVENTORY}
    ${HELIXSR_COMPACT_DIR} ${HELIXSR_ASSET_HEADER} ${HELIXSR_ASSET_SOURCE} ${HELIXSR_ASSET_RECEIPT})
  add_test(NAME runtime_plan COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_plan.py
    $<TARGET_FILE:helixsr_graph> $<TARGET_FILE:helixsr_runtime_plan_test>
    ${HELIXSR_GENERATED_INVENTORY} ${HELIXSR_GENERATED_DIR} ${HELIXSR_COMPACT_INVENTORY}
    ${HELIXSR_COMPACT_DIR} ${CMAKE_CURRENT_BINARY_DIR}/runtime-plan.json)
endif()
if(HELIXSR_HOST_VULKAN AND HELIXSR_GENERATED_DIR AND HELIXSR_COMPACT_DIR)
  add_library(ps5helixsr STATIC src/ps5_helixsr.cpp)
  target_include_directories(ps5helixsr PUBLIC include)
  target_link_libraries(ps5helixsr PUBLIC Vulkan::Vulkan PRIVATE helixsr_runtime_plan)
  target_compile_features(ps5helixsr PUBLIC cxx_std_17)
  add_executable(helixsr_host_api tools/host_api.cpp)
  target_link_libraries(helixsr_host_api PRIVATE ps5helixsr)
  target_compile_features(helixsr_host_api PRIVATE cxx_std_17)
  target_include_directories(ps5helixsr PUBLIC src)
  target_compile_definitions(ps5helixsr PRIVATE PS5HELIXSR_DEBUG_SHADERS=1)
  add_executable(helixsr_host_frames tools/host_frames.cpp)
  target_link_libraries(helixsr_host_frames PRIVATE ps5helixsr)
  target_compile_features(helixsr_host_frames PRIVATE cxx_std_17)
  add_executable(helixsr_frame_runner samples/frame_runner.cpp)
  target_link_libraries(helixsr_frame_runner PRIVATE ps5helixsr)
  target_compile_definitions(helixsr_frame_runner PRIVATE HELIXSR_HOST=1)
  target_compile_features(helixsr_frame_runner PRIVATE cxx_std_17)
  # The showcase's off-screen host build: tools/build_showcase.py --host makes the headers and builds this.
  set(HELIXSR_SHOWCASE_GENERATED "" CACHE PATH "Shader and font headers made by tools/build_showcase.py --host")
  set(HELIXSR_SHOWCASE_DEFINES "" CACHE STRING "Extra definitions of the showcase's host build")
  if(HELIXSR_SHOWCASE_GENERATED)
    add_executable(helixsr_showcase examples/helixsr_showcase/main.c examples/helixsr_showcase/hud.c)
    target_include_directories(helixsr_showcase PRIVATE ${HELIXSR_SHOWCASE_GENERATED})
    target_compile_definitions(helixsr_showcase PRIVATE SHOWCASE_HOST=1 _DEFAULT_SOURCE ${HELIXSR_SHOWCASE_DEFINES})
    target_link_libraries(helixsr_showcase PRIVATE ps5helixsr m)
    set_target_properties(helixsr_showcase PROPERTIES C_STANDARD 11)
  endif()
  add_executable(helixsr_c_header_test tests/c_header.c)
  target_include_directories(helixsr_c_header_test PRIVATE include)
  target_link_libraries(helixsr_c_header_test PRIVATE Vulkan::Vulkan)
endif()
add_executable(helixsr_graph tools/graph.cpp)
target_link_libraries(helixsr_graph PRIVATE helixsr_planner)
add_executable(helixsr_numerics tools/numerics.cpp)
target_include_directories(helixsr_numerics PRIVATE third_party/helixsr)
target_compile_features(helixsr_numerics PRIVATE cxx_std_17)
target_compile_options(helixsr_numerics PRIVATE -ffp-contract=off -fno-fast-math)
add_executable(helixsr_repack_probe tools/repack_probe.cpp)
target_link_libraries(helixsr_repack_probe PRIVATE helixsr_planner)
if(HELIXSR_SANITIZE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target helixsr_planner helixsr_graph helixsr_numerics helixsr_repack_probe)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PUBLIC -fsanitize=address,undefined)
  endforeach()
  foreach(target helixsr_runtime_plan ps5helixsr helixsr_host_api)
    if(TARGET ${target})
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PUBLIC -fsanitize=address,undefined)
    endif()
  endforeach()
endif()
enable_testing()
if(TARGET helixsr_c_header_test)
  add_test(NAME c_api_header COMMAND helixsr_c_header_test)
endif()
add_test(NAME graph_contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/graph_contract.py $<TARGET_FILE:helixsr_graph>)
add_test(NAME source_contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/source_contract.py $<TARGET_FILE:helixsr_graph>)
add_test(NAME numeric_policy COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/numerics.py $<TARGET_FILE:helixsr_numerics>)
add_test(NAME weight_layout COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/repack.py $<TARGET_FILE:helixsr_repack_probe>)
add_test(NAME showcase_tables COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/showcase.py)
add_test(NAME deterministic_fixtures COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures.py)
add_test(NAME clear_operation COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/clear_operation.py $<TARGET_FILE:helixsr_graph>)
add_test(NAME generated_bindings COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/generated_bindings.py)
add_test(NAME runtime_bindings COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_bindings.py)
add_test(NAME reference_capabilities COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_capabilities.py
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/reference_capabilities.py)
if(HELIXSR_SETUP_DIR)
  add_test(NAME published_reference_capabilities COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/reference_capabilities.py ${HELIXSR_SETUP_DIR}
    ${HELIXSR_GENERATED_INVENTORY} ${CMAKE_CURRENT_BINARY_DIR}/reference-capabilities.json
    --expect-incomplete)
endif()
if(HELIXSR_GENERATED_DIR AND HELIXSR_COMPACT_DIR)
  add_test(NAME native_capabilities COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/native_capabilities.py
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/native_capabilities.py ${HELIXSR_GENERATED_INVENTORY} ${HELIXSR_GENERATED_DIR}
    ${HELIXSR_COMPACT_INVENTORY} ${HELIXSR_COMPACT_DIR} ${CMAKE_CURRENT_BINARY_DIR}/native-capabilities.json)
  add_test(NAME generated_runtime_bindings COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/generated_runtime_bindings.py
    $<TARGET_FILE:helixsr_graph> ${HELIXSR_GENERATED_INVENTORY} ${HELIXSR_GENERATED_DIR}
    ${HELIXSR_COMPACT_INVENTORY} ${HELIXSR_COMPACT_DIR} ${CMAKE_CURRENT_BINARY_DIR}/runtime-bindings.json)
  add_test(NAME runtime_sequence COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_sequence.py
    $<TARGET_FILE:helixsr_graph> ${HELIXSR_GENERATED_INVENTORY} ${HELIXSR_GENERATED_DIR}
    ${HELIXSR_COMPACT_INVENTORY} ${HELIXSR_COMPACT_DIR} ${CMAKE_CURRENT_BINARY_DIR}/runtime-sequence.json)
endif()
if(HELIXSR_HOST_VULKAN AND HELIXSR_DXC)
  find_program(HELIXSR_SPIRV_VAL spirv-val REQUIRED)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/numeric_probe.spv
    COMMAND ${HELIXSR_DXC} -spirv -fspv-target-env=vulkan1.2 -T cs_6_2 -E main -HV 2021 -O3
      -I ${CMAKE_CURRENT_SOURCE_DIR}/third_party/helixsr
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/numeric_probe.hlsl -Fo ${CMAKE_CURRENT_BINARY_DIR}/numeric_probe.spv
    COMMAND ${HELIXSR_SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/numeric_probe.spv
    DEPENDS shaders/numeric_probe.hlsl third_party/helixsr/common/fp16.h VERBATIM)
  add_custom_target(helixsr_probe_shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/numeric_probe.spv)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/subgroup_probe.spv
    COMMAND ${HELIXSR_DXC} -spirv -fspv-target-env=vulkan1.2 -T cs_6_2 -E main -HV 2021 -O3
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/subgroup_probe.hlsl -Fo ${CMAKE_CURRENT_BINARY_DIR}/subgroup_probe.spv
    COMMAND ${HELIXSR_SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/subgroup_probe.spv
    DEPENDS shaders/subgroup_probe.hlsl VERBATIM)
  add_custom_target(helixsr_subgroup_shader ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/subgroup_probe.spv)
  if(HELIXSR_QUALIFY_WAVE32)
    add_test(NAME host_subgroups COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/subgroups.py
      $<TARGET_FILE:helixsr_host_compute> ${CMAKE_CURRENT_BINARY_DIR}/subgroup_probe.spv $<TARGET_FILE:helixsr_host_capabilities>)
    set_tests_properties(host_subgroups PROPERTIES TIMEOUT 180)
    if(HELIXSR_VULKAN_ICD)
      set_tests_properties(host_subgroups PROPERTIES ENVIRONMENT "VK_ICD_FILENAMES=${HELIXSR_VULKAN_ICD}")
    endif()
  endif()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/convolution_pool.spv
    COMMAND ${HELIXSR_DXC} -spirv -fspv-target-env=vulkan1.2 -T cs_6_2 -E main -HV 2021 -O3
      -I ${CMAKE_CURRENT_SOURCE_DIR}/third_party/helixsr
      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/convolution_pool.hlsl -Fo ${CMAKE_CURRENT_BINARY_DIR}/convolution_pool.spv
    COMMAND ${HELIXSR_SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/convolution_pool.spv
    DEPENDS shaders/convolution_pool.hlsl shaders/convolution_core.hlsli third_party/helixsr/common/fp16.h VERBATIM)
  add_custom_target(helixsr_convolution_shader ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/convolution_pool.spv)
  foreach(name convolution_1x1_pool encode_e5m3)
    set(shader_source shaders/${name}.hlsl)
    set(shader_defines)
    if(name STREQUAL "convolution_1x1_pool")
      set(shader_source shaders/convolution_pool.hlsl)
      set(shader_defines -D KERNEL_SIZE=1)
    endif()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv
      COMMAND ${HELIXSR_DXC} -spirv -fspv-target-env=vulkan1.2 -T cs_6_2 -E main -HV 2021 -O3 ${shader_defines}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/third_party/helixsr ${CMAKE_CURRENT_SOURCE_DIR}/${shader_source}
        -Fo ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv
      COMMAND ${HELIXSR_SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv
      DEPENDS ${shader_source} third_party/helixsr/common/fp16.h VERBATIM)
    add_custom_target(helixsr_${name} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv)
  endforeach()
  set(HELIXSR_COMPACT_VARIANTS
    conv_16_32_k3_half:16:32:3:0
    conv_32_64_k1_half:32:64:1:0
    conv_64_128_k3_e5:64:128:3:1
    conv_128_128_k3_half:128:128:3:0
    conv_128_256_k3_half:128:256:3:0)
  foreach(variant IN LISTS HELIXSR_COMPACT_VARIANTS)
    string(REPLACE ":" ";" fields ${variant})
    list(GET fields 0 name)
    list(GET fields 1 channels)
    list(GET fields 2 outputs)
    list(GET fields 3 kernel)
    list(GET fields 4 e5)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv
      COMMAND ${HELIXSR_DXC} -spirv -fspv-target-env=vulkan1.2 -T cs_6_2 -E main -HV 2021 -O3
        -D CONV_CHANNELS=${channels} -D CONV_OUTPUTS=${outputs} -D KERNEL_SIZE=${kernel} -D INPUT_E5=${e5}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/third_party/helixsr -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/convolution_pool_resident.hlsl -Fo ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv
      COMMAND ${HELIXSR_SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv
      DEPENDS shaders/convolution_pool_resident.hlsl shaders/convolution_core.hlsli third_party/helixsr/common/fp16.h VERBATIM)
    add_custom_target(helixsr_${name} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv)
  endforeach()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/encode_e5m3_resident.spv
    COMMAND ${HELIXSR_DXC} -spirv -fspv-target-env=vulkan1.2 -T cs_6_2 -E main -HV 2021 -O3
      -I ${CMAKE_CURRENT_SOURCE_DIR}/third_party/helixsr ${CMAKE_CURRENT_SOURCE_DIR}/shaders/encode_e5m3_resident.hlsl
      -Fo ${CMAKE_CURRENT_BINARY_DIR}/encode_e5m3_resident.spv
    COMMAND ${HELIXSR_SPIRV_VAL} --target-env vulkan1.2 ${CMAKE_CURRENT_BINARY_DIR}/encode_e5m3_resident.spv
    DEPENDS shaders/encode_e5m3_resident.hlsl third_party/helixsr/common/fp16.h VERBATIM)
  add_custom_target(helixsr_encode_e5m3_resident ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/encode_e5m3_resident.spv)
  add_test(NAME host_vulkan_arithmetic COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/host_compute.py
    $<TARGET_FILE:helixsr_host_compute> ${CMAKE_CURRENT_BINARY_DIR}/numeric_probe.spv $<TARGET_FILE:helixsr_numerics>)
  set_tests_properties(host_vulkan_arithmetic PROPERTIES TIMEOUT 180)
  if(HELIXSR_VULKAN_ICD)
    set_tests_properties(host_vulkan_arithmetic PROPERTIES ENVIRONMENT "VK_ICD_FILENAMES=${HELIXSR_VULKAN_ICD}")
  endif()
  if(HELIXSR_PTX_DIR)
    add_test(NAME async_copy_model COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/async_copy.py)
    add_test(NAME fused_stage COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fused_stage.py
      $<TARGET_FILE:helixsr_graph> $<TARGET_FILE:helixsr_host_compute> $<TARGET_FILE:helixsr_host_buffers> ${CMAKE_CURRENT_BINARY_DIR}
      ${HELIXSR_PTX_DIR} ${CMAKE_CURRENT_BINARY_DIR}/fused-stage)
    set_tests_properties(fused_stage PROPERTIES TIMEOUT 300)
    if(HELIXSR_VULKAN_ICD)
      set_tests_properties(fused_stage PROPERTIES ENVIRONMENT "VK_ICD_FILENAMES=${HELIXSR_VULKAN_ICD}")
    endif()
    add_test(NAME convolution_stages COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/convolution_stages.py
      $<TARGET_FILE:helixsr_graph> $<TARGET_FILE:helixsr_host_compute> ${CMAKE_CURRENT_BINARY_DIR}/convolution_pool.spv
      $<TARGET_FILE:helixsr_host_buffers> ${CMAKE_CURRENT_BINARY_DIR}
      ${HELIXSR_PTX_DIR} ${CMAKE_CURRENT_BINARY_DIR}/convolution-stages)
    set_tests_properties(convolution_stages PROPERTIES TIMEOUT 300)
    if(HELIXSR_VULKAN_ICD)
      set_tests_properties(convolution_stages PROPERTIES ENVIRONMENT "VK_ICD_FILENAMES=${HELIXSR_VULKAN_ICD}")
    endif()
  endif()
  if(HELIXSR_GENERATED_DIR AND HELIXSR_COMPACT_DIR)
    add_test(NAME host_vulkan_record COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/host_record.py
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/build_record_fixture.py $<TARGET_FILE:helixsr_host_record>
      $<TARGET_FILE:helixsr_graph> ${HELIXSR_GENERATED_INVENTORY} ${HELIXSR_GENERATED_DIR}
      ${HELIXSR_COMPACT_INVENTORY} ${HELIXSR_COMPACT_DIR} ${CMAKE_CURRENT_BINARY_DIR}/host-record.json)
    set_tests_properties(host_vulkan_record PROPERTIES TIMEOUT 180)
    if(HELIXSR_VULKAN_ICD)
      set_tests_properties(host_vulkan_record PROPERTIES ENVIRONMENT "VK_ICD_FILENAMES=${HELIXSR_VULKAN_ICD}")
    endif()
    if(HELIXSR_MODEL_WEIGHTS)
      add_test(NAME public_vulkan_api COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/host_api.py
        $<TARGET_FILE:helixsr_host_api> ${HELIXSR_MODEL_WEIGHTS} ${CMAKE_CURRENT_BINARY_DIR}/host-api.json)
      set_tests_properties(public_vulkan_api PROPERTIES TIMEOUT 300)
      if(HELIXSR_VULKAN_ICD)
        set_tests_properties(public_vulkan_api PROPERTIES ENVIRONMENT "VK_ICD_FILENAMES=${HELIXSR_VULKAN_ICD}")
      endif()
    endif()
  endif()
endif()
