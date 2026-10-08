# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
#
# HelixSR for PS5 on the ps5-vulkan driver (the external/ps5-vulkan submodule).
# BUILDING.md describes the targets in order.
PYTHON ?= python3
CC ?= cc
DRIVER ?= external/ps5-vulkan
DRIVER_URL ?= https://github.com/blackbearreloaded/ps5-vulkan.git
# The driver revision this checkout pins (its submodule commit); empty outside git.
DRIVER_REV ?= $(shell git ls-tree HEAD $(DRIVER) 2>/dev/null | awk '{print $$3}')
# The driver profile HelixSR needs: 16-bit integers, the three subgroup operations of the
# generated kernels, RG16F and R16F storage images and the refresh of sampled inputs.
DRIVER_PROFILE = PS5VK_HELIXSR_DIAGNOSTIC=1 PS5VK_EXTENDED_COMPUTE_DIAGNOSTIC=1
VULKAN_CFLAGS = -I$(DRIVER)/third_party/vulkan-headers/include

# Pinned public inputs (make inputs) and what HelixSR's setup makes from NVIDIA's DLL (make model).
INPUTS ?= build/inputs
DXC ?= $(INPUTS)/dxc/bin/dxc
SPIRV_OPT ?= $(shell command -v spirv-opt)
HELIXSR_RELEASE ?= $(INPUTS)/helixsr
SETUP_OUT ?= build/helixsr-setup
SETUP_WORK ?= build/helixsr-setup-work
MODEL ?= $(SETUP_OUT)/helixsr_weights.bin
PTX ?= $(SETUP_WORK)/ptx
GENERATED ?= build/main-inventory
RESIDENT ?= build/compact-resident
# The Vulkan driver host runs use. The generated kernels need 32-lane subgroups: see BUILDING.md.
VULKAN_ICD ?= /usr/share/vulkan/icd.d/lvp_icd.json
VULKAN_LIBRARY ?= /usr/lib/x86_64-linux-gnu/libvulkan.so.1

.PHONY: all inputs driver driver-source driver-headers driver-deps driver-psbc driver-sdk compile-probe \
	model shaders network-files host check check-offline check-shaders runner showcase
all: shaders host

# Pinned public inputs into build/inputs; eval "$$(python3 tools/fetch_build_inputs.py --env)"
# then points the PS5 builds at them.
inputs:
	$(PYTHON) tools/fetch_build_inputs.py

# Fetch the driver when it is missing: the pinned submodule in a git checkout,
# otherwise a clone of DRIVER_URL at DRIVER_REV (or its default branch).
driver-source: $(DRIVER)/Makefile
$(DRIVER)/Makefile:
	@if [ -n "$(DRIVER_REV)" ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		git submodule update --init -- $(DRIVER); \
	else \
		rmdir $(DRIVER) 2>/dev/null || true; \
		git clone $(DRIVER_URL) $(DRIVER) && \
		{ [ -z "$(DRIVER_REV)" ] || git -C $(DRIVER) checkout -q $(DRIVER_REV); }; \
	fi
driver-headers: | $(DRIVER)/Makefile
	$(MAKE) -C $(DRIVER) vulkan-headers

# Driver: pinned dependencies, the PSBC compiler (PS5 and host), then its staged SDK.
driver: driver-deps driver-psbc driver-sdk
driver-deps: | $(DRIVER)/Makefile
	$(MAKE) -C $(DRIVER) vulkan-headers native-deps compiler-deps
driver-psbc: | $(DRIVER)/Makefile
	cd $(DRIVER) && $(PYTHON) tools/build_psbc.py --target ps5 && $(PYTHON) tools/build_psbc.py --host
driver-sdk: | $(DRIVER)/Makefile
	cd $(DRIVER) && $(DRIVER_PROFILE) $(PYTHON) tools/build_sdk.py
$(DRIVER)/build/libpsbc.host.a: | $(DRIVER)/Makefile
	cd $(DRIVER) && $(PYTHON) tools/build_psbc.py --host

# Compiles SPIR-V with the PS5 shader compiler on the host: every shader is checked with it.
compile-probe: build/compile_probe
build/compile_probe: tools/compile_probe.c | $(DRIVER)/Makefile $(DRIVER)/build/libpsbc.host.a
	mkdir -p build
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror $(VULKAN_CFLAGS) -I$(DRIVER)/src -I$(DRIVER)/include \
		-I$(DRIVER)/third_party/opengnm/include -I$(DRIVER)/third_party/psbc-reference \
		$(DRIVER)/src/ps5vk_compiler.c $(DRIVER)/src/ps5_compiler_shims.c tools/compile_probe.c \
		$(DRIVER)/build/libpsbc.host.a -lstdc++ -lm -lpthread -o $@

# The network, built on this machine by HelixSR's own setup from NVIDIA's DLSS 310.7.0 DLL:
# the weights (MODEL) and the kernels' PTX (PTX). Nothing of it may be committed or shared.
# DLSS_DLL=/path/to/nvngx_dlss.dll uses a local copy; without one the setup downloads the DLL
# from NVIDIA's GitHub, which needs ACCEPT_NVIDIA_DLSS_LICENSE=1.
model: $(MODEL)
$(MODEL):
	test -f $(HELIXSR_RELEASE)/setup/helixsr_setup.py || { echo "run make inputs first"; exit 1; }
	test -n "$(DLSS_DLL)" || test "$(ACCEPT_NVIDIA_DLSS_LICENSE)" = 1 || { \
		echo "set DLSS_DLL=/path/to/nvngx_dlss.dll (310.7.0), or ACCEPT_NVIDIA_DLSS_LICENSE=1 to let"; \
		echo "HelixSR's setup download it under https://github.com/NVIDIA/DLSS/blob/v310.7.0/LICENSE.txt"; exit 1; }
	rm -rf $(SETUP_OUT) $(SETUP_WORK) && cp -r $(HELIXSR_RELEASE) $(SETUP_OUT)
	cd $(SETUP_OUT) && $(PYTHON) setup/helixsr_setup.py $(abspath $(SETUP_OUT)) \
		$(if $(DLSS_DLL),--dlss $(abspath $(DLSS_DLL)),--yes) --dxc $(abspath $(DXC)) --keep $(abspath $(SETUP_WORK))

# The planner alone (no Vulkan, no model): what the shader generator asks for its graphs.
build/helixsr_graph:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
	cmake --build build --target helixsr_graph -j

# The runtime's shaders: the kernels translated from the PTX, and the early convolutions and
# the encoder written here. Their identities are pinned in validation/.
shaders: build/helixsr_graph build/compile_probe
	$(PYTHON) tools/build_generated.py --setup $(HELIXSR_RELEASE)/setup --ptx $(PTX) --dxc $(DXC) \
		--spirv-opt $(SPIRV_OPT) --compiler build/compile_probe --graph build/helixsr_graph --out $(GENERATED)
	$(PYTHON) tools/build_compact_resident.py --dxc $(DXC) --compiler build/compile_probe --out $(RESIDENT)

# The two files an application built without the network needs beside it: the weights and
# the kernels. They are NVIDIA's material made on this machine: for your own console only.
NETWORK_FILES ?= build/network-files
network-files: $(MODEL)
	mkdir -p $(NETWORK_FILES)
	cp $(MODEL) $(NETWORK_FILES)/model.bin
	$(PYTHON) tools/pack_kernels.py --generated $(GENERATED)/generated --out $(NETWORK_FILES)/kernels.bin

# The host build: the runtime as a library, the frame runners and every test.
host: driver-headers
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DHELIXSR_HOST_VULKAN=ON \
		-DVulkan_INCLUDE_DIR=$(abspath $(DRIVER))/third_party/vulkan-headers/include \
		-DVulkan_LIBRARY=$(VULKAN_LIBRARY) -DHELIXSR_DXC=$(abspath $(DXC)) -DHELIXSR_VULKAN_ICD=$(VULKAN_ICD) \
		-DHELIXSR_GENERATED_DIR=$(abspath $(GENERATED))/generated -DHELIXSR_COMPACT_DIR=$(abspath $(RESIDENT)) \
		-DHELIXSR_MODEL_WEIGHTS=$(abspath $(MODEL)) -DHELIXSR_PTX_DIR=$(abspath $(PTX)) \
		-DHELIXSR_SETUP_DIR=$(abspath $(HELIXSR_RELEASE))/setup
	cmake --build build -j
check: host
	ctest --test-dir build --output-on-failure
# The tests that need neither Vulkan nor NVIDIA's material: planner, numerics, contracts.
check-offline:
	cmake -S . -B build/offline -DCMAKE_BUILD_TYPE=Debug
	cmake --build build/offline -j
	ctest --test-dir build/offline --output-on-failure

# The shaders written here, which need nothing of NVIDIA's: the early convolutions as generated,
# and the showcase's scene, composition and presentation.
check-shaders:
	mkdir -p build/check-shaders
	for n in $$($(PYTHON) -c "import sys; sys.path.insert(0, 'tools'); import conv_kernels; print(*conv_kernels.VARIANTS)"); do \
		$(PYTHON) tools/conv_kernels.py $$n > build/check-shaders/$$n.comp && \
		glslangValidator -V --target-env vulkan1.2 build/check-shaders/$$n.comp -o build/check-shaders/$$n.spv > /dev/null && \
		spirv-val --target-env vulkan1.2 build/check-shaders/$$n.spv || exit 1; done
	for s in city.comp compose.comp blit.vert blit.frag; do \
		glslangValidator -V --target-env vulkan1.1 examples/helixsr_showcase/$$s -o build/check-shaders/$$s.spv > /dev/null || exit 1; done
	glslangValidator -V --target-env vulkan1.1 -DSCENE_COLOR_ONLY examples/helixsr_showcase/city.comp \
		-o build/check-shaders/city-color.spv > /dev/null

# PS5 titles. They need the driver SDK (make driver), the shaders and PS5_PAYLOAD_SDK and
# PS5_NATIVE_APP_TEMPLATE in the environment (make inputs).
# RUNNER_CASES lists frame-input directories for the frame runner (VALIDATION.md).
runner:
	PS5VK_SDK=$(abspath $(DRIVER))/dist-sdk $(PYTHON) tools/build_runner.py $(RUNNER_CASES) --model $(MODEL) \
		--generated-dir $(GENERATED)/generated --compact-dir $(RESIDENT) $(RUNNER_ARGS)
# PPSA99013, built without the network: it reads model.bin and kernels.bin (make network-files)
# from its assets folder. SHOWCASE_ARGS takes --model FILE, --kernels FILE, --selftest, --keys,
# --host or --version VERSION.
showcase:
	PS5VK_SDK=$(abspath $(DRIVER))/dist-sdk $(PYTHON) tools/build_showcase.py \
		--generated-dir $(GENERATED)/generated --compact-dir $(RESIDENT) $(SHOWCASE_ARGS)
