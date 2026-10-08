#!/usr/bin/env python3
# Copyright 2026 lonewolf0622. Licensed under the Apache License, Version 2.0.
"""Texture sampler settings of the network kernels (see SAMPLERS)."""
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # helixsr-setup

# CUDA texture samplers NGX creates for each kernel's texture parameters (they are not in the PTX):
# parameter offset:filter (0 point, 1 linear):address U:address V (1 clamp). Fixed per DLSS version, the same in
# every recorded launch (187 records, 4 resolutions, Quality to Ultra Performance).
K8_SAMPLERS = '280:1:1:1,288:1:1:1,304:0:1:1,312:0:1:1,320:1:1:1,344:0:1:1,352:0:1:1,360:0:1:1'
SAMPLERS = {
    'k5': '456:1:1:1,472:1:1:1,480:1:1:1',
    'k5f': '456:1:1:1,464:1:1:1,472:1:1:1,480:1:1:1',
    'k5n': '456:1:1:1,472:1:1:1,480:1:1:1',
    'u4': '456:1:1:1,464:1:1:1,472:1:1:1,480:1:1:1',
    'luma_convert': '40:1:1:1',
    'copy_exposure': '8:0:1:1',
}


def spec(name):
    return K8_SAMPLERS if name.startswith('k8') else SAMPLERS.get(name, '')
