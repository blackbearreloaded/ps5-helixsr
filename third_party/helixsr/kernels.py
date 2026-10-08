"""Work package F kernel table: PTX source, launch block, numerics, capture kernel name.
mma: 'orig' (the original kernels' numerics, 'two' = sites rounded fp32->fp16 in two steps) or a chain mode ('f32' / 'pk') per site
range, as a list of (first_site, last_site, mode)."""
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # helixsr-setup

K = {
    # fast-1x1 family (nchw8): installed = packed-fp16 chains ('pk'); 'two' sites are the original's (for the orig check)
    'k9': dict(block=(256, 1, 1), two='88-95', installed='pk',
               cap='dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_128_128_032_064_128_fp16_e5m3_kernel'),
    'k10': dict(block=(128, 1, 1), two='144-159', installed='pk',
                cap='dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_064_128_032_128_064_fp16_fp16_kernel'),
    'k11': dict(block=(128, 1, 1), two='all', installed='pk',
                cap='dltss_nchw8_conv_1x1_064_032_032_fp16_fp16_kernel'),
    'k4': dict(block=(256, 1, 1), two='', installed='pk',
               cap='dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_128_064_064_032_064_e5m3_e5m3_kernel'),
    'k7': dict(block=(128, 1, 1), two='all', installed='orig',
               cap='dltss_nchw8_bilinear_upsample_conv_1x1_128_128_032_fp16_fp16_kernel'),
    # NHWC (Ultra Performance)
    'u5': dict(block=(256, 1, 1), two='', installed='pk',
               cap='dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_256_064_064_032_064_e5m3_e5m3_kernel'),
    'u6': dict(block=(256, 1, 1), two='', installed='pk',
               cap='dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_128_128_032_064_128_fp16_e5m3_kernel'),
    'u8': dict(block=(128, 1, 1), two='', installed='pk',
               cap='dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_064_128_032_128_032_fp16_fp16_kernel'),
    'u9': dict(block=(128, 1, 1), two='', installed='0-63:pk,64-127:orig',
               cap='dltss_nhwc_bilinear_upsample_conv_1x1_128_128_032_fp16_fp16_kernel'),
    'u10': dict(block=(128, 1, 1), two='', installed='orig',
                cap='dltss_nhwc_conv_1x1_064_032_064_fp16_fp16_kernel'),
    # output kernels
    'k5': dict(block=(256, 1, 1), two='all', installed='pk2',
               cap='dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_mid_3x3_128_032_032_048_032_e5m3_fp16_kernel'),
    'k5f': dict(block=(256, 1, 1), two='all', installed='pk2',
                cap='dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_fastpath_3x3_128_032_032_048_032_e5m3_fp16_kernel'),
    # below a 1.5 scale ratio (incl. native / DLAA); never captured: cap = dfn_mid (same argument layout and texture
    # objects) for the sampler settings only, no cases
    'k5n': dict(block=(256, 1, 1), two='all', installed='pk2',
                cap='dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_mid_3x3_128_032_032_048_032_e5m3_fp16_kernel'),
    'u4': dict(block=(256, 1, 1), two='all', installed='orig',
               cap='dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_fastpath_5x5_128_032_032_048_032_e5m3_fp16_kernel'),
    # engine input (K8) variants and the exposure kernels
    'k8_hdr_mvdiff_mvhi': dict(block=(128, 1, 1),
                               cap='cuda_engine_input_kernel_rel_hdr_mvdiff_mvhi'),
    'k8_ldr_mvdiff_mvhi': dict(block=(128, 1, 1)),
    'k8_hdr_mvdiff_mvlo': dict(block=(128, 1, 1)),
    'k8_ldr_mvdiff_mvlo': dict(block=(128, 1, 1)),
    'k8_hdr_colvar_mvhi': dict(block=(128, 1, 1)),
    'k8_ldr_colvar_mvhi': dict(block=(128, 1, 1)),
    'k8_hdr_colvar_mvlo': dict(block=(128, 1, 1)),
    'k8_ldr_colvar_mvlo': dict(block=(128, 1, 1)),
    'k8_rel': dict(block=(128, 1, 1)),
    'luma_convert': dict(block=(8, 8, 1), cap='cuda_luma_convert_kernel'),
    'reduce_sum': dict(block=(256, 1, 1), cap='cuda_reduce_sum_kernel'),
    'copy_exposure': dict(block=(1, 1, 1), cap='cuda_copy_exposure_kernel'),
    'auto_exposure_copy': dict(block=(1, 1, 1),
                               cap='cuda_auto_exposure_copy_kernel'),
    'auto_exposure': dict(block=(256, 1, 1)),
    'clear_buffer': dict(block=(16, 16, 1)),
}

# Kernel (.entry) name of every port, so the PTX can be taken from an extracted nvngx_dlss.dll instead (HELIXSR_PTX_DIR:
# a directory of <entry>.sm_<NN>.ptx files from tools/extract_ptx.py; HELIXSR_PTX_SM selects the target, default 80).
ENTRY = {
    'k9': 'dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_128_128_032_064_128_fp16_e5m3_kernel',
    'k10': 'dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_064_128_032_128_064_fp16_fp16_kernel',
    'k11': 'dltss_nchw8_conv_1x1_064_032_032_fp16_fp16_kernel',
    'k4': 'dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_128_064_064_032_064_e5m3_e5m3_kernel',
    'k7': 'dltss_nchw8_bilinear_upsample_conv_1x1_128_128_032_fp16_fp16_kernel',
    'u5': 'dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_256_064_064_032_064_e5m3_e5m3_kernel',
    'u6': 'dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_128_128_032_064_128_fp16_e5m3_kernel',
    'u8': 'dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_064_128_032_128_032_fp16_fp16_kernel',
    'u9': 'dltss_nhwc_bilinear_upsample_conv_1x1_128_128_032_fp16_fp16_kernel',
    'u10': 'dltss_nhwc_conv_1x1_064_032_064_fp16_fp16_kernel',
    'k5': 'dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_mid_3x3_128_032_032_048_032_e5m3_fp16_kernel',
    'k5f': 'dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_fastpath_3x3_128_032_032_048_032_e5m3_fp16_kernel',
    'k5n': 'dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_3x3_256_032_032_048_032_e5m3_fp16_kernel',
    'u4': 'dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_fastpath_5x5_128_032_032_048_032_e5m3_fp16_kernel',
    'k8_hdr_mvdiff_mvhi': 'cuda_engine_input_kernel_rel_hdr_mvdiff_mvhi',
    'k8_ldr_mvdiff_mvhi': 'cuda_engine_input_kernel_rel_ldr_mvdiff_mvhi',
    'k8_hdr_mvdiff_mvlo': 'cuda_engine_input_kernel_rel_hdr_mvdiff_mvlo',
    'k8_ldr_mvdiff_mvlo': 'cuda_engine_input_kernel_rel_ldr_mvdiff_mvlo',
    'k8_hdr_colvar_mvhi': 'cuda_engine_input_kernel_rel_hdr_colvar_mvhi',
    'k8_ldr_colvar_mvhi': 'cuda_engine_input_kernel_rel_ldr_colvar_mvhi',
    'k8_hdr_colvar_mvlo': 'cuda_engine_input_kernel_rel_hdr_colvar_mvlo',
    'k8_ldr_colvar_mvlo': 'cuda_engine_input_kernel_rel_ldr_colvar_mvlo',
    'k8_rel': 'cuda_engine_input_kernel_rel',
    'luma_convert': 'cuda_luma_convert_kernel',
    'reduce_sum': 'cuda_reduce_sum_kernel',
    'copy_exposure': 'cuda_copy_exposure_kernel',
    'auto_exposure_copy': 'cuda_auto_exposure_copy_kernel',
    'auto_exposure': 'cuda_auto_exposure_kernel',
    'clear_buffer': 'cuda_clear_buffer_kernel',
}

import os as _os
for _n, _k in K.items():
    if True:
        _k['ptx'] = _os.path.join(_os.environ.get('HELIXSR_PTX_DIR', ''),
                                  f"{ENTRY[_n]}.sm_{_os.environ.get('HELIXSR_PTX_SM', '80')}.ptx")

