// Portable fp16 helpers shared by the HLSL shaders and the CPU emulators. Integer code plus plain IEEE fp32 operations
// only (every fp32 operation is a single correctly rounded add/sub/mul; no fused multiply-add, no fp16 conversion
// instructions), so the GPU (any fp16 denormal mode, any FMA lowering) and the CPU produce identical bits.
// The includer defines: uint (32-bit unsigned), ASUINT(float) -> uint, ASFLOAT(uint) -> float, PRECISE (HLSL: precise).

// fp16 bits -> float (exact, denormals kept).
float fp_h2f(uint h)
{
    uint s = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float den = float(m) * 5.9604644775390625e-08f;                 // m * 2^-24, exact
    uint nb = (e == 31u) ? (s | 0x7F800000u | (m << 13)) : (s | ((e + 112u) << 23) | (m << 13));
    return (e == 0u) ? ((s != 0u) ? -den : den) : ASFLOAT(nb);
}

// E5M3 byte (stored as the fp16 bits byte << 7) -> float, exact.
float fp_e5m3(uint b)
{
    return fp_h2f((b & 0xFFu) << 7);
}

// float -> fp16 bits, round to nearest even, with a sticky adjustment: 'err' is the exact remainder of the value
// (true value = f + err, |err| at most half an fp32 ulp of f), so the result is the single rounding of f + err.
// err = 0 gives the plain RNE of f.
uint fp_rne16s(float f, float err)
{
    uint u = ASUINT(f);
    uint s = (u >> 16) & 0x8000u;
    uint a = u & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return 0x7E00u;                           // NaN
    // direction of err relative to |f|: +1 away from zero, -1 toward zero, 0 none
    uint ea = ASUINT(err);
    int dir = ((ea & 0x7FFFFFFFu) == 0u) ? 0 : ((((ea ^ u) & 0x80000000u) == 0u) ? 1 : -1);
    if (a >= 0x7F800000u) return s | 0x7C00u;                      // inf
    uint sh, mant;
    if (a >= 0x38800000u) {                                        // fp16-normal range: keep 10 mantissa bits
        sh = 13u;
        mant = a;                                                  // rounding on the whole encoding carries into the exponent
    } else {
        uint e = a >> 23;
        if (e < 102u) {                                            // below a quarter of the smallest subnormal (2^-25 > |f|)
            // |f| < 2^-25 (half the smallest subnormal): result 0 unless exactly half with positive err; |f| < 2^-25 strictly
            return s;
        }
        sh = 126u - e;                                             // 14..24: shift of the (implicit-1) mantissa
        mant = (a & 0x7FFFFFu) | 0x800000u;
    }
    uint q = mant >> sh;
    uint rem = mant & ((1u << sh) - 1u);
    uint half_ = 1u << (sh - 1u);
    bool up;
    if (rem > half_) up = true;
    else if (rem < half_) up = (rem == 0u) ? false : false;
    else up = (dir > 0) ? true : ((dir < 0) ? false : ((q & 1u) != 0u));
    // rem == 0: exact at the fp16 grid; err (smaller than half an fp32 ulp) cannot cross half an fp16 ulp
    // rem < half: a negative err only lowers it further; a positive err cannot reach half (|err| < fp32 ulp/2 << fp16 ulp)
    // rem > half: a negative err cannot cross back below half for the same reason
    q += up ? 1u : 0u;
    uint r;
    if (a >= 0x38800000u) {
        r = q - (0x38000000u >> 13);                               // rebias exponent (127 -> 15): subtract 112 << 10
        if (r >= 0x7C00u) r = 0x7C00u;                             // overflow to inf
    } else {
        r = q;                                                     // subnormal (or rounds up into the smallest normal)
    }
    return s | r;
}

// float -> fp16 bits, round to nearest even (single rounding of an exact float).
uint fp_rne16(float f)
{
    return fp_rne16s(f, 0.0f);
}

// fp16 bits of fp16(x + y) for floats x, y, rounded once (TwoSum remainder as sticky).
uint fp_add_rne16(float x, float y)
{
    PRECISE float s = x + y;
    PRECISE float bb = s - x;
    PRECISE float e0 = x - (s - bb);
    PRECISE float e1 = y - bb;
    PRECISE float err = e0 + e1;
    return fp_rne16s(s, err);
}

// fp16 bits of fp16(a + b) for fp16 bit patterns a, b (one rounding).
uint fp_add16(uint a, uint b)
{
    return fp_add_rne16(fp_h2f(a), fp_h2f(b));
}

// fp16 bits of fp16(a * b + c) (a, b fp16 bit patterns, c float), rounded once: a * b is exact in fp32.
uint fp_mixlo(uint a, uint b, float c)
{
    PRECISE float p = fp_h2f(a) * fp_h2f(b);
    return fp_add_rne16(p, c);
}

// fp32 a*b + c for fp16 bit patterns a, b: the product is exact in fp32, so this equals the fused fma.
float fp_fmix(uint a, uint b, float c)
{
    PRECISE float p = fp_h2f(a) * fp_h2f(b);
    PRECISE float r = p + c;
    return r;
}

// fp16 bits of fp16(h + (h & 0xFC00) * 0.0625) (one rounding; the sum is exact in fp32): the E5M3 pre-rounding.
uint fp_e5m3s(uint h)
{
    PRECISE float r = fp_h2f(h) + fp_h2f(h & 0xFC00u) * 0.0625f;
    return fp_rne16(r);
}

// E5M3 byte of an fp16 value (pre-round, then take the top byte below the sign shift), as NVIDIA's cvt sequence.
uint fp_to_e5m3(uint h)
{
    return ((fp_e5m3s(h) << 1) >> 8) & 0xFFu;
}

// Round-to-odd fixup: f + err (err = the exact TwoSum remainder of f) moved to the fp32 neighbour with an odd
// mantissa when inexact. Rounding that value to fp16 with RNE equals the single RNE rounding of f + err
// (24 >= 11 + 2 bits), so a hardware fp32 -> fp16 RNE conversion gives a correctly rounded fp16 sum.
float fp_rto(float f, float err)
{
    uint u = ASUINT(f), ea = ASUINT(err);
    if ((ea & 0x7FFFFFFFu) != 0u && (u & 1u) == 0u) u = (((ea ^ u) & 0x80000000u) == 0u) ? u + 1u : u - 1u;
    return ASFLOAT(u);
}

// relu on fp16 bits: negatives, -0 and NaN -> +0.
uint fp_relu16(uint r)
{
    return (((r & 0x8000u) != 0u) || (r > 0x7C00u)) ? 0u : r;
}
