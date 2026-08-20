// ---------------------------------------------------------------------------
// Display Scanlines — integer-locked CRT emulation for Hyprland
//
// This file is the shared shader BODY. It is not usable on its own: the
// `display-scanlines` CLI prepends a preamble that supplies `#version 300 es`
// and `#define CRT_LEVEL <1|2>` (1 = Light, 2 = Heavy), then writes
// the result to the generated active shader. Keeping one body means the three
// levels can never drift apart.
//
// Design notes (why this looks clean instead of moire-ridden):
//
//  1. RESOLUTION IS DISCOVERED, NOT CONFIGURED.
//     textureSize(tex, 0) returns the *physical* pixel size of the output
//     currently being rendered — verified 2256x1504 on a 1.6x-scaled display,
//     i.e. it ignores the logical/scaled size. Because Hyprland runs this
//     shader once per output, a single file adapts to every monitor
//     simultaneously, at any resolution, scale, or refresh rate, and follows
//     hotplug and mode changes with no regeneration.
//
//  2. SCANLINES ARE INTEGER BY CONSTRUCTION.
//     Naive CRT shaders do sin(uv.y * height), which accumulates floating
//     point phase error and produces uneven lines and banding from top to
//     bottom. Here the beam phase comes from `int(gl_FragCoord.y) % pitch`.
//     gl_FragCoord maps 1:1 to physical rows (verified: alternating pure
//     black/white rows land exactly on alternating pixels), so every scanline
//     cell is bit-identical to every other one. Consistency is guaranteed by
//     integer arithmetic rather than hoped for.
//
//  3. PITCH PREFERS AN EXACT DIVISOR OF THE PANEL HEIGHT.
//     Integer modulo already guarantees even lines; picking a pitch that
//     divides the height also avoids a single truncated cell at the bottom
//     edge. Selection is automatic and derived from panel height.
//
//  4. LIGHT IS MIXED IN LINEAR SPACE.
//     Beam profile, mask, blur and glow are applied to linearized samples and
//     then re-encoded. Doing this in gamma space is what makes most CRT
//     shaders look muddy and dim; in linear light the highlights stay punchy
//     and the brightness compensation below is physically meaningful.
//
//  5. BRIGHTNESS COMPENSATION IS ANALYTIC.
//     Scanlines and the phosphor mask both remove light. Rather than guessing
//     a fudge factor, the average beam weight over one cell and the average
//     mask weight over one triad are summed exactly and divided back out, so
//     the desktop keeps its intended luminance at any pitch.
//
//  6. NO `time` UNIFORM ON PURPOSE.
//     Hyprland only disables damage tracking for shaders that animate. This
//     shader is static, so damage tracking keeps working and idle desktops
//     cost no extra GPU or battery. That rules out flicker/rolling artifacts,
//     which is the right trade for a display you actually work on.
// ---------------------------------------------------------------------------

precision highp float;

in vec2 v_texcoord;
uniform sampler2D tex;

layout(location = 0) out vec4 fragColor;

// ============================ level parameters =============================
// Three curated presets. Light is the subtle everyday setting; Heavy is an
// unapologetic novelty mode.

// NOTE ON THE `USE_*` FLAGS BELOW:
// The GLSL/C preprocessor evaluates `#if` using *integer* arithmetic only — a
// float literal in a conditional is invalid and silently mis-evaluates rather
// than erroring on some drivers. (`#if GLOW_STRENGTH > 0.0` originally
// compiled but took the wrong branch, blowing the Heavy preset out to pure
// white.) So every optional pass is gated on an explicit integer USE_* flag
// and the float strength is only ever used in real GLSL code.

#if CRT_LEVEL == 1
// ---- LIGHT: subtle, all-day usable. --------------------------------------
#define BEAM_MIN        0.34  // beam width in dark areas (cell units)
#define BEAM_MAX        0.58  // beam width in bright areas
#define SCANLINE_GAIN   0.94  // how fully scanlines are applied
#define MASK_STRENGTH   0.28  // aperture-grille strength
#define MASK_DIM        0.70  // non-native channel level within a triad
#define BLUR_X          0.62  // beam spot size, physical px (horizontal)
#define BLUR_Y          0.22  // ... and vertical (tight, like a real spot)
#define CONTRAST        0.34  // S-curve strength around mid-grey
#define SATURATION      1.12  // phosphors are more vivid than an LCD
#define BRIGHTNESS      1.03  // final luminance trim
#define VIGNETTE        0.10  // corner falloff
#define WARMTH          0.00  // amber/phosphor tint
#define GLOW_STRENGTH   0.00  // bloom
#define USE_GLOW        0     // integer gates (see note above)
#define USE_CONTRAST    1
#define USE_SATURATION  1
#define USE_WARMTH      0
#define USE_MASK        1
#define USE_VIGNETTE    1

#else
// ---- HEAVY: heavy-handed on purpose — glow, warm cast, deep vignette and
// pitch-black gaps. Novelty mode. Uses the same fine scanline pitch as Light;
// the intensity comes from everything except line size.
// Beam width is what sets gap darkness, and it widens with content
// brightness — so BEAM_MAX (bright content) is the lever for "pitch black",
// not BEAM_MIN. At this 2px pitch, 0.24 puts the gap at 0.00/255 across the
// whole brightness range; the old 0.46 left it at ~115/255 (grey, not black).
#define BEAM_MIN        0.18  // pitch-black gaps
#define BEAM_MAX        0.24  // ... at bright content too
#define SCANLINE_GAIN   1.00
#define MASK_STRENGTH   0.55
#define MASK_DIM        0.42
#define BLUR_X          0.72
#define BLUR_Y          0.30  // kept proportional to BLUR_X (~0.41x)
#define CONTRAST        0.34  // same as Light, by design
#define SATURATION      1.32
#define BRIGHTNESS      1.14
#define VIGNETTE        0.46  // darker vignette frame
#define WARMTH          0.26  // strong warm filter
#define GLOW_STRENGTH   0.42  // glowy vibe
#define USE_GLOW        1
#define USE_CONTRAST    1
#define USE_SATURATION  1
#define USE_WARMTH      1
#define USE_MASK        1
#define USE_VIGNETTE    1
#endif

// Glow geometry. Only compiled when the level enables it.
#define GLOW_RADIUS     3.4   // outermost glow radius in physical px
#define GLOW_TAPS       12    // directions sampled around the source
#define GLOW_RINGS      3     // radii per direction (see note in main)
#define GLOW_THRESHOLD  0.34  // only light above this blooms

// ---------------------------- color helpers --------------------------------

vec3 srgbToLinear(vec3 c) {
    // Exact piecewise sRGB transfer, not the 2.2 approximation. The toe
    // matters here because scanline gaps push a lot of pixels into near-black,
    // where a pow() approximation visibly crushes shadow detail.
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 linearToSrgb(vec3 c) {
    c = max(c, vec3(0.0));
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// -------------------------- pitch determination ----------------------------

// Physical pixels per emulated scanline. Identical for both active levels —
// they differ in beam profile, colour and blur, never in line size.
//
// The target scales with panel height so the effect keeps a consistent apparent
// size instead of vanishing on dense panels: ~2px up to 1440p, 3px at 4K, 4px
// beyond. Then we search outward from the target for a pitch that divides the
// height exactly, so the bottom cell isn't clipped.
//
// Mirrored by scanlinePitch() in Model.js for the panel's readout — verified
// to agree across 17 real panel heights.
int scanlinePitch(int h) {
    int target = int(floor(float(h) / 720.0 + 0.5));
    target = clamp(target, 2, 4);

    for (int d = 0; d <= 2; d++) {
        int up = target + d;
        if (up <= 6 && h % up == 0) return up;
        int down = target - d;
        if (down >= 2 && h % down == 0) return down;
    }
    // Odd/prime heights: modulo still guarantees even lines, only the final
    // partial cell differs, which is off-screen-edge invisible.
    return target;
}

// ------------------------------ beam profile -------------------------------

// Gaussian-ish beam falloff. `d` is distance from the beam centre in cell
// units (0 .. 0.5), `w` is beam width.
float beam(float d, float w) {
    float x = d / w;
    return exp(-x * x * 2.0);
}

void main() {
    ivec2 res = textureSize(tex, 0);
    vec2 texel = 1.0 / vec2(res);

    int pitch = scanlinePitch(res.y);

    // ---- sample the source with an asymmetric CRT beam spot ---------------
    // 3x3 separable tap set weighted to smear horizontally, because a real
    // CRT spot is stretched along the sweep. All taps are linearized before
    // mixing so the blur is energy-correct.
    vec3 lin = vec3(0.0);
    float wsum = 0.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 off = vec2(float(x) * BLUR_X, float(y) * BLUR_Y) * texel;
            float wx = exp(-float(x * x) * 0.85);
            float wy = exp(-float(y * y) * 2.60);
            float w = wx * wy;
            lin += srgbToLinear(texture(tex, v_texcoord + off).rgb) * w;
            wsum += w;
        }
    }
    lin /= wsum;

    // ---- glow / bloom -----------------------------------------------------
    // Ring of taps around the pixel; only light above GLOW_THRESHOLD blooms,
    // so bright text and highlights halo while dark UI stays clean. Additive
    // in linear space, which is how real phosphor bleed behaves.
#if USE_GLOW
    // Sample several radii per direction with a Gaussian weight that peaks at
    // the source and decays outward.
    //
    // The previous version sampled only two fixed radii (5px and 2.5px) with
    // the *outer* ring weighted lower but nothing at all sampled inside
    // 2.5px. That left a hole in the middle of the bloom and parked its energy
    // in a band a few pixels away from bright pixels — which is precisely what
    // reads as a hard "halo" ring instead of a glow hugging the source.
    // Grading the radii and weighting near taps highest gives a monotonically
    // decaying falloff, so bright text gets a soft bleed rather than an
    // outline.
    vec3 glow = vec3(0.0);
    float glowWsum = 0.0;
    for (int i = 0; i < GLOW_TAPS; i++) {
        float a = 6.2831853 * (float(i) + 0.5) / float(GLOW_TAPS);
        vec2 dir = vec2(cos(a), sin(a));
        for (int r = 1; r <= GLOW_RINGS; r++) {
            float t = float(r) / float(GLOW_RINGS);       // 0..1 outward
            float radius = GLOW_RADIUS * t;
            float w = exp(-t * t * 2.2);                  // peaks near source
            vec3 s = srgbToLinear(texture(tex, v_texcoord + dir * radius * texel).rgb);
            glow += max(s - GLOW_THRESHOLD, vec3(0.0)) * w;
            glowWsum += w;
        }
    }
    glow /= max(glowWsum, 1e-4);
    // Cap the added halo. Two reasons for a hard ceiling: a large near-white
    // region sums enough over-threshold light to drive the frame into
    // clipping (washed-out screen rather than a glow), and an additive bloom
    // lifts *blacks* fastest in perceptual terms — uncapped, Heavy raised
    // pure black to ~0.6 sRGB, turning the desktop grey. 0.12 linear keeps
    // blacks near-black while still haloing bright text.
    lin += min(glow * float(GLOW_STRENGTH), vec3(0.12));
#endif

    // ---- contrast + saturation (in linear light) -------------------------
#if USE_CONTRAST
    {
        // Smoothstep S-curve pinned at 0 and 1, so pure black and pure white
        // survive instead of being crushed. Clamped first because the glow
        // pass above can push values past 1.0, and the raw cubic diverges
        // hard outside [0,1] — that overshoot is what turned Heavy white.
        vec3 c = clamp(lin, 0.0, 1.0);
        vec3 s = c * c * (3.0 - 2.0 * c);
        lin = mix(c, s, float(CONTRAST));
    }
#endif

#if USE_SATURATION
    {
        float l = luminance(lin);
        lin = max(mix(vec3(l), lin, float(SATURATION)), vec3(0.0));
    }
#endif

    // ---- warm phosphor cast ----------------------------------------------
    // Push toward amber. Applied as a per-channel gain normalised so overall
    // luminance is unchanged — the image gets warmer, not brighter.
#if USE_WARMTH
    {
        vec3 warm = vec3(1.14, 1.00, 0.78);
        warm /= luminance(warm);
        lin *= mix(vec3(1.0), warm, float(WARMTH));
    }
#endif

    // ---- scanline beam ----------------------------------------------------
    // Phase from the integer physical row: exact, never drifts.
    int row = int(gl_FragCoord.y);
    int phase = row % pitch;
    if (phase < 0) phase += pitch; // defensive; row is never negative in practice

    float o = float(phase) / float(pitch);
    // Distance to the nearest beam centre, wrapped, in cell units.
    float d = min(o, 1.0 - o);

    // Brighter content -> wider beam (highlight blooming).
    float lum = clamp(luminance(lin), 0.0, 1.0);
    float width = mix(float(BEAM_MIN), float(BEAM_MAX), lum);

    float w = beam(d, width);

    // Analytic normalisation: average beam weight across one whole cell, so
    // average luminance is preserved regardless of pitch or beam width.
    float wAvg = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= pitch) break;
        float oi = float(i) / float(pitch);
        wAvg += beam(min(oi, 1.0 - oi), width);
    }
    wAvg /= float(pitch);

    float scan = w / max(wAvg, 1e-4);
    lin *= mix(1.0, scan, float(SCANLINE_GAIN));

    // ---- aperture grille mask --------------------------------------------
#if USE_MASK
    {
        int col = int(gl_FragCoord.x);
        int sub = col % 3;
        if (sub < 0) sub += 3;

        vec3 phosphor = vec3(float(MASK_DIM));
        if (sub == 0) phosphor.r = 1.0;
        else if (sub == 1) phosphor.g = 1.0;
        else phosphor.b = 1.0;

        vec3 mask = mix(vec3(1.0), phosphor, float(MASK_STRENGTH));
        // Each channel is lit on 1 of 3 columns and dimmed on the other 2;
        // divide the exact triad average back out to hold luminance.
        float maskAvg = (1.0 + 2.0 * float(MASK_DIM)) / 3.0;
        mask /= mix(1.0, maskAvg, float(MASK_STRENGTH));
        lin *= mask;
    }
#endif

    // ---- vignette ---------------------------------------------------------
#if USE_VIGNETTE
    {
        vec2 p = v_texcoord * (1.0 - v_texcoord);
        float v = clamp(pow(p.x * p.y * 16.0, 0.28), 0.0, 1.0);
        lin *= mix(1.0, v, float(VIGNETTE));
    }
#endif

    lin *= float(BRIGHTNESS);

    fragColor = vec4(linearToSrgb(lin), 1.0);
}
