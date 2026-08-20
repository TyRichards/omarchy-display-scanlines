// ---------------------------------------------------------------------------
// Display Scanlines — integer-locked CRT emulation for Hyprland
//
// This file is the shared shader BODY. It is not usable on its own: the
// `display-scanlines` CLI prepends a preamble that supplies `#version 300 es`
// and `#define CRT_LEVEL <1|2>` (1 = Light, 2 = Heavy), then writes
// the result to the generated active shader. Keeping one body means the two
// presets can never drift apart.
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
//     Beam and colour transforms are applied to a linearized source sample and
//     then re-encoded. Doing this in gamma space is what makes most CRT shaders
//     look muddy and dim; in linear light the highlights stay punchy and the
//     brightness compensation below is physically meaningful.
//
//  5. BRIGHTNESS COMPENSATION IS ANALYTIC.
//     Scanlines remove light. Rather than guessing a fudge factor, the average
//     beam weight over one cell is summed exactly and divided back out, so the
//     desktop keeps its intended luminance at any pitch.
//
//  6. DAMAGE-SAFE AND STATIC.
//     Every output pixel samples only its matching source pixel. Neighbouring
//     texture taps conflict with Hyprland's partial-damage rectangles and can
//     expose stale/background pixels as moving coloured boxes. Staying local
//     keeps damage tracking correct; omitting `time` also keeps idle desktops
//     from continuously redrawing.
// ---------------------------------------------------------------------------

precision highp float;

in vec2 v_texcoord;
uniform sampler2D tex;

layout(location = 0) out vec4 fragColor;

// ============================ level parameters =============================
// Two curated presets. Light is the subtle everyday setting; Heavy is an
// unapologetic novelty mode.

// NOTE ON THE `USE_*` FLAGS BELOW:
// The GLSL/C preprocessor evaluates `#if` using *integer* arithmetic only — a
// float literal in a conditional is invalid and can silently mis-evaluate on
// some drivers. Every optional pass is therefore gated by an integer flag.

#if CRT_LEVEL == 1
// ---- LIGHT: subtle, all-day usable. --------------------------------------
#define BEAM_MIN        0.34  // beam width in dark areas (cell units)
#define BEAM_MAX        0.58  // beam width in bright areas
#define SCANLINE_GAIN   0.94  // how fully scanlines are applied
#define CONTRAST        0.34  // S-curve strength around mid-grey
#define SATURATION      1.12  // phosphors are more vivid than an LCD
#define BRIGHTNESS      1.03  // final luminance trim
#define VIGNETTE        0.10  // corner falloff
#define WARMTH          0.00  // amber/phosphor tint
#define USE_CONTRAST    1
#define USE_SATURATION  1
#define USE_WARMTH      0
#define USE_VIGNETTE    1

#else
// ---- HEAVY: heavy-handed on purpose — warm cast, deep vignette and
// pitch-black gaps. Novelty mode. Uses the same scanline pitch as Light; the
// intensity comes from everything except line size.
// Beam width is what sets gap darkness, and it widens with content
// brightness — so BEAM_MAX (bright content) is the lever for "pitch black",
// not BEAM_MIN. At this 2px pitch, 0.24 puts the gap at 0.00/255 across the
// whole brightness range; the old 0.46 left it at ~115/255 (grey, not black).
#define BEAM_MIN        0.18  // pitch-black gaps
#define BEAM_MAX        0.24  // ... at bright content too
#define SCANLINE_GAIN   1.00
#define CONTRAST        0.34  // same as Light, by design
#define SATURATION      1.32
#define BRIGHTNESS      1.14
#define VIGNETTE        0.46  // darker vignette frame
#define WARMTH          0.26  // strong warm filter
#define USE_CONTRAST    1
#define USE_SATURATION  1
#define USE_WARMTH      1
#define USE_VIGNETTE    1
#endif

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
// they differ in beam profile and colour treatment, never in line spacing.
//
// Low-height displays need fewer, wider-spaced lines to keep desktop fonts
// readable: 1000 physical rows and below use a fixed 4px pitch (250 lines at
// 1000p, exactly half the old 2px/500-line density). Above that threshold the
// target scales with panel height: ~2px through 1600p, 3px at 4K, and 4px
// beyond. We prefer a pitch that divides the height exactly so the bottom cell
// isn't clipped.
//
// Mirrored by scanlinePitch() in Model.js for the panel's readout — verified
// to agree across 17 real panel heights.
int scanlinePitch(int h) {
    if (h <= 1000) return 4;

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
    int pitch = scanlinePitch(res.y);

    // ---- damage-safe source sample ---------------------------------------
    // Never sample outside this fragment. Hyprland redraws partial damage
    // rectangles and does not expand them for custom shader tap radii; blur or
    // glow taps can therefore pull stale/background pixels across a damage
    // boundary. One matching source sample keeps moving UI perfectly stable.
    vec3 lin = srgbToLinear(texture(tex, v_texcoord).rgb);

    // ---- contrast + saturation (in linear light) -------------------------
#if USE_CONTRAST
    {
        // Smoothstep S-curve pinned at 0 and 1, so pure black and pure white
        // survive instead of being crushed. Clamping also keeps the cubic
        // well-defined when a source surface arrives outside nominal range.
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

    // Brighter content -> wider beam (highlight blooming). At 1000 physical
    // rows and below, pitch doubles to 4px; floor the beam width so even Heavy
    // leaves three useful source rows around its single dark gap row.
    float lum = clamp(luminance(lin), 0.0, 1.0);
    float width = mix(float(BEAM_MIN), float(BEAM_MAX), lum);
    if (res.y <= 1000) width = max(width, 0.34);

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

    // Deliberately no per-column RGB aperture mask. A physical 3px RGB triad
    // aliases against LCD subpixels, fractional display scaling, browser zoom,
    // and screenshot resampling. That produces red/blue box ghosts around UI
    // text instead of a stable CRT effect. Horizontal beam modulation supplies
    // the scanline character without altering source hue from column to column.

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
