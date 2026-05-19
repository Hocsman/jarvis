#version 330 core

// Reactive orb vertex shader.
//
// Inputs: unit-sphere position + normal (geometry.build_icosphere).
// Outputs: displaced world-space position to the fragment stage along
// with the world normal and the view direction so the rim-light /
// fresnel pass can run per-pixel.
//
// The displacement is layered:
//   - a slow Simplex 3D noise (low frequency) drives the overall
//     breathing wobble,
//   - a faster octave modulated by the audio bands (bass / mid / high)
//     gives the speech-reactive surface motion,
//   - the audio amplitude (u_amplitude, the EMA-smoothed RMS) gates
//     the whole thing so silence collapses back to the bare sphere.

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

uniform mat4 u_mvp;        // model-view-projection
uniform mat4 u_model;      // for transforming the normal to world space
uniform float u_time;      // seconds since first frame
uniform float u_bass;      // EMA-smoothed band 0-250 Hz, in [0, 1]
uniform float u_mid;       // EMA-smoothed band 250-2000 Hz, in [0, 1]
uniform float u_high;      // EMA-smoothed band 2000+ Hz, in [0, 1]
uniform float u_amplitude; // EMA-smoothed global RMS, in [0, 1]
uniform float u_displacement_scale;  // master gain, set per-state by the controller

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec3 v_view_dir;
out float v_displacement;  // forwarded to fragment for rim modulation

// ── 3D Simplex noise (Stefan Gustavson port, public domain) ───────────
//
// Compact reference impl. Returns a single scalar in roughly [-1, 1].
// We use it twice with different frequencies to get a slow base wobble
// and a faster audio-reactive layer.

vec4 _mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 _mod289_v3(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 _permute(vec4 x) { return _mod289(((x * 34.0) + 1.0) * x); }
vec4 _taylor_inv_sqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float simplex3(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = _mod289_v3(i);
    vec4 p = _permute(_permute(_permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = _taylor_inv_sqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

void main() {
    // Slow breathing layer: independent of audio.
    float slow = simplex3(in_position * 1.7 + vec3(0.0, 0.0, u_time * 0.25));

    // Fast audio-reactive layer: frequency rises with high band, amplitude
    // rises with bass + mid. Empirical mix favours bass for a heart-beat feel.
    float audio_drive = (u_bass * 1.2 + u_mid * 0.8 + u_high * 0.4);
    float fast = simplex3(in_position * (3.5 + u_high * 2.0) + vec3(u_time * (0.8 + u_mid * 1.6), 0.0, 0.0));

    float disp = slow * 0.05 + fast * 0.10 * audio_drive;
    disp *= u_displacement_scale;
    disp *= (0.4 + 0.6 * u_amplitude);  // silence collapses the surface

    vec3 displaced = in_position + in_normal * disp;
    vec4 world = u_model * vec4(displaced, 1.0);
    v_world_pos = world.xyz;
    v_world_normal = mat3(u_model) * in_normal;
    v_view_dir = normalize(-world.xyz);   // assuming camera at origin in view space
    v_displacement = disp;

    gl_Position = u_mvp * vec4(displaced, 1.0);
}
