#version 330 core

// Reactive orb fragment shader.
//
// State-driven emissive colour + cubic fresnel rim light. The rim
// intensity is amplified by displacement so audio-driven bumps catch
// extra highlight. Output is pre-postprocess; the bloom + chromatic
// aberration pass lives in the widget's framebuffer compositor.

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec3 v_view_dir;
in float v_displacement;

uniform vec3 u_color;         // emissive colour for the current state
uniform float u_intensity;    // global brightness multiplier in [0, 1]
uniform float u_time;         // for subtle animated highlights

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 V = normalize(v_view_dir);

    // Cubic fresnel: bright at glancing angles, dark facing camera.
    float ndv = clamp(dot(N, V), 0.0, 1.0);
    float rim = pow(1.0 - ndv, 3.0);

    // Highlight punch on displaced peaks: high-displacement vertices
    // catch more rim energy so audio impulses pop visually.
    rim *= 1.0 + clamp(v_displacement * 6.0, -0.4, 1.5);

    // Surface base: the state colour, gently breathing via the time
    // uniform so static states still show a heartbeat without audio.
    float breath = 0.92 + 0.08 * sin(u_time * 1.6);
    vec3 surface = u_color * (0.45 + 0.35 * ndv) * breath;

    // Rim adds emissive light tinted toward white-shifted state colour.
    vec3 rim_tint = mix(u_color, vec3(1.0), 0.35);
    vec3 emissive = rim_tint * rim * 1.4;

    vec3 final_rgb = (surface + emissive) * u_intensity;

    // Soft alpha falloff so the silhouette dissolves at extreme rim;
    // pairs with the framebuffer's premultiplied composite.
    float alpha = clamp(0.30 + 0.70 * (ndv + rim * 0.6), 0.0, 1.0);

    frag_color = vec4(final_rgb, alpha);
}
