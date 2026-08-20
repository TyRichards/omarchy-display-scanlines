#version 300 es
// Neutral handoff used briefly before unloading the CRT shader. Keeping the
// final-shader pipeline active while Hyprland presents full clean frames avoids
// exposing striped buffer history during the transition to no screen shader.

precision highp float;

in vec2 v_texcoord;
uniform sampler2D tex;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = texture(tex, v_texcoord);
}
