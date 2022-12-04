@vertex
fn vert_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    switch vertex_index {
        case 0u: { return vec4<f32>(-1.0, -1.0, 0.0, 1.0); }
        case 1u: { return vec4<f32>(-1.0, 1.0, 0.0, 1.0); }
        case 2u: { return vec4<f32>(1.0, -1.0, 0.0, 1.0); }
        case 3u: { return vec4<f32>(-1.0, 1.0, 0.0, 1.0); }
        case 4u: { return vec4<f32>(1.0, -1.0, 0.0, 1.0); }
        default: { return vec4<f32>(1.0, 1.0, 0.0, 1.0); }
    }
}

@fragment
fn frag_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(position.x, position.y, 0.0, 1.0);
}
