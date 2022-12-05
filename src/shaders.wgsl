struct Matrices {
    dims: array<u32, 4>,
    weights: array<f32>,
}

@group(0) @binding(0) var<storage, read> matrices: Matrices;
@group(1) @binding(0) var<uniform> time: f32;

@vertex
fn vert_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    switch vertex_index {
        case 0u: { return vec4<f32>(-1.0, -1.0, 0.0, 1.0); }
        case 1u: { return vec4<f32>(1.0, -1.0, 0.0, 1.0); }
        case 2u: { return vec4<f32>(-1.0, 1.0, 0.0, 1.0); }
        case 3u: { return vec4<f32>(-1.0, 1.0, 0.0, 1.0); }
        case 4u: { return vec4<f32>(1.0, -1.0, 0.0, 1.0); }
        default: { return vec4<f32>(1.0, 1.0, 0.0, 1.0); }
    }
}

fn do_mm(x: array<f32, 16>, m: u32, n: u32, offset: u32) -> array<f32, 16> {
    // x is m by 1
    // weights is m by n
    // out is n by 1
    var out: array<f32, 16> = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // TODO: why does naga disallow indexing into a function parameter by a loop index?
    var y: array<f32, 16> = x;
    for(var j: u32 = 0u; j < n; j++) {
        for(var k: u32 = 0u; k < min(16u, m); k++) {
            out[j] += matrices.weights[offset + j + k * n] * y[k];
        }
    }
    for(var k: u32 = 0u; k < 16u; k++) {
        out[k] = tanh(out[k]);
    }
    return out;
}

@fragment
fn frag_main(@builtin(position) position: vec4<f32>, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    var u: f32 = 8.0 * ((position.x / 512.0) - 0.5);
    var v: f32 = 8.0 * ((position.y / 512.0) - 0.5);
    var tmp: array<f32, 16> = array<f32, 16>(time, 1.0, u, v, u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v, 0.0, 0.0, 0.0, 0.0, 0.0);
    var offset: u32 = 0u;
    for(var matrix_index: u32 = 0u; matrix_index < 3u; matrix_index++) {
        var m: u32 = matrices.dims[matrix_index];
        var n: u32 = matrices.dims[matrix_index+1u];
        tmp = do_mm(tmp, m, n, offset);
        offset += m * n;
    }
    for(var k: u32 = 0u; k < 3u; k++) {
        tmp[k] = (tmp[k] + 1.0) / 2.0;
    }

    return vec4<f32>(tmp[0], tmp[1], tmp[2], 1.0);
    /*if front_facing {
        return vec4<f32>(u, v, 0.0, 1.0);
    } else {
        return vec4<f32>(0.0, u, v, 1.0);
    }*/
}
