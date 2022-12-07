// TODO: override expressions (https://github.com/gfx-rs/wgpu/issues/1762)
let NUM_LAYERS: u32 = 4u;
let IMAGE_SIZE: f32 = 512.0;
let IMAGE_SCALE: f32 = 8.0;
//let IMAGE_SCALE: f32 = 32.0;
let POLYNOMIAL_FEATURES: bool = true;

struct Matrices {
    dims: array<u32, NUM_LAYERS>,
    weights: array<f32>,
}

@group(0) @binding(0) var<storage, read> matrices: Matrices;
@group(1) @binding(0) var<uniform> time: f32;

@vertex
fn vert_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // TODO: triangle strips
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
        //out[k] = max(0.0, out[k]);
    }
    return out;
}

fn do_mm4(x: array<vec4<f32>, 4>, m: u32, n: u32, offset: u32) -> array<vec4<f32>, 4> {
    // x is m by 1
    // weights is m by n
    // out is n by 1
    var out = array<vec4<f32>, 4>();
    let m = min(16u, m);
    var y = x;
    for(var j: u32 = 0u; j < n/4u; j++) {
        //let nn = min(4u, n - 4u*j);
        for(var k: u32 = 0u; k < m/4u; k++) {
            //let mm = min(4u, m - 4u*k);
            var z = mat4x4<f32>();
            // if we used nn/mm directly here, we'd get a slowdown relative to the unvectorized version due to excessive branching
            for(var i=0u; i<4u; i++) {
                for(var l=0u; l<4u; l++) {
                    z[l][i] = matrices.weights[offset + (4u*j+i) + (4u*k+l) * n];
                }
            }
            out[j] += z * y[k];
        }
    }
    // handle the remainder if the matrix size isn't a multiple of 4
    if(n % 4u != 0u) {
        let j = n/4u;
        let nn = min(4u, n - 4u*j);
        for(var k: u32 = 0u; k < m/4u; k++) {
            var z = mat4x4<f32>();
            for(var i=0u; i<nn; i++) {
                for(var l=0u; l<4u; l++) {
                    z[l][i] = matrices.weights[offset + (4u*j+i) + (4u*k+l) * n];
                }
            }
            out[j] += z * y[k];
        }
    }
    if(m % 4u != 0u) {
        let k = m/4u;
        let mm = min(4u, m - 4u*k);
        for(var j: u32 = 0u; j < n/4u; j++) {
            var z = mat4x4<f32>();
            for(var i=0u; i<4u; i++) {
                for(var l=0u; l<mm; l++) {
                    z[l][i] = matrices.weights[offset + (4u*j+i) + (4u*k+l) * n];
                }
            }
            out[j] += z * y[k];
        }
    }
    if(n % 4u != 0u && m % 4u != 0u) {
        let j = n/4u;
        let k = m/4u;
        let nn = min(4u, n - 4u*j);
        let mm = min(4u, m - 4u*k);
        var z = mat4x4<f32>();
        for(var i=0u; i<nn; i++) {
            for(var l=0u; l<mm; l++) {
                z[l][i] = matrices.weights[offset + (4u*j+i) + (4u*k+l) * n];
            }
        }
        out[j] += z * y[k];
    }
    for(var k: u32 = 0u; k < 4u; k++) {
        out[k] = tanh(out[k]);
    }
    return out;
}

@fragment
fn frag_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let u: f32 = IMAGE_SCALE * ((position.x / IMAGE_SIZE) - 0.5);
    let v: f32 = IMAGE_SCALE * ((position.y / IMAGE_SIZE) - 0.5);
    //var tmp: array<f32, 16> = array<f32, 16>(time, 1.0, u, v, u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v, 0.0, 0.0, 0.0, 0.0, 0.0);
    //var tmp: array<f32, 16> = array<f32, 16>(time, 1.0, u, v, cos(u), cos(v), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var tmp: array<vec4<f32>, 4>;
    if POLYNOMIAL_FEATURES {
        tmp = array<vec4<f32>, 4>(vec4(time, 1.0, u, v), vec4(u*u, u*v, v*v, u*u*u), vec4(u*u*v, u*v*v, v*v*v, 0.0), vec4<f32>());
    } else {
        tmp = array<vec4<f32>, 4>(vec4(time, 1.0, u, v), vec4(cos(u), cos(v), 0.0, 0.0), vec4<f32>(), vec4<f32>());
    }
    var offset: u32 = 0u;
    for(var matrix_index: u32 = 0u; matrix_index < (NUM_LAYERS - 1u); matrix_index++) {
        let m: u32 = matrices.dims[matrix_index];
        let n: u32 = matrices.dims[matrix_index+1u];
        //tmp = do_mm(tmp, m, n, offset);
        tmp = do_mm4(tmp, m, n, offset);
        offset += m * n;
    }

    //let tmp2 = vec4<f32>(tmp[0], tmp[1], tmp[2], 1.0);
    var tmp2 = tmp[0];
    tmp2[3] = 1.0;
    return (tmp2 + 1.0) / 2.0;
}
