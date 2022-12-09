// TODO: override expressions (https://github.com/gfx-rs/wgpu/issues/1762)
let NUM_LAYERS: u32 = 4u;
let IMAGE_SIZE: f32 = 512.0;
let IMAGE_SCALE: f32 = 8.0;
//let IMAGE_SCALE: f32 = 32.0;
let POLYNOMIAL_FEATURES: bool = true;
let MAX_DIM: u32 = 16u;
let MAX_DIM_QUARTER: u32 = 4u; // TODO: const eval in naga?

struct Matrices {
    dims: array<u32, NUM_LAYERS>,
    weights: array<atomic<u32>>,
}

@group(0) @binding(0) var<storage, read> matrices: Matrices;
@group(1) @binding(0) var<uniform> time: f32;

@group(2) @binding(0) var<storage, read_write> write_matrices: Matrices;
@group(2) @binding(1) var<storage, read> target_image: array<vec3<f32>, 262144>;
//@group(2) @binding(2) var<uniform> alpha: f64;

//var<workgroup> foo: atomic<u32> = atomic<u32>(0u);
//var<workgroup> foo: u32 = 0u;

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

fn do_mm(x: array<f32, MAX_DIM>, m: u32, n: u32, offset: u32) -> array<f32, MAX_DIM> {
    // x is m by 1
    // weights is m by n
    // out is n by 1
    var out: array<f32, MAX_DIM> = array<f32, MAX_DIM>();
    // as of https://github.com/gpuweb/gpuweb/pull/1801, WGSL disallows indexing into a function parameter by a loop index
    var y: array<f32, MAX_DIM> = x;
    for(var j: u32 = 0u; j < n; j++) {
        for(var k: u32 = 0u; k < min(MAX_DIM, m); k++) {
            out[j] += bitcast<f32>(atomicLoad(&matrices.weights[offset + j + k * n])) * y[k];
        }
    }
    for(var k: u32 = 0u; k < MAX_DIM; k++) {
        out[k] = tanh(out[k]);
        //out[k] = max(0.0, out[k]);
    }
    return out;
}

fn matrix_vector_transpose_dot(x: array<f32, MAX_DIM>, m: u32, n: u32, offset: u32) -> array<f32, MAX_DIM> {
    var out: array<f32, MAX_DIM> = array<f32, MAX_DIM>();
    var y: array<f32, MAX_DIM> = x;
    for(var j: u32 = 0u; j < min(MAX_DIM, n); j++) {
        for(var k: u32 = 0u; k < min(MAX_DIM, m); k++) {
            out[k] += bitcast<f32>(atomicLoad(&matrices.weights[offset + j + k * n])) * y[j];
        }
    }
    return out;

}

fn do_mm4(x: array<vec4<f32>, MAX_DIM_QUARTER>, m: u32, n: u32, offset: u32) -> array<vec4<f32>, MAX_DIM_QUARTER> {
    // x is m by 1
    // weights is m by n
    // out is n by 1
    var out = array<vec4<f32>, MAX_DIM_QUARTER>();
    var y = x;
    let m = min(MAX_DIM, m);
    for(var j: u32 = 0u; j < n/4u; j++) {
        //let nn = min(4u, n - 4u*j);
        for(var k: u32 = 0u; k < m/4u; k++) {
            //let mm = min(4u, m - 4u*k);
            var z = mat4x4<f32>();
            // if we used nn/mm directly here, we'd get a slowdown relative to the unvectorized version due to excessive branching
            for(var i=0u; i<4u; i++) {
                for(var l=0u; l<4u; l++) {
                    z[l][i] = bitcast<f32>(atomicLoad(&matrices.weights[offset + (4u*j+i) + (4u*k+l) * n]));
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
                    z[l][i] = bitcast<f32>(atomicLoad(&matrices.weights[offset + (4u*j+i) + (4u*k+l) * n]));
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
                    z[l][i] = bitcast<f32>(atomicLoad(&matrices.weights[offset + (4u*j+i) + (4u*k+l) * n]));
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
                z[l][i] = bitcast<f32>(atomicLoad(&matrices.weights[offset + (4u*j+i) + (4u*k+l) * n]));
            }
        }
        out[j] += z * y[k];
    }
    for(var k: u32 = 0u; k < MAX_DIM_QUARTER; k++) {
        out[k] = tanh(out[k]);
    }
    return out;
}

@fragment
fn frag_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let u: f32 = IMAGE_SCALE * ((position.x / IMAGE_SIZE) - 0.5);
    let v: f32 = IMAGE_SCALE * ((position.y / IMAGE_SIZE) - 0.5);
    //var tmp: array<f32, MAX_DIM> = array<f32, MAX_DIM>(time, 1.0, u, v, u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v, 0.0, 0.0, 0.0, 0.0, 0.0);
    //var tmp: array<f32, MAX_DIM> = array<f32, MAX_DIM>(time, 1.0, u, v, cos(u), cos(v), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var tmp: array<vec4<f32>, MAX_DIM_QUARTER>;
    if POLYNOMIAL_FEATURES {
        tmp = array<vec4<f32>, MAX_DIM_QUARTER>(vec4(time, 1.0, u, v), vec4(u*u, u*v, v*v, u*u*u), vec4(u*u*v, u*v*v, v*v*v, 0.0), vec4<f32>());
    } else {
        tmp = array<vec4<f32>, MAX_DIM_QUARTER>(vec4(time, 1.0, u, v), vec4(cos(u), cos(v), 0.0, 0.0), vec4<f32>(), vec4<f32>());
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

@compute @workgroup_size(1)
fn backprop_gpu(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_id) local_invocation_id: vec3<u32>) {
    let alpha = 0.001;
    let position = IMAGE_SCALE * ((vec3<f32>(global_invocation_id) / IMAGE_SIZE) - 0.5);
    let u = position.x;
    let v = position.y;
    var fw: array<array<f32, MAX_DIM>, NUM_LAYERS> = array<array<f32, MAX_DIM>, NUM_LAYERS>();
    fw[0] = array<f32, MAX_DIM>(time, 1.0, u, v, u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v, 0.0, 0.0, 0.0, 0.0, 0.0);
    var offset: u32 = 0u;
    for(var i = 0u; i < (NUM_LAYERS - 1u); i++) {
        let m: u32 = matrices.dims[i];
        let n: u32 = matrices.dims[i+1u];
        fw[i+1u] = do_mm(fw[i], m, n, offset);
        offset += m * n;
    }
    var bk: array<f32, MAX_DIM> = array<f32, MAX_DIM>();
    var fw_prime: array<f32, MAX_DIM> = array<f32, MAX_DIM>();
    let pixel = target_image[global_invocation_id.y * u32(IMAGE_SIZE) + global_invocation_id.x];
    bk[0] = pixel[0];
    bk[1] = pixel[1];
    bk[2] = pixel[2];
    for(var i = 0u; i < matrices.dims[0]; i++) {
        fw[0][i] = tanh(fw[0][i]);
    }
    for(var i = 0u; i < matrices.dims[NUM_LAYERS - 1u]; i++) {
        bk[i] = fw[NUM_LAYERS - 1u][i] - bk[i];
    }
    for(var layer = 0u; layer < (NUM_LAYERS - 2u); layer++) {
        let layer_back = NUM_LAYERS - 1u - layer;
        let m: u32 = matrices.dims[layer_back - 1u];
        let n: u32 = matrices.dims[layer_back];
        for(var i = 0u; i < n; i++) {
            fw_prime[i] = 1.0 - pow(fw[layer_back][i], 2.0);
        }
        offset -= m * n;
        for(var i = 0u; i < m; i++) {
            for(var j = 0u; j < n; j++) {
                let weightPtr = &write_matrices.weights[offset + j + i * n];
                var old = atomicLoad(weightPtr);
                var exchanged = false;
                for(var k=0u; !exchanged /*&& k < 100u*/; k++) {
                    //let newValF32 = bitcast<f32>(old) + 1.0;
                    let newValF32 = bitcast<f32>(old) - alpha * bk[j] * fw[layer_back - 1u][i];
                    //let newValF32 = bitcast<f32>(old);
                    let newVal = bitcast<u32>(newValF32);
                    storageBarrier();
                    let result = atomicCompareExchangeWeak(weightPtr, old, newVal);
                    old = result.old_value;
                    exchanged = result.exchanged;
                }
            }
        }
        var bk_dot_w = matrix_vector_transpose_dot(bk, m, n, offset);
        //var bk_dot_w = array<f32, MAX_DIM>();
        for(var i = 0u; i < min(m, MAX_DIM); i++) {
            bk[i] = fw_prime[i] * bk_dot_w[i];
        }
    }
}
