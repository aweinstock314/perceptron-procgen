// TODO: override expressions (https://github.com/gfx-rs/wgpu/issues/1762)
let NUM_LAYERS: u32 = 4u;
let FORWARD_IMAGE_SIZE: f32 = 512.0;
let IMAGE_SCALE: f32 = 8.0;
//let IMAGE_SCALE: f32 = 32.0;
let POLYNOMIAL_FEATURES: bool = true;
let MAX_DIM: u32 = 16u;
let MAX_DIM_QUARTER: u32 = 4u; // TODO: const eval in naga?

struct Matrices {
    dims: array<u32, NUM_LAYERS>,
    weights: array<atomic<u32>>,
}

struct ScalarParams {
    time: f32,
    image_width: u32,
    image_height: u32,
}

struct MatrixDeltas {
    stride: u32,
    count: u32,
    frob_norms: array<atomic<u32>, NUM_LAYERS>,
    weights: array<atomic<u32>>,
}

@group(0) @binding(0) var<storage, read> matrices: Matrices;
@group(1) @binding(0) var<uniform> scalars: ScalarParams;

@group(2) @binding(0) var<storage, read_write> write_matrices: MatrixDeltas;
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
fn matrix_vector_transpose_dot4(x: array<vec4<f32>, MAX_DIM_QUARTER>, m: u32, n: u32, offset: u32) -> array<vec4<f32>, MAX_DIM_QUARTER> {
    var out: array<vec4<f32>, MAX_DIM_QUARTER> = array<vec4<f32>, MAX_DIM_QUARTER>();
    var y: array<vec4<f32>, MAX_DIM_QUARTER> = x;
    for(var j: u32 = 0u; j < n; j++) {
        for(var k: u32 = 0u; k < m; k++) {
            out[k/4u][k%4u] += bitcast<f32>(atomicLoad(&matrices.weights[offset + j + k * n])) * y[j/4u][j%4u];
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

fn frobenius(m: u32, n: u32, offset: u32) -> f32 {
    var sum = 0.0;
    let d = min(m, n);
    for(var i = 0u; i < d; i++) {
        for(var j = 0u; j < d; j++) {
            sum += pow(bitcast<f32>(atomicLoad(&matrices.weights[offset + i + j * n])), 2.0);
        }
    }
    return sqrt(sum);
}

@fragment
fn frag_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let u: f32 = IMAGE_SCALE * ((position.x / FORWARD_IMAGE_SIZE) - 0.5);
    let v: f32 = IMAGE_SCALE * ((position.y / FORWARD_IMAGE_SIZE) - 0.5);
    //var tmp: array<f32, MAX_DIM> = array<f32, MAX_DIM>(scalars.time, 1.0, u, v, u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v, 0.0, 0.0, 0.0, 0.0, 0.0);
    //var tmp: array<f32, MAX_DIM> = array<f32, MAX_DIM>(scalars.time, 1.0, u, v, cos(u), cos(v), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var tmp: array<vec4<f32>, MAX_DIM_QUARTER>;
    if POLYNOMIAL_FEATURES {
        tmp = array<vec4<f32>, MAX_DIM_QUARTER>(vec4(scalars.time, 1.0, u, v), vec4(u*u, u*v, v*v, u*u*u), vec4(u*u*v, u*v*v, v*v*v, 0.0), vec4<f32>());
    } else {
        tmp = array<vec4<f32>, MAX_DIM_QUARTER>(vec4(scalars.time, 1.0, u, v), vec4(cos(u), cos(v), 0.0, 0.0), vec4<f32>(), vec4<f32>());
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
fn calc_norms() {
    var offset: u32 = 0u;
    for(var i = 0u; i < (NUM_LAYERS - 1u); i++) {
        let m: u32 = matrices.dims[i];
        let n: u32 = matrices.dims[i+1u];
        atomicStore(&write_matrices.frob_norms[i+1u], bitcast<u32>(frobenius(m, n, offset)));
        offset += m * n;
    }
}

@compute @workgroup_size(16)
fn backprop_gpu(@builtin(global_invocation_id) global_invocation_id: vec3<u32>, @builtin(local_invocation_id) local_invocation_id: vec3<u32>) {
    let alpha = 0.001 / sqrt(f32(scalars.image_width) * f32(scalars.image_height));
    let lambda = 0.001;
    let position = IMAGE_SCALE * ((vec3<f32>(global_invocation_id) / vec3<f32>(f32(scalars.image_width), f32(scalars.image_height), 1.0)) - 0.5);
    let multiplicity_offset = ((3u * global_invocation_id.x + 5u * global_invocation_id.y) % write_matrices.count) * write_matrices.stride;
    let u = position.x;
    let v = position.y;
    var fw: array<array<vec4<f32>, MAX_DIM_QUARTER>, NUM_LAYERS> = array<array<vec4<f32>, MAX_DIM_QUARTER>, NUM_LAYERS>();
    //fw[0] = array<f32, MAX_DIM>(scalars.time, 1.0, u, v, u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v, 0.0, 0.0, 0.0, 0.0, 0.0);
    fw[0] = array<vec4<f32>, MAX_DIM_QUARTER>(vec4(scalars.time, 1.0, u, v), vec4(u*u, u*v, v*v, u*u*u), vec4(u*u*v, u*v*v, v*v*v, 0.0), vec4<f32>());
    var offset: u32 = 0u;
    for(var i = 0u; i < (NUM_LAYERS - 1u); i++) {
        let m: u32 = matrices.dims[i];
        let n: u32 = matrices.dims[i+1u];
        fw[i+1u] = do_mm4(fw[i], m, n, offset);
        offset += m * n;
    }
    var bk: array<vec4<f32>, MAX_DIM_QUARTER> = array<vec4<f32>, MAX_DIM_QUARTER>();
    var fw_prime: array<vec4<f32>, MAX_DIM_QUARTER> = array<vec4<f32>, MAX_DIM_QUARTER>();
    let pixel = target_image[global_invocation_id.y * scalars.image_width + global_invocation_id.x];
    bk[0] = vec4(pixel[0], pixel[1], pixel[2], 0.0);
    let sz = matrices.dims[0];
    for(var i = 0u; i < sz/4u; i++) {
        fw[0][i] = tanh(fw[0][i]);
    }
    for(var i = 0u; i < sz%4u; i++) {
        fw[0][sz/4u][i] = tanh(fw[0][sz/4u][i]);
    }
    let sz = matrices.dims[NUM_LAYERS - 1u];
    for(var i = 0u; i < sz/4u; i++) {
        bk[i] = fw[NUM_LAYERS - 1u][i] - bk[i];
    }
    for(var i = 0u; i < sz%4u; i++) {
        bk[sz/4u][i] = fw[NUM_LAYERS - 1u][sz/4u][i] - bk[sz/4u][i];
    }
    for(var layer = 0u; layer < (NUM_LAYERS - 1u); layer++) {
        let layer_back = NUM_LAYERS - 1u - layer;
        let m: u32 = matrices.dims[layer_back - 1u];
        let n: u32 = matrices.dims[layer_back];
        for(var i = 0u; i < n/4u; i++) {
            fw_prime[i] = 1.0 - fw[layer_back][i] * fw[layer_back][i];
        }
        for(var i = 0u; i < n%4u; i++) {
            fw_prime[n/4u][i] = 1.0 - fw[layer_back][n/4u][i] * fw[layer_back][n/4u][i];
        }
        offset -= m * n;
        for(var i = 0u; i < m; i++) {
            for(var j = 0u; j < n; j++) {
                let weightPtr = &write_matrices.weights[multiplicity_offset + offset + j + i * n];
                //storageBarrier();
                var old = atomicLoad(weightPtr);
                var exchanged = false;
                for(var k=0u; !exchanged /*&& k < 100u*/; k++) {
                    //let newValF32 = bitcast<f32>(old) + 1.0;
                    let newValF32 = bitcast<f32>(old) - alpha * bk[j/4u][j%4u] * fw[layer_back - 1u][i/4u][i%4u];
                    //let newValF32 = bitcast<f32>(old);
                    let newVal = bitcast<u32>(newValF32);
                    let result = atomicCompareExchangeWeak(weightPtr, old, newVal);
                    old = result.old_value;
                    exchanged = result.exchanged;
                }
            }
        }
        var bk_dot_w = matrix_vector_transpose_dot4(bk, m, n, offset);
        //var bk_dot_w = array<f32, MAX_DIM>();
        let frob_error = lambda * bitcast<f32>(atomicLoad(&write_matrices.frob_norms[layer_back]));
        for(var i = 0u; i < m/4u; i++) {
            bk[i] = fw_prime[i] * bk_dot_w[i] + lambda * frob_error;
        }
        for(var i = 0u; i < m%4u; i++) {
            bk[m/4u][i] = fw_prime[m/4u][i] * bk_dot_w[m/4u][i] + lambda * frob_error;
        }
    }
}

@compute @workgroup_size(1)
fn sum_weights() {
    for(var i = 0u; i < write_matrices.stride; i++) {
        let x = atomicLoad(&matrices.weights[i]);
        let y = atomicLoad(&write_matrices.weights[i]);
        atomicStore(&write_matrices.weights[i], bitcast<u32>(bitcast<f32>(x) + bitcast<f32>(y)));
    }
    for(var k = 1u; k < write_matrices.count; k++) {
        for(var i = 0u; i < write_matrices.stride; i++) {
            let x = atomicLoad(&write_matrices.weights[k * write_matrices.stride + i]);
            let y = atomicLoad(&write_matrices.weights[i]);
            atomicStore(&write_matrices.weights[i], bitcast<u32>(bitcast<f32>(x) + bitcast<f32>(y)));
        }
    }
}
