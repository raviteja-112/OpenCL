__kernel void vector_add(__global const int* A,
                         __global const int* B,
                         __global int* C) {
    int id = get_global_id(0);   // each work-item gets a unique ID
    C[id] = A[id] + B[id];       // compute for "its own" element
}
