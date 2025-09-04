// check_opencl.c
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char *src =
"__kernel void add1(__global float* a) { int i = get_global_id(0); a[i] += 1.0f; }";

int main() {
    cl_platform_id plat;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel k;
    cl_int err;

    err = clGetPlatformIDs(1, &plat, NULL);
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);

    char name[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("Device: %s\n", name);

    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    q = clCreateCommandQueue(ctx, dev, 0, &err);

    float val = 42.0f;
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), &val, &err);

    prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    k = clCreateKernel(prog, "add1", &err);
    clSetKernelArg(k, 0, sizeof(cl_mem), &buf);

    size_t g = 1;
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &g, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(q, buf, CL_TRUE, 0, sizeof(float), &val, 0, NULL, NULL);

    printf("Result: %f (expected 43.0)\n", val);

    clReleaseMemObject(buf);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
