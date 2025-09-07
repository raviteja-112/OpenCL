#include <CL/cl.h>
#include <stdio.h>
#include <time.h>

const int SIZE = 2;

void takeinput(float A[SIZE][SIZE]) {  
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("Enter element %d %d: ", i, j);
            scanf("%f", &A[i][j]);
        }
    }
}

void matrix_multiplication(cl_context context, cl_command_queue queue, cl_program program,
                           cl_kernel kernel, cl_mem bufferA, cl_mem bufferB, cl_mem bufferC,
                           int M, int N, int K, double* gpu_time) {
    
    cl_event kernel_event, write_event;
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    // Run kernel with event profiling
    size_t globalSize[2] = {M, N};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 
                          0, NULL, &kernel_event);

    // Wait for kernel to finish
    clFinish(queue);

    // Get kernel execution time
    cl_ulong start, end;
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, 
                           sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, 
                           sizeof(cl_ulong), &end, NULL);
    
    *gpu_time = (end - start) * 1e-6; // Convert nanoseconds to milliseconds

    clReleaseEvent(kernel_event);
}

int main() {
    float A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
    double gpu_time = 0.0;

    takeinput(A);
    takeinput(B);
    
    // 1. Get platform and device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // 2. Create context and PROFILING queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 
                                                 CL_QUEUE_PROFILING_ENABLE, NULL); // ✅ Enable profiling

    // 3. Create buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    SIZE * SIZE * sizeof(float), A, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    SIZE * SIZE * sizeof(float), B, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    SIZE * SIZE * sizeof(float), NULL, NULL);

    // 4. Load kernel source
    FILE *fp = fopen("matrix.cl", "r");
    if (!fp) {
        printf("Failed to open kernel file\n");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char *source = (char*)malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &size, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrix_multiplication", NULL);

    // Measure GPU time
    matrix_multiplication(context, queue, program, kernel, bufferA, bufferB, bufferC, 
                         SIZE, SIZE, SIZE, &gpu_time);

    // Read result back
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, SIZE * SIZE * sizeof(float), C, 0, NULL, NULL);

    printf("\nResult Matrix C:\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%8.2f ", C[i][j]);
        }
        printf("\n");
    }

    printf("\nGPU Computation Time: %.3f ms\n", gpu_time);
    printf("GPU Computation Time: %.6f seconds\n", gpu_time / 1000.0);

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source);

    return 0;
}