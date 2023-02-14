//
//  main.cpp
//  Lab6
//
//  Created by Adrian Daniel Bodirlau on 15/12/2022.
//

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <OpenCL/OpenCL.h>
#include <cmath>
using namespace std;

// Function to display error messages
void displayError(cl_int errorCode, string message){
    // Display error message and exit
    if(errorCode!=CL_SUCCESS){
        cerr<<"Error: "<<message<<" ("<<errorCode<<")"<<endl;
        exit(EXIT_FAILURE);
    }
}

// Function to execute nbody on the CPU
void nbody_cpu(int NParticles, int NSteps, float eps, float dt, cl_float4* pos, cl_float4* vel){
    // Calculate the force between two particles
    auto force = [](cl_float4 p1, cl_float4 p2, float eps){
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        float r2 = dx*dx + dy*dy + dz*dz + eps*eps;
        float r = sqrt(r2);
        float f = 1.0/(r2*r);
        return f;
    };
    // Calculate the acceleration of a particle
    auto acceleration = [&](int i){
        cl_float4 acc = {0.0, 0.0, 0.0, 0.0};
        for(int j=0; j<NParticles; j++){
            if(i!=j){
                float f = force(pos[i], pos[j], eps);
                acc.x += f*(pos[j].x - pos[i].x);
                acc.y += f*(pos[j].y - pos[i].y);
                acc.z += f*(pos[j].z - pos[i].z);
            }
        }
        return acc;
    };
    // Calculate the velocity of a particle
    auto velocity = [&](int i, cl_float4 acc){
        vel[i].x += acc.x*dt;
        vel[i].y += acc.y*dt;
        vel[i].z += acc.z*dt;
    };
    // Calculate the position of a particle
    auto position = [&](int i){
        pos[i].x += vel[i].x*dt;
        pos[i].y += vel[i].y*dt;
        pos[i].z += vel[i].z*dt;
    };
    // Calculate the new position and velocity of the particles
    for(int step=0; step<NSteps; step++){
        for(int i=0; i<NParticles; i++){
            cl_float4 acc = acceleration(i);
            velocity(i, acc);
            position(i);
        }
    }
}

void nbody_init(int n, cl_float4* pos, cl_float4* vel){
    // Initialize the particles with fixed positions and fixed velocities
    for(int i=0; i<n; i++){
        pos[i].x = i;
        pos[i].y = i;
        pos[i].z = i;
        pos[i].w = 1.0; // Mass always 1.0
        // Velocity at 0.0
        vel[i].x = 0.0;
        vel[i].y = 0.0;
        vel[i].z = 0.0;
        vel[i].w = 0.0;
    }
}

void nbody_output(int n, cl_float4* pos, cl_float4* vel){
    // Outputting the particle's position and velocities
    for(int i=0; i<n; i++){
        cout<<setprecision(3)<<fixed;
        cout<<"Particle "<<i<<": ("<<pos[i].x<<", "<<pos[i].y<<", "<<pos[i].z<<")\t";
        cout<<"("<<vel[i].x<<", "<<vel[i].y<<", "<<vel[i].z<<")"<<endl;
    }
}

int main(int argc, const char * argv[]) {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    // Create platforms
    cl_int errorCode = clGetPlatformIDs(1, &platform, nullptr);
    displayError(errorCode, "Could not get platforms");
    // Create device
    errorCode = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    displayError(errorCode, "Could not get devices");
    
    // Get device max group size
    //    size_t size;
    //    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &size);
    //    int value[size];
    //    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, size, value, NULL);
    //    for (int i = 0; i < size; i++) {
    //        cout << value[i] << endl;
    //    }
    
    // Create context with properties
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    context = clCreateContext(properties, 1, &device, NULL, NULL, &errorCode);
    displayError(errorCode, "Could not create context");
    // Create commands with properties enabling profiling
    cl_command_queue_properties commandsProperties = CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_PRIORITY_DEFAULT_APPLE;
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, commandsProperties, &errorCode);
    displayError(errorCode, "Could not create a command queue");
    
    // Get the kernel file as plain C string
    ifstream kernelFile{"source.CL"};
    string kernelString((istreambuf_iterator<char>(kernelFile)), istreambuf_iterator<char>());
    const char* kernelSrc = kernelString.c_str();
    kernelFile.close();
    
    // Create the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSrc, nullptr, &errorCode);
    displayError(errorCode, "Could not create program");
    // Build the program
    errorCode = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    // If an error occurs
    if(errorCode!=CL_SUCCESS){
        // Get the program log and exit
        cout<<"Failed to build the program"<<endl;
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        char* log = new char[logSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        cerr<<log<<endl;
        delete[] log;
        exit(EXIT_FAILURE);
    }
    // Create the kernel
    cl_kernel tilesKernel = clCreateKernel(program, "nbody_tiles", &errorCode);
    displayError(errorCode, "Failed to create kernel");
    
    // Step and burst parameters, step used for the number of iterations and
    // burst for the number of particles to be computed in each iteration
    // before needing to read the data back from the GPU
    int NSteps {200};
    int NBursts {10};
    // Number of particles, power of 2
    int NParticles = 8192*2;
    // Derivative of the time step
    float dt = 0.01;
    // Epsilon constant to avoid division by zero
    float eps = 0.01;
    // Local work size
    size_t localWorkSize = 128;
    // Global work size
    size_t globalWorkSize = NParticles;
    // Calculate the number of tiles
    int tiles = NParticles/localWorkSize;
    
    // Array for the particles
    cl_float4 *inArray = new cl_float4[NParticles];
    cl_float4 *outArray = new cl_float4[NParticles];
    cl_float4 *velArray = new cl_float4[NParticles];
    
    // Randomly initialize the particles
    nbody_init(NParticles, inArray, velArray);
    
    // Write the input buffer to the device
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NParticles*sizeof(cl_float4), inArray, &errorCode);
    displayError(errorCode, "Could not create input buffer");
    // Write the output buffer to the device
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NParticles*sizeof(cl_float4), outArray, &errorCode);
    displayError(errorCode, "Could not create output buffer");
    // Write the velocity buffer to the device
    cl_mem velocityBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NParticles*sizeof(cl_float4), velArray, &errorCode);
    displayError(errorCode, "Could not create velocity buffer");
    // Set the kernel arguments
    errorCode = clSetKernelArg(tilesKernel, 0, sizeof(float), &dt);
    displayError(errorCode, "Could not set kernel argument 0");
    errorCode = clSetKernelArg(tilesKernel, 1, sizeof(float), &eps);
    displayError(errorCode, "Could not set kernel argument 1");
    errorCode = clSetKernelArg(tilesKernel, 4, sizeof(cl_mem), &velocityBuffer);
    displayError(errorCode, "Could not set kernel argument 4");
    errorCode = clSetKernelArg(tilesKernel, 5, sizeof(int), &tiles);
    displayError(errorCode, "Could not set kernel argument 5");
    // Initialize local memory
    errorCode = clSetKernelArg(tilesKernel, 6, localWorkSize*sizeof(cl_float4), nullptr);
    
    // Display the particles before the simulation
//     nbody_output(NParticles, inArray, velArray);
    cl_event event;
    // Run the kernel for step iterations and in burst mode
    double totalElapsedTime = 0.0;
    int numSteps = 0;
    
    for(int step=0; step<NSteps; step+=NBursts){
        // For loop running in bursts
        for(int burst=0; burst<NBursts; burst++){
            // Set the kernel arguments
            clSetKernelArg(tilesKernel, 2, sizeof(cl_mem), &inputBuffer);
            clSetKernelArg(tilesKernel, 3, sizeof(cl_mem), &outputBuffer);
            clEnqueueNDRangeKernel(commandQueue, tilesKernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, &event);
            
            // Wait for the event object to complete
            clWaitForEvents(1, &event);
            
            // Get the elapsed time in nanoseconds
            cl_ulong startTime, endTime;
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(startTime), &startTime, nullptr);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(endTime), &endTime, nullptr);
            
            // Convert the elapsed time to seconds and add it to the total elapsed time
            double elapsedTime = (double)(endTime - startTime);
            totalElapsedTime += elapsedTime;
            numSteps++;
            // Swap the buffers
            cl_mem temp = inputBuffer;
            inputBuffer = outputBuffer;
            outputBuffer = temp;
        }
        //        // Read the output buffer back to the host each NBursts
        //        errorCode = clEnqueueReadBuffer(commandQueue, inputBuffer, CL_TRUE, 0, NParticles*sizeof(cl_float4), inBuffer, 0, nullptr, nullptr);
        //        displayError(errorCode, "Could not read output buffer");
        //        // Read the velocity buffer back to the host
        //        errorCode = clEnqueueReadBuffer(commandQueue, velocityBuffer, CL_TRUE, 0, NParticles*sizeof(cl_float4), velBuffer, 0, nullptr, nullptr);
        //        displayError(errorCode, "Could not read velocity buffer");
    }
    cout<<"Steps: "<<numSteps<<endl;
    // Compute the average elapsed time
    double averageElapsedTime = totalElapsedTime / (double)numSteps;
    
    // Print the average elapsed time
    cout << "Average elapsed time: "<<setprecision(2) << averageElapsedTime << " seconds." << endl;
    
    // Read the output buffer back to the host
    errorCode = clEnqueueReadBuffer(commandQueue, inputBuffer, CL_TRUE, 0, NParticles*sizeof(cl_float4), inArray, 0, nullptr, nullptr);
    displayError(errorCode, "Could not read output buffer");
    // Read the velocity buffer back to the host
    errorCode = clEnqueueReadBuffer(commandQueue, velocityBuffer, CL_TRUE, 0, NParticles*sizeof(cl_float4), velArray, 0, nullptr, nullptr);
    displayError(errorCode, "Could not read velocity buffer");
    
    // Display the particles after the simulation to see the results
    nbody_output(NParticles, inArray, velArray);
    
    
    // N Kernel simulation, no local memory
    // Intialize the particles again
    nbody_init(NParticles, inArray, velArray);
    
    cl_kernel kernelN = clCreateKernel(program, "nbody_N", &errorCode);
    displayError(errorCode, "Could not create kernel");
    
    // Write the data to the buffers again
    errorCode = clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, sizeof(cl_float4)*NParticles, inArray, 0, NULL, NULL);
    displayError(errorCode, "Could not create input buffer");
    errorCode = clEnqueueWriteBuffer(commandQueue, velocityBuffer, CL_TRUE, 0, sizeof(cl_float4)*NParticles, velArray, 0, NULL, NULL);
    displayError(errorCode, "Could not create velocity buffer");
    
    // Set arguments for the kernel
    errorCode = clSetKernelArg(kernelN, 0, sizeof(float), &dt);
    displayError(errorCode, "Could not set kernel argument 0");
    errorCode = clSetKernelArg(kernelN, 1, sizeof(float), &eps);
    displayError(errorCode, "Could not set kernel argument 1");
    errorCode = clSetKernelArg(kernelN, 4, sizeof(cl_mem), &velocityBuffer);
    displayError(errorCode, "Could not set kernel argument 4");
    errorCode = clSetKernelArg(kernelN, 5, sizeof(int), &NParticles);
    displayError(errorCode, "Could not set kernel argument 5");
    cl_event event2;
    numSteps = 0;
    totalElapsedTime = 0.0;
    for(int i=0;i<NSteps;i++){
        errorCode = clSetKernelArg(kernelN, 2, sizeof(cl_mem), &inputBuffer);
        displayError(errorCode, "Could not set kernel argument 2");
        errorCode = clSetKernelArg(kernelN, 3, sizeof(cl_mem), &outputBuffer);
        displayError(errorCode, "Could not set kernel argument 3");
        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernelN, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, &event2);
        clWaitForEvents(1, &event2);
        // Timers
        cl_ulong start, end;
        clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
        clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
        double elapsed = (double)(end-start);
        totalElapsedTime += elapsed;
        numSteps++;
        
        // Swap buffers
        cl_mem temp = inputBuffer;
        inputBuffer = outputBuffer;
        outputBuffer = temp;
    }
    // Compute the average elapsed time
    averageElapsedTime = totalElapsedTime / (double)numSteps;
    cout<<"Steps: "<<numSteps<<endl;
    // Print the average elapsed time
    cout << "Average elapsed time: "<<setprecision(2) << averageElapsedTime << " seconds." << endl;
    // Wait for the kernel to finish
    clFinish(commandQueue);
    // Read back the results
    errorCode = clEnqueueReadBuffer(commandQueue, inputBuffer, CL_TRUE, 0, sizeof(cl_float4)*NParticles, inArray, 0, nullptr, nullptr);
    displayError(errorCode, "Could not read output buffer");
    errorCode = clEnqueueReadBuffer(commandQueue, velocityBuffer, CL_TRUE, 0, sizeof(cl_float4)*NParticles, velArray, 0, nullptr, nullptr);
    displayError(errorCode, "Could not read output buffer");
    // Display the particles after the simulation to see the results
    nbody_output(NParticles, inArray, velArray);
//
//    // Intialize the particles again
//    nbody_init(NParticles, inBuffer, velBuffer);
//    // Simulate the particles on the CPU to compare results
//    nbody_cpu(NParticles, NSteps, dt, eps, inBuffer, velBuffer);
//    // Display the particles after the simulation
//    nbody_output(NParticles, inBuffer, velBuffer);
    
    // Free the buffers
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(velocityBuffer);
    // Free the kernel
    clReleaseKernel(tilesKernel);
    // Free the program
    clReleaseProgram(program);
    // Free the command queue
    clReleaseCommandQueue(commandQueue);
    // Free the context
    clReleaseContext(context);
    // Free the buffers
    delete[] inArray;
    delete[] outArray;
    delete[] velArray;
    
    // Exit
    return 0;
}



//// Set up your OpenCL context, command queue, and other necessary objects
//
//// Create the buffer object
//cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &error);
//if (error != CL_SUCCESS) {
//  // Handle error
//}
//
//// Map the buffer to the host memory space
//void* mapped_ptr = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
//if (error != CL_SUCCESS) {
//  // Handle error
//}
//
//// Use the memset function to set all the bytes in the buffer to 0
//memset(mapping_ptr, 0, size);
//
//// Unmap the buffer
//error = clEnqueueUnmapMemObject(queue, buffer, mapping_ptr, 0, NULL, NULL);
//if (error != CL_SUCCESS) {
//  // Handle error
//}
//
//// You can now use the buffer as usual
