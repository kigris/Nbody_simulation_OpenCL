//
//kernel void nbody_tiles(const float dt ,const float eps, global float4* in, global float4* out, global float4* vel, const int NTiles, local float4* localPos){
//    int localId = get_local_id(0); // Get local id
//    int globalId = get_global_id(0); // Get global id
//    int localSize = get_local_size(0); // Get local size
//    const float4 dtK = (float4)(dt, dt, dt, 0.0f); // Vectorize dt for the kernel from argument
//
//    float4 globalPos = in[globalId]; // Global position of particle
//    float4 globalVel = vel[globalId]; // Global velocity of particle
//    float4 globalAccel = (float4)(0.0f, 0.0f, 0.0f, 0.0f); // Set global acceleration of particle to zero
//
//    // Iterate over all tiles
//    for(int i=0;i<NTiles;i++){
//        // Copy data from global memory to local memory
//        localPos[localId]=in[i*localSize+localId];
//        // Give all the threads withing a workgroup the chance to load the data
//        barrier(CLK_LOCAL_MEM_FENCE);
//        // Calculate acceleration iterating over the block from localMemory
//        for(int j=0;j<localSize;j++){
//            // Get current position from local tile
//            float4 currentPos = localPos[j];
//            // Calculate distance
//            float4 dist = currentPos-globalPos;
//            // Calculate inverse square
//            float invDist = rsqrt(dist.x*dist.x+dist.y*dist.y+dist.z*dist.z+eps);
//            // Calculate distance cube
//            float invDistCube = invDist*invDist*invDist;
//            // Calculate force by multiplying distance cube with mass of particle j
//            float force = currentPos.w*invDistCube;
//            // Accumulate acceleration
//            globalAccel += force*dist;
//        }
//        // Sync threads before loading the next tile
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    // Update position = position + velocity*dt + 0.5*acceleration*dt^2
//    globalPos += globalVel*dtK + 0.5f*globalAccel*dtK*dtK;
//    // Update velocity = dt*acceleration
//    globalVel += globalAccel*dtK;
//    // Write back to global memory the position
//    out[globalId] = globalPos;
//    // Write back to global memory the velocity
//    vel[globalId] = globalVel;
//}



//kernel void nbody_tiles(const float dt ,const float eps, global float4* in, global float4* out, global float4* vel, const int NTiles, local float4* localPos){
//    int localId = get_local_id(0); // Get local id
//    int globalId = get_global_id(0); // Get global id
//    int localSize = get_local_size(0); // Get local size
//    const float4 dtK = (float4)(dt, dt, dt, 0.0f); // Vectorize dt for the kernel from argument
//    const float4 epsK = (float4)(eps, eps, eps, eps); // Vectorize eps for the kernel from argument
//
//    float4 globalPos = in[globalId]; // Global position of particle
//    float4 globalVel = vel[globalId]; // Global velocity of particle
//    float4 globalAccel = (float4)(0.0f, 0.0f, 0.0f, 0.0f); // Set global acceleration of particle to zero
//    float4 currentPos, dist, invDist, invDistCube;
//    float force;
//
//    // Iterate over all tiles
//    for(int i=0;i<NTiles;i++){
//        // Copy data from global memory to local memory
//        localPos[localId]=in[i*localSize+localId];
//        // Give all the threads withing a workgroup the chance to load the data
//        barrier(CLK_LOCAL_MEM_FENCE);
//        // Calculate acceleration iterating over the block from localMemory
//        for(int j=0;j<localSize;j++){
//            // Get current position from local tile
//            currentPos = localPos[j];
//            // Calculate distance
//            dist = currentPos-globalPos;
//            // Calculate inverse square
//            invDist = rsqrt(dist[0]*dist[0]+dist[1]*dist[1]+dist[2]*dist[2]+epsK);
//            // Calculate distance cube
//            invDistCube = invDist*invDist*invDist;
//            // Calculate force by multiplying distance cube with mass of particle j
//            force = currentPos[3]*invDistCube[0];
//            // Accumulate acceleration
//            globalAccel += force*dist;
//        }
//        // Sync threads before loading the next tile
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    // Update position = position + velocity*dt + 0.5*acceleration*dt^2
//    globalPos += globalVel*dtK + 0.5f*globalAccel*dtK*dtK;
//    // Update velocity = dt*acceleration
//    globalVel += globalAccel*dtK;
//    // Write back to global memory the position
//    out[globalId] = globalPos;
//    // Write back to global memory the velocity
//    vel[globalId] = globalVel;
//}


kernel void nbody_tiles(const float dt ,const float eps, global float4* in, global float4* out, global float4* vel, const int NTiles, local float4* localPos){
    int localId = get_local_id(0); // Get local id
    int globalId = get_global_id(0); // Get global id
    int localSize = get_local_size(0); // Get local size
    const float8 dtK = (float8)(dt, dt, dt, 0.0f, dt, dt, dt, 0.0f); // Vectorize dt for the kernel from argument
    const float8 epsK = (float8)(eps, eps, eps, eps, eps, eps, eps, eps); // Vectorize eps for the kernel from argument

    float4 globalPos = in[globalId]; // Global position of particle
    float4 globalVel = vel[globalId]; // Global velocity of particle
    float4 globalAccel = (float4)(0.0f, 0.0f, 0.0f, 0.0f); // Set global acceleration of particle to zero
    float8 currentPos, dist, invDist, invDistCube;
    float2 force;

    // Iterate over all tiles
    for(int i=0;i<NTiles;i++){
        // Copy data from global memory to local memory
        localPos[localId]=in[i*localSize+localId];
        // Give all the threads withing a workgroup the chance to load the data
        barrier(CLK_LOCAL_MEM_FENCE);
        // Calculate acceleration iterating over the block from localMemory
        for(int j=0;j<localSize;j+=2){
            // Get current position from local tile
            currentPos = (float8)(localPos[j], localPos[j+1]);
            // Calculate distance
            dist = currentPos-((float8)(globalPos, globalPos));
            // Calculate inverse square
            invDist.lo = rsqrt((float4)(dist[0]*dist[0]+dist[1]*dist[1]+dist[2]*dist[2]+epsK.lo));
            invDist.hi = rsqrt((float4)(dist[4]*dist[4]+dist[5]*dist[5]+dist[6]*dist[6]+epsK.hi));
            // Calculate distance cube
            invDistCube = invDist*invDist*invDist;
            // Calculate force by multiplying distance cube with mass of particle j
            force = (float2)(currentPos[3]*invDistCube[0], currentPos[7]*invDistCube[4]);
            // Accumulate acceleration
            globalAccel += force[0]*dist.lo;
            globalAccel += force[1]*dist.hi;
        }
        // Sync threads before loading the next tiler
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Update position = position + velocity*dt + 0.5*acceleration*dt^2
    globalPos += globalVel*dtK.lo + 0.5f*globalAccel*dtK.lo*dtK.lo;
    // Update velocity = dt*acceleration
    globalVel += globalAccel*dtK.lo;
    // Write back to global memory the position
    out[globalId] = globalPos;
    // Write back to global memory the velocity
    vel[globalId] = globalVel;
}



//kernel void nbody_N(const float dt, const float eps, global float4* in, global float4* out, global float4* vel, const int N){
//    int globalId = get_global_id(0); // Get global id
//    const float4 dtK = {dt, dt, dt, 0}; // Initialize time derivative to float4 for speed
//
//    float4 globalPos = in[globalId];
//    float4 globalVel = vel[globalId];
//    float4 globalAcc = {0.0f, 0.0f, 0.0f, 0.0f};
//
//    for(int i=0; i<N; i++){
//        // Get current particle's position
//        float4 currentPos = in[i];
//        // Get distance between currentParticle and globalParticle
//        float4 distance = currentPos-globalPos;
//        // Calculate inverse square of distance
//        float invDist = rsqrt(distance.x*distance.x+distance.y*distance.y+distance.z*distance.z+eps);
//        // Calculate distance cube
//        float invDistCube = invDist*invDist*invDist;
//        // Calculate force by multiplying distance cube with mass of the current particle
//        float force = currentPos.w*invDistCube;
//        // Accumulate acceleration
//        globalAcc += force*distance;
//    }
//    // Update position = position + velocity*dt + 0.5+acceleration*dt^2
//    globalPos += globalVel*dtK+0.5f*globalAcc*dtK*dtK;
//    // Update velocity = dt*acceleration
//    globalVel += globalAcc*dtK;
//    // Write results back
//    out[globalId] = globalPos;
//    vel[globalId] = globalVel;
//}

kernel void nbody_N(const float dt, const float eps, global float4* in, global float4* out, global float4* vel, const int N){
    int globalId = get_global_id(0); // Get global id
    const float4 dtK = {dt, dt, dt, 0}; // Initialize time derivative to float4 for speed
    
    float4 globalPos = in[globalId];
    float4 globalVel = vel[globalId];
    float4 globalAcc = {0.0f, 0.0f, 0.0f, 0.0f};
    float8 invDist;
    float2 force;
    for(int i=0; i<N; i+=2){
        // Get current particle's position
        float8 currentPos = (float16)(in[i], in[i+1]);
        // Get distance between currentParticle and globalParticle
        float8 distance = currentPos-((float8)(globalPos, globalPos));
        // Calculate inverse square of distance
        invDist.lo = rsqrt(distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2]+eps);
        invDist.hi = rsqrt(distance[4]*distance[4]+distance[5]*distance[5]+distance[6]*distance[6]+eps);
        // Calculate distance cube
        float8 invDistCube = invDist*invDist*invDist;
        // Calculate force by multiplying distance cube with mass of the current particle
        force = (float2)(currentPos[3]*invDistCube[0], currentPos[7]*invDistCube[4]);
        // Accumulate acceleration
        globalAcc += force[0]*distance.lo;
        globalAcc += force[1]*distance.hi;
    }
    // Update position = position + velocity*dt + 0.5+acceleration*dt^2
    globalPos += globalVel*dtK+0.5f*globalAcc*dtK*dtK;
    // Update velocity = dt*acceleration
    globalVel += globalAcc*dtK;
    // Write results back
    out[globalId] = globalPos;
    vel[globalId] = globalVel;
}
