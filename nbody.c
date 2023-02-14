#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ncurses.h>
#include "timer.h"
#include <unistd.h>
#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

void bodyForce(Body *p, float dt, int n) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
        
        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }
        
        p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
    }
}

int main(const int argc, const char** argv) {
    // Initialize ncurses
    initscr();
    
    int nBodies = 2;
    if (argc > 1) nBodies = atoi(argv[1]);
    
    const float dt = 0.01f; // time step
    const int nIters = 100;  // simulation iterations
    
    int bytes = nBodies*sizeof(Body);
    float *buf = (float*)malloc(bytes);
    Body *p = (Body*)buf;
    
    randomizeBodies(buf, 6*nBodies); // Init pos / vel data
    
    double totalTime = 0.0;
    
    for (int iter = 1; iter <= nIters; iter++) {
        StartTimer();
        
        bodyForce(p, dt, nBodies); // compute interbody forces
        
        for (int i = 0 ; i < nBodies; i++) { // integrate position
            p[i].x += p[i].vx*dt;
            p[i].y += p[i].vy*dt;
            p[i].z += p[i].vz*dt;
        }
        
        // Clear the screen
        clear();
        
        // Loop through each body and print its position
        for (int j = 0; j < nBodies; j++) {
            mvprintw(j, 0, "Body %d: (%.3f, %.3f, %.3f)", j, p[j].x, p[j].y, p[j].z);
        }
        // Calculate the angle of the projection
        float phi = M_PI / 4.0;

        // Transform the positions of the bodies
        for (int j = 0; j < nBodies; j++) {
          float x = p[j].x * cos(phi) - p[j].y * sin(phi);
          float y = p[j].x * sin(phi) + p[j].y * cos(phi);
          p[j].x = x;
          p[j].y = y;
        }

        // Draw lines between the bodies
        for (int j = 0; j < nBodies; j++) {
            for (int k = j + 1; k < nBodies; k++) {
                // Get the positions of the two bodies
                int x1 = (int)(p[j].x * 10.0);
                int y1 = (int)(p[j].y * 10.0);
                int z1 = (int)(p[j].z * 10.0);
                int x2 = (int)(p[k].x * 10.0);
                int y2 = (int)(p[k].y * 10.0);
                int z2 = (int)(p[k].z * 10.0);
                
                // Calculate the slope and intercept of the line
                float mx = (float)(x2 - x1) / (float)(y2 - y1);
                float my = (float)(y2 - y1) / (float)(z2 - z1);
                float mz = (float)(z2 - z1) / (float)(x2 - x1);
                int bx = (int)(x1 - mx * y1);
                int by = (int)(y1 - my * z1);
                int bz = (int)(z1 - mz * x1);
                
                // Draw the line
                for (int x = 0; x < COLS; x++) {
                    for (int y = 0; y < LINES; y++) {
                        int z = (int)(mx * x + bx);
                        if (z >= 0 && z < COLS && y >= 0 && y < LINES) {
                            mvaddch(z, y, '*');
                        }
                    }
                }
                
                for (int y = 0; y < LINES; y++) {
                    for (int z = 0; z < COLS; z++) {
                        int x = (int)(my * y + by);
                        if (x >= 0 && x < COLS && z >= 0 && z < LINES) {
                            mvaddch(z, y, '*');
                        }
                    }
                }
                
                for (int z = 0; z < COLS; z++) {
                    for (int x = 0; x < LINES; x++) {
                        int y = (int)(mz * z + bz);
                        if (x >= 0 && x < COLS && y >= 0 && y < LINES) {
                            mvaddch(z, y, '*');
                        }
                    }
                }
            }
        }
        
        // Refresh the screen
        refresh();
        
        //        // Pause for 100 milliseconds
        //        usleep(1000000);
        // Pause for 100 milliseconds
        napms(1000);
        
        const double tElapsed = GetTimer() / 1000.0;
        if (iter > 1) { // First iter is warm up
            totalTime += tElapsed;
        }
        
        printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
    }
    
    double avgTime = totalTime / (double)(nIters-1);
    
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
    
    free(buf);
    
    // Clean up ncurses
    endwin();
}

