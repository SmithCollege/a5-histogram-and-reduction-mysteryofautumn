#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 1000000

double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { printf("get time ofday error"); }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

float min(float i, float j){
    if (i<j){return i;}
    else{return j;}
}

float max(float i, float j){
    if (i>j){return i;}
    else{return j;}
}


void Reduction(float* M, int operation){
    for(int i=1; i<SIZE; i++){

    switch (operation){
        case 1: //sum
            M[0] = M[0] + M[i];
            break;
        case 2: //product
            M[0] = M[0] * M[i];
            break;
        case 3: //max
            M[0] = max(M[0], M[i]);
            break;
        case 4: //min
            M[0] = min(M[0], M[i]);
            break;
        }
    }
}

int main(){
    float* M = malloc(sizeof(float) * SIZE);

    // initialization
    for (int i = 0; i < SIZE; i++) {
        M[i] = 1;
    }

    double t0 = get_clock();

    Reduction(M,1);

    double t1 = get_clock();

    printf("size: %d\n",SIZE);
    printf("time: %f ns\n", (1000000000.0 * (t1 - t0)));

    // error checking
    
    printf("M[0]: %f\n", M[0]);
  
    return 0;
}