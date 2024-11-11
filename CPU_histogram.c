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


int main(){
    int* M = malloc(sizeof(int) * SIZE);
    int* N = malloc(sizeof(int) * 10);

    // initialization
    for (int i = 0; i < SIZE; i++) {
        M[i] = 1;
    }

    double t0 = get_clock();

    for(int i = 0; i < SIZE; i++){
        N[(M[i]%10)-1]+=1;
    }

    double t1 = get_clock();

    printf("size: %d\n",SIZE);
    printf("time: %f ns\n", (1000000000.0 * (t1 - t0)));

    // error checking
    
    printf("N[0]: %d\n", N[0]);
  
    return 0;
}