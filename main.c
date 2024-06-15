#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "include/codeGPU.cuh"
#include "include/codeSHA.cuh"
#include "include/codeERR.cuh"

double timing_CPU(struct timespec begin, struct timespec end){
     return ((end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0));
}

void sortu_matrizeak(float *A, float *B, float *C, float *D, int tam1,  int tam2, int tam3){

    int i, j;

    // A matrizea generatu
    for (i = 0; i < tam1; i++) 
        for (j = 0; j < tam2; j++) 
            A[i*tam2+j] = i;
            //A[i*tam2+j] = sin(i);  

    // B matrizea generatu
    for (i = 0; i < tam2; i++) 
        for (j = 0; j < tam3; j++) 
            B[i*tam2+j] = j;
            //B[i*tam3+j] = cos(j);

    // C matrizea generatu
    for (i = 0; i < tam1; i++) 
        for (j = 0; j < tam3; j++) 
            C[i*tam3+j] = i+j;

    for (i = 0; i < tam1; i++) 
        for (j = 0; j < tam3; j++) 
            D[i*tam3+j] = 0;
        
    
}

void matrizeak_biderkatu(float *A, float *B, float *D, int tam1,  int tam2, int tam3){
    
    int i, j, k;
    float sum = 0.0;

    for (i = 0; i < tam1; i++){
        for (j = 0; j < tam3; j++){
            D[i*tam3+j] = 0.0;
            for (k = 0; k < tam2; k++){
                D[i*tam3+j] += A[i*tam2+k] * B[k*tam3+j];
            }
        }
    }
}

void matrizeak_gehitu(float *C, float *D, int tam1, int tam3){

    int i, j;

    for (i = 0; i < tam1; i++)
        for (j = 0; j < tam3; j++)
            D[i*tam3+j] = D[i*tam3+j] + C[i*tam3+j];

}

double codeCPU(float *A, float *B, float *C, float *D, int tam1,  int tam2, int tam3){
    
    struct timespec begin, end;

    clock_gettime(CLOCK_MONOTONIC, &begin); //clock_gettime(CLOCK_REALTIME, &begin);

    matrizeak_biderkatu(A, B, D, tam1, tam2, tam3);
    
    matrizeak_gehitu(C, D, tam1, tam3);

    clock_gettime(CLOCK_MONOTONIC, &end); //clock_gettime(CLOCK_REALTIME, &end);

    return(timing_CPU(begin, end));

}

int main(int argc, char *argv[]){
    
    int i, j;
    double time_cpu;
    float time_gpu, time_sha, time_err;
    
    if (argc != 4){
        printf("Sartu hiru zenbaki exekuzioan. \n");
        return 0;
    }

    int tam1 = strtoul(argv[1], NULL, 10);
	int tam2 = strtoul(argv[2], NULL, 10);
	int tam3 = strtoul(argv[3], NULL, 10);

    float *A, *B, *C, *D;

    // matrizeentzako tokia egin
    A = (float *)malloc(tam1 * tam2 * sizeof(float));
    B = (float *)malloc(tam2 * tam3 * sizeof(float));
    C = (float *)malloc(tam1 * tam3 * sizeof(float));
    D = (float *)malloc(tam1 * tam3 * sizeof(float));

    // matrizeak bete
    sortu_matrizeak(A, B, C, D, tam1, tam2, tam3);

    // matrizeak konprobatu
    printf("A matrizea:\n");
    for (i = 0; i < tam1; i++){
        for (j = 0; j < tam2; j++){
            printf("%.2f ", A[i * tam2 + j]);
        }
        printf("\n");
    }
    
    printf("\n\nB matrizea:\n");
    for (i = 0; i < tam2; i++){
        for (j = 0; j < tam3; j++){
            printf("%.2f ", B[i * tam2 + j]);
        }
        printf("\n");
    }  

    printf("\n\nC matrizea:\n");
    for (i = 0; i < tam1; i++){
        for (j = 0; j < tam3; j++){
            printf("%.2f ", C[i * tam1 + j]);
        }
        printf("\n");
    }

    time_cpu = codeCPU(A, B, C, D, tam1, tam2, tam3);

    printf("\n\nD matrizea CPU eragiketa ondoren:\n");
    for (i = 0; i < tam1; i++){
        for (j = 0; j < tam3; j++){
            printf("%.2f ", D[i * tam3 + j]);
        }
        printf("\n");
    }

    //Matrizeetako balioak berriro hasieratu arazorik ez izateko badaezpada
    sortu_matrizeak(A, B, C, D, tam1, tam2, tam3);

    time_gpu = codeGPU(A, B, C, D, tam1, tam2, tam3);

    printf("\n\nD matrizea GPU eragiketa ondoren:\n");
    for (i = 0; i < tam1; i++){
        for (j = 0; j < tam3; j++){
            printf("%.2f ", D[i * tam3 + j]);
        }
        printf("\n");
    }

    //Matrizeetako balioak berriro hasieratu arazorik ez izateko badaezpada
    sortu_matrizeak(A, B, C, D, tam1, tam2, tam3);

    time_sha = codeSHA(A, B, C, D, tam1, tam2, tam3);

    printf("\n\nD matrizea GPU eragiketa ondoren shared memory erabiliz:\n");
    for (i = 0; i < tam1; i++){
        for (j = 0; j < tam3; j++){
            printf("%.2f ", D[i * tam3 + j]);
        }
        printf("\n");
    }  

    //Matrizeetako balioak berriro hasieratu arazorik ez izateko badaezpada
    sortu_matrizeak(A, B, C, D, tam1, tam2, tam3);

    time_err = codeERR(A, B, C, D, tam1, tam2, tam3);

    printf("\n\nD matrizea erraldoien eragiketen ondoren:\n");
    for (i = 0; i < tam1; i++){
        for (j = 0; j < tam3; j++){
            printf("%.2f ", D[i * tam3 + j]);
        }
        printf("\n");
    }  

    printf("\nCPU exekuzio denbora: %fms\n", time_cpu);
    printf("GPU exekuzio denbora: %fms\n", time_gpu);
    printf("Shared exekuzio denbora: %fms\n", time_sha);
    printf("Matrize erraldoien exekuzio denbora: %fms\n", time_err);

    // matrizeak askatu
    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}