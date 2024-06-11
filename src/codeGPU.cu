#include <stdio.h>
#include <math.h>

#define N 16 //Hau temporala da eta aldatu edo kendu egin behar da, tam aldagaiak aldatzen ditu erroreak ez agertzeko kodetzeko orduan


__global__ void GPU_eragiketak(float *A, float *B, float *C, float *D, int tam1,  int tam2, int tam3){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < tam1*tam3){
        float tmp_sum = 0.0f;
        int x = i / tam3;
        int y = i % tam3;
        for (int k = 0; k < tam2; k++){
            tmp_sum += A[x*tam2+k] * B[k*tam3+y];
            //D[x*tam3+y] = A[x*tam2+k] * B[k*tam3+y];
        }
        D[x*tam3+y] = tmp_sum + C[x*tam3+y];
    }
}

float codeGPU (float *A, float *B, float *C, float *D, int tam1,  int tam2, int tam3){
    
    // GPU-an non gordeko diren matrizeen espazioak reserbatu eta beharrezko datuak bidali
    float *d_A, *d_B, *d_C, *d_D;

    cudaMalloc (&d_A, tam1 * tam2 * sizeof(float));
    cudaMalloc (&d_B, tam2 * tam3 * sizeof(float));
    cudaMalloc (&d_C, tam1 * tam3 * sizeof(float));
    cudaMalloc (&d_D, tam1 * tam3 * sizeof(float));

    cudaMemcpy (d_A, A, tam1 * tam2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy (d_B, B, tam2 * tam3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy (d_C, C, tam1 * tam3 * sizeof(float), cudaMemcpyHostToDevice);

    //Denborari dagokion aldagaiak sortu eta hasieratu
    float Tex;
    cudaEvent_t t0, t1;

    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    
    cudaEventRecord(t0);

    //Funtzioari deitzeko datu egiturak sortu
    /*
    //2D
    dim3 threadsPerBlock (N, N); //dim3 threadsPerBlock (BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid (ceil((tam1*tam3)/N), ceil((tam1*tam3)/N));//dim3 blocksPerGrid (ceil(N/BLOCK_SIZE), ceil(N/BLOCK_SIZE));
  */  

    // 1D
    int threadsPerBlock = N; // Hari kopurua bloke bakoitzean, biren berretura eta 16 gutxienez
    int blocksPerGrid = (tam1 * tam3 + N - 1) / N; // Bloke kopurua, ondorioz, elementu kopurua zati hari kopurua goruntz borobilduta

    //Gure kernelaren exekuzioa 
    GPU_eragiketak<<<blocksPerGrid,  threadsPerBlock>>> (d_A, d_B, d_C, d_D, tam1, tam2, tam3);

    // Emaitza soilik itzuli GPU-tik
    cudaMemcpy (D, d_D, tam1 * tam3 * sizeof(float), cudaMemcpyDeviceToHost);

    //Bukaera den borak lortu eta hauen konparazioa Tex aldagaian gorde
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime (&Tex, t0, t1);

    //Denbora aldagaiak liberatu
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    //Matrizeak GPU-tik askatu
    cudaFree (d_A);
    cudaFree (d_B);
    cudaFree (d_C);
    cudaFree (d_D);

    return Tex;
}
