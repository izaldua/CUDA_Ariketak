#define BLOCK_SIZE 16

__global__ void SHA_eragiketak(float *A, float *B, float *C, float *D, int tam1,  int tam2, int tam3){
    
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float tmp_sum = 0.0f;

    for (int i = 0; i < (tam2 + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        if (row < tam1 && i * BLOCK_SIZE + tx < tam2) {
            As[ty][tx] = A[row * tam2 + i * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        if (i * BLOCK_SIZE + tx < tam2 && row < tam3) {
            Bs[ty][tx] = B[(by * BLOCK_SIZE + ty) * tam3 + i * BLOCK_SIZE + tx];
        } else {
            Bs[ty][tx] = 0.0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp_sum += As[ty][k] * Bs[ty][k];
        }
        __syncthreads();
    }

    if (row < tam1 && col < tam3) D[row*tam3+col] = tmp_sum + C[row * tam3 + col];

}

float codeSHA (float *A, float *B, float *C, float *D, int tam1,  int tam2, int tam3){
    
    // GPU-an non gordeko diren matrizeak sortu eta gorde
    float *d_A, *d_B, *d_C, *d_D;

    cudaMalloc (&d_A, tam1 * tam2 * sizeof(float));
    cudaMalloc (&d_B, tam2 * tam3 * sizeof(float));
    cudaMalloc (&d_C, tam1 * tam3 * sizeof(float));
    cudaMalloc (&d_D, tam1 * tam3 * sizeof(float));

    cudaMemcpy (d_A, A, tam1 * tam2, cudaMemcpyHostToDevice);
    cudaMemcpy (d_B, B, tam2 * tam3, cudaMemcpyHostToDevice);
    cudaMemcpy (d_C, C, tam1 * tam3, cudaMemcpyHostToDevice);
    cudaMemcpy (d_D, D, tam1 * tam3, cudaMemcpyHostToDevice);

    //Denborari dagokion aldagaiak sortu eta hasieratu
    float Tex;
    cudaEvent_t t0, t1;

    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    
    cudaEventRecord(t0);

    //Funtzioari deitzeko datu egiturak sortu
    // Zein tamainakoak jarri behar dira?
    dim3 threadsPerBlock (1, 1); //dim3 threadsPerBlock (BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid (tam1, tam3);//dim3 blocksPerGrid (ceil(N/BLOCK_SIZE), ceil(N/BLOCK_SIZE));

    //Gure kernelaren exekuzioa
    SHA_eragiketak<<<threadsPerBlock, blocksPerGrid>>> (d_A, d_B, d_C, d_D, tam1, tam2, tam3);

    cudaMemcpy (D, d_D, tam1 * tam3, cudaMemcpyDeviceToHost);

    //Bukaera denborak lortu eta hauen konparazioa Tex aldagaian gorde
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