#define N 16

__global__ void ERR_biderketa(float *A, float *B, float *D, int tam1,  int tam2, int tam3){

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float tmp_sum = 0.00f;

    __shared__ float As[N][N];
    __shared__ float Bs[N][N];

    for(int k = 0; k < (tam2 + N - 1 / N); k++){
        if (row < tam1 && k * N + tx < tam2)
            As[ty][tx] = A[row * tam2 + k * N + tx];
        else
            As[ty][tx] = 0.0f;

        if (k * N + ty < tam2 && col < tam3)
            Bs[ty][tx] = B[(k * N + ty) * tam3 + col];
        else
            Bs[ty][tx] = 0.0f;
        __syncthreads();

        for (int e = 0; e < N; e++) tmp_sum += As[ty][e] * Bs[e][tx];

        __syncthreads();
    }

    if (row < tam1 && col < tam3)
        D[row * tam3 + col] = tmp_sum;
}

__global__ void ERR_gehiketa(float *C, float *D, int tam1,  int tam2, int tam3){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < tam1*tam3){
        int x = i / tam3;
        int y = i % tam3;
        D[x*tam3+y] += C[x*tam3+y];
    }
}

float codeERR (float *A, float *B, float *C, float *D, int tam1,  int tam2, int tam3){
    
    // GPU-an non gordeko diren matrizeak sortu eta gorde
    float *d_A, *d_B, *d_C, *d_D;

    cudaMalloc (&d_A, tam1 * tam2 * sizeof(float));
    cudaMalloc (&d_B, tam2 * tam3 * sizeof(float));
    cudaMalloc (&d_C, tam1 * tam3 * sizeof(float));
    cudaMalloc (&d_D, tam1 * tam3 * sizeof(float));

    cudaMemcpy (d_A, A, tam1 * tam2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy (d_B, B, tam2 * tam3 * sizeof(float), cudaMemcpyHostToDevice);

    //Denborari dagokion aldagaiak sortu eta hasieratu
    float Tex;
    cudaEvent_t t0, t1;

    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    
    cudaEventRecord(t0);

    //Funtzioari deitzeko datu egiturak sortu
    dim3 threadsPerBlock_b(N, N);// Hari kopurua bloke bakoitzean, biren berretura eta 16 gutxienez
    dim3 blocksPerGrid_b((tam3 + N - 1) / N, (tam1 + N - 1) / N);// Bloke kopurua, ondorioz, elementu kopurua zati hari kopurua goruntz borobilduta

    //Gure kernelaren exekuzioa 
    ERR_biderketa<<<blocksPerGrid_b, threadsPerBlock_b>>> (d_A, d_B, d_D, tam1, tam2, tam3);

    cudaFree (d_A);
    cudaFree (d_B);

    cudaMemcpy (d_C, C, tam1 * tam3 * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock_g = N; // Hari kopurua bloke bakoitzean, biren berretura eta 16 gutxienez
    int blocksPerGrid_g = (tam1 * tam3 + N - 1) / N; // Bloke kopurua, ondorioz, elementu kopurua zati hari kopurua goruntz borobilduta

    ERR_gehiketa<<<blocksPerGrid_g, threadsPerBlock_g>>> (d_C, d_D, tam1, tam2, tam3);

    cudaMemcpy (D, d_D, tam1 * tam3 * sizeof(float), cudaMemcpyDeviceToHost);

    //Bukaera denborak lortu eta hauen konparazioa Tex aldagaian gorde
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime (&Tex, t0, t1);

    //Denbora aldagaiak liberatu
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    //Matrizeak GPU-tik askatu
    cudaFree (d_C);
    cudaFree (d_D);

    return Tex;
}