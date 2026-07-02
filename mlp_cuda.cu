#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

// =====================================================================
// CONFIGURACIÓN DE LA RED (arquitectura fija de 1 capa oculta)
// =====================================================================
#define N_ENTRADA 784
#define N_OCULTA  128   // <-- CAMBIA AQUÍ el tamaño de la capa oculta
#define N_SALIDA  10

// =====================================================================
// KERNELS CUDA
// =====================================================================
// Cada kernel corre en la GPU. "idx" identifica qué neurona le toca
// calcular a cada hilo (thread). Es el equivalente en GPU a los
// "#pragma omp for" que usabas en la versión de CPU.

// Propaga una capa completa: a_salida = sigmoide(W * a_entrada)
// a_entrada[0] siempre es el bias (=1.0)
__global__ void kernel_forward(const float* W, const float* a_entrada,
                                float* a_salida, int n_in, int n_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_out) {
        float u = 0.0f;
        for (int j = 0; j <= n_in; ++j) {
            u += W[i * (n_in + 1) + j] * a_entrada[j];
        }
        a_salida[i + 1] = 1.0f / (1.0f + expf(-u)); // sigmoide
    }
}

// Calcula el delta (error) de la capa de salida
__global__ void kernel_delta_salida(const float* a_salida, int etiqueta,
                                     float* delta_salida, int n_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_out) {
        float d_k = (k == etiqueta) ? 1.0f : 0.0f;
        float y_k = a_salida[k + 1];
        delta_salida[k] = (d_k - y_k) * y_k * (1.0f - y_k); // regla delta + derivada sigmoide
    }
}

// Retropropaga el delta desde la capa de salida hacia la capa oculta
__global__ void kernel_delta_oculta(const float* W_salida, const float* delta_salida,
                                     const float* a_oculta, float* delta_oculta,
                                     int n_oculta, int n_salida) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_oculta) {
        float suma_error = 0.0f;
        for (int k = 0; k < n_salida; ++k) {
            suma_error += delta_salida[k] * W_salida[k * (n_oculta + 1) + (j + 1)];
        }
        float h = a_oculta[j + 1];
        delta_oculta[j] = suma_error * h * (1.0f - h);
    }
}

// Actualiza los pesos de una capa: W += eta * delta * a_entrada
__global__ void kernel_actualizar_pesos(float* W, const float* delta, const float* a_entrada,
                                         int n_in, int n_out, float eta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_out * (n_in + 1);
    if (idx < total) {
        int i = idx / (n_in + 1); // neurona destino
        int j = idx % (n_in + 1); // peso de origen (0 = bias)
        W[idx] += eta * delta[i] * a_entrada[j];
    }
}

// =====================================================================
// LECTOR DE ARCHIVOS CSV DE MNIST (igual que en tus versiones anteriores)
// =====================================================================
bool cargar_dataset_csv(const std::string& ruta_archivo,
                         std::vector<std::vector<float>>& X_data,
                         std::vector<int>& y_data) {
    std::ifstream archivo(ruta_archivo);
    if (!archivo.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo CSV: " << ruta_archivo << "\n";
        return false;
    }

    std::string linea;
    while (std::getline(archivo, linea)) {
        if (linea.empty()) continue;
        std::stringstream ss(linea);
        std::string valor;

        if (std::getline(ss, valor, ',')) {
            y_data.push_back(std::stoi(valor));
        }

        std::vector<float> features;
        features.push_back(1.0f); // Bias x0

        while (std::getline(ss, valor, ',')) {
            features.push_back((float)std::stoi(valor) / 255.0f);
        }

        if (features.size() == N_ENTRADA + 1) X_data.push_back(features);
        else y_data.pop_back();
    }
    archivo.close();
    return true;
}

// Reserva memoria en GPU e inicializa una matriz de pesos con Xavier/Glorot
float* crear_pesos_en_gpu(int n_in, int n_out, std::default_random_engine& gen) {
    int total = n_out * (n_in + 1);
    std::vector<float> w_host(total);

    float limite = std::sqrt(6.0f / (n_in + n_out));
    std::uniform_real_distribution<float> dist(-limite, limite);
    for (int i = 0; i < total; ++i) w_host[i] = dist(gen);

    float* w_device;
    cudaMalloc(&w_device, total * sizeof(float));
    cudaMemcpy(w_device, w_host.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    return w_device;
}

// =====================================================================
// MAIN CONTROLADOR
// =====================================================================
int main() {
    std::string archivo_train = "mnist_train.csv";
    std::string archivo_test  = "mnist_test.csv";

    std::vector<std::vector<float>> X_train, X_test;
    std::vector<int> y_train, y_test;

    std::cout << "Cargando datasets desde archivos CSV...\n";
    if (!cargar_dataset_csv(archivo_train, X_train, y_train)) return -1;
    if (!cargar_dataset_csv(archivo_test, X_test, y_test)) return -1;
    std::cout << "-> Entrenamiento: " << X_train.size() << " ejemplos.\n";
    std::cout << "-> Prueba: " << X_test.size() << " ejemplos.\n\n";

    float tasa_aprendizaje = 0.5f;
    int epocas = 20;

    // -----------------------------------------------------------------
    // Reservar e inicializar pesos en la GPU
    // -----------------------------------------------------------------
    std::default_random_engine gen(42);
    float* d_W1 = crear_pesos_en_gpu(N_ENTRADA, N_OCULTA, gen); // 784 -> oculta
    float* d_W2 = crear_pesos_en_gpu(N_OCULTA, N_SALIDA, gen);  // oculta -> 10

    // Buffers de activaciones y deltas (reutilizados en cada ejemplo)
    float *d_a_entrada, *d_a_oculta, *d_a_salida;
    float *d_delta_oculta, *d_delta_salida;
    cudaMalloc(&d_a_entrada, (N_ENTRADA + 1) * sizeof(float));
    cudaMalloc(&d_a_oculta, (N_OCULTA + 1) * sizeof(float));
    cudaMalloc(&d_a_salida, (N_SALIDA + 1) * sizeof(float));
    cudaMalloc(&d_delta_oculta, N_OCULTA * sizeof(float));
    cudaMalloc(&d_delta_salida, N_SALIDA * sizeof(float));

    // El bias de la capa oculta (a_oculta[0]) siempre es 1.0
    float uno = 1.0f;
    cudaMemcpy(d_a_oculta, &uno, sizeof(float), cudaMemcpyHostToDevice);

    float a_salida_host[N_SALIDA + 1]; // para leer la predicción de vuelta en CPU

    int hilos_por_bloque = 128;

    std::cout << "=== INICIANDO ENTRENAMIENTO (CUDA) ===\n";

    for (int e = 1; e <= epocas; ++e) {
        int errores_en_epoca = 0;

        for (size_t i = 0; i < X_train.size(); ++i) {
            // Copiar el ejemplo actual a la GPU
            cudaMemcpy(d_a_entrada, X_train[i].data(), (N_ENTRADA + 1) * sizeof(float), cudaMemcpyHostToDevice);

            // --- FORWARD ---
            kernel_forward<<<(N_OCULTA + hilos_por_bloque - 1) / hilos_por_bloque, hilos_por_bloque>>>(
                d_W1, d_a_entrada, d_a_oculta, N_ENTRADA, N_OCULTA);

            kernel_forward<<<(N_SALIDA + hilos_por_bloque - 1) / hilos_por_bloque, hilos_por_bloque>>>(
                d_W2, d_a_oculta, d_a_salida, N_OCULTA, N_SALIDA);

            // Traer la salida a la CPU para calcular la predicción (argmax)
            cudaMemcpy(a_salida_host, d_a_salida, (N_SALIDA + 1) * sizeof(float), cudaMemcpyDeviceToHost);

            int prediccion = 0;
            float max_val = -1e9f;
            for (int k = 0; k < N_SALIDA; ++k) {
                if (a_salida_host[k + 1] > max_val) {
                    max_val = a_salida_host[k + 1];
                    prediccion = k;
                }
            }
            if (prediccion != y_train[i]) errores_en_epoca++;

            // --- BACKWARD ---
            kernel_delta_salida<<<(N_SALIDA + hilos_por_bloque - 1) / hilos_por_bloque, hilos_por_bloque>>>(
                d_a_salida, y_train[i], d_delta_salida, N_SALIDA);

            kernel_delta_oculta<<<(N_OCULTA + hilos_por_bloque - 1) / hilos_por_bloque, hilos_por_bloque>>>(
                d_W2, d_delta_salida, d_a_oculta, d_delta_oculta, N_OCULTA, N_SALIDA);

            // --- ACTUALIZAR PESOS ---
            int total_w2 = N_SALIDA * (N_OCULTA + 1);
            kernel_actualizar_pesos<<<(total_w2 + hilos_por_bloque - 1) / hilos_por_bloque, hilos_por_bloque>>>(
                d_W2, d_delta_salida, d_a_oculta, N_OCULTA, N_SALIDA, tasa_aprendizaje);

            int total_w1 = N_OCULTA * (N_ENTRADA + 1);
            kernel_actualizar_pesos<<<(total_w1 + hilos_por_bloque - 1) / hilos_por_bloque, hilos_por_bloque>>>(
                d_W1, d_delta_oculta, d_a_entrada, N_ENTRADA, N_OCULTA, tasa_aprendizaje);
        }

        std::cout << "Epoca " << e << " -> Errores: " << errores_en_epoca
                   << " (" << (double)errores_en_epoca / X_train.size() * 100.0 << "%)\n";
    }

    // -----------------------------------------------------------------
    // EVALUACIÓN EN TEST SET
    // -----------------------------------------------------------------
    std::cout << "\n=== EVALUANDO EN EL TEST-SET ===\n";
    int aciertos = 0;

    for (size_t i = 0; i < X_test.size(); ++i) {
        cudaMemcpy(d_a_entrada, X_test[i].data(), (N_ENTRADA + 1) * sizeof(float), cudaMemcpyHostToDevice);

        kernel_forward<<<(N_OCULTA + hilos_por_bloque - 1) / hilos_por_bloque, hilos_por_bloque>>>(
            d_W1, d_a_entrada, d_a_oculta, N_ENTRADA, N_OCULTA);
        kernel_forward<<<(N_SALIDA + hilos_por_bloque - 1) / hilos_por_bloque, hilos_por_bloque>>>(
            d_W2, d_a_oculta, d_a_salida, N_OCULTA, N_SALIDA);

        cudaMemcpy(a_salida_host, d_a_salida, (N_SALIDA + 1) * sizeof(float), cudaMemcpyDeviceToHost);

        int prediccion = 0;
        float max_val = -1e9f;
        for (int k = 0; k < N_SALIDA; ++k) {
            if (a_salida_host[k + 1] > max_val) {
                max_val = a_salida_host[k + 1];
                prediccion = k;
            }
        }
        if (prediccion == y_test[i]) aciertos++;
    }

    std::cout << "Resultados Finales:\n";
    std::cout << "-> Clasificaciones correctas: " << aciertos << " de " << X_test.size() << "\n";
    std::cout << "-> Precision global en Test: " << ((double)aciertos / X_test.size()) * 100.0 << "%\n";

    // Liberar memoria de GPU
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_a_entrada);
    cudaFree(d_a_oculta);
    cudaFree(d_a_salida);
    cudaFree(d_delta_oculta);
    cudaFree(d_delta_salida);

    return 0;
}
