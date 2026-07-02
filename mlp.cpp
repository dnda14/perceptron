#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <omp.h>

class MNISTDeepMLP {
private:
    // Configuración de la arquitectura, ej: {784, 128, 10} o {784, 256, 128, 10}
    std::vector<int> capa_sizes;
    int n_capas; // Número total de capas (incluye entrada y salida)

    // W[l] contiene la matriz de pesos que va desde la capa l hasta la capa l+1
    // Dimensiones: W[l][neurona_destino][peso_origen] -> la posición 0 es el Bias
    std::vector<std::vector<std::vector<double>>> W; 
    double eta;

    double sigmoide(double u) const {
        return 1.0 / (1.0 + std::exp(-u));
    }

    double derivada_sigmoide(double f_u) const {
        return f_u * (1.0 - f_u);
    }

public:
    MNISTDeepMLP(const std::vector<int>& arquitectura, double tasa_aprendizaje = 0.05) {
        capa_sizes = arquitectura;
        n_capas = arquitectura.size();
        eta = tasa_aprendizaje;

        W.resize(n_capas - 1);

        // Inicialización Xavier/Glorot
        std::default_random_engine generator(42);
        
        for (int l = 0; l < n_capas - 1; ++l) {
            int n_in = capa_sizes[l];
            int n_out = capa_sizes[l + 1];
            
            double limite = std::sqrt(6.0 / (n_in + n_out));
            std::uniform_real_distribution<double> distribution(-limite, limite);

            W[l].assign(n_out, std::vector<double>(n_in + 1));
            for (int i = 0; i < n_out; ++i) {
                for (int j = 0; j <= n_in; ++j) {
                    W[l][i][j] = distribution(generator);
                }
            }
        }
    }

    int n_capas_ocultas() const {
        return n_capas - 2; // Total de capas menos entrada y salida
    }

    // =====================================================================
    // MÓDULO 2: PROPAGACIÓN HACIA ADELANTE GENERALIZADA (FORWARD)
    // =====================================================================
    int forward(const std::vector<double>& x, 
                std::vector<std::vector<double>>& a) const {
        
        for (size_t j = 0; j < x.size(); ++j) {
            a[0][j] = x[j];
        }

        for (int l = 0; l < n_capas - 1; ++l) {
            int n_out = capa_sizes[l + 1];
            int n_in = capa_sizes[l];

            a[l + 1][0] = 1.0; // Bias de la capa siguiente

            for (int i = 0; i < n_out; ++i) {
                double u = 0.0;
                for (int j = 0; j <= n_in; ++j) {
                    u += a[l][j] * W[l][i][j];
                }
                a[l + 1][i + 1] = sigmoide(u);
            }
        }

        int capa_final = n_capas - 1;
        double max_val = -1e9;
        int clase_ganadora = 0;
        for (int i = 0; i < capa_sizes[capa_final]; ++i) {
            if (a[capa_final][i + 1] > max_val) {
                max_val = a[capa_final][i + 1];
                clase_ganadora = i;
            }
        }
        return clase_ganadora;
    }

    // =====================================================================
    // MÓDULO 3: BACKPROPAGATION MULTICAPA PARALELO
    // =====================================================================
    void entrenar_epoca(const std::vector<std::vector<double>>& X_train, const std::vector<int>& labels_train) {
        int errores_en_epoca = 0;
        size_t n_ejemplos = X_train.size();
        int capa_final = n_capas - 1;

        #pragma omp parallel reduction(+:errores_en_epoca)
        {
            std::vector<std::vector<double>> a(n_capas);
            for (int l = 0; l < n_capas; ++l) {
                a[l].resize(capa_sizes[l] + 1);
            }

            std::vector<std::vector<double>> delta(n_capas - 1);
            for (int l = 0; l < n_capas - 1; ++l) {
                delta[l].resize(capa_sizes[l + 1]);
            }

            std::vector<double> d_out(capa_sizes[capa_final]);

            #pragma omp for schedule(guided)
            for (size_t i = 0; i < n_ejemplos; ++i) {
                const std::vector<double>& x = X_train[i];
                int etiqueta_real = labels_train[i];

                int prediccion = forward(x, a);
                if (prediccion != etiqueta_real) {
                    errores_en_epoca++;
                }

                for (int k = 0; k < capa_sizes[capa_final]; ++k) {
                    d_out[k] = (etiqueta_real == k) ? 1.0 : 0.0;
                }

                int l_del = n_capas - 2;
                for (int k = 0; k < capa_sizes[capa_final]; ++k) {
                    delta[l_del][k] = (d_out[k] - a[capa_final][k + 1]) * derivada_sigmoide(a[capa_final][k + 1]);
                }

                for (int l = n_capas - 3; l >= 0; --l) {
                    int n_actual = capa_sizes[l + 1];
                    int n_siguiente = capa_sizes[l + 2];

                    for (int j = 0; j < n_actual; ++j) {
                        double suma_error = 0.0;
                        for (int k = 0; k < n_siguiente; ++k) {
                            suma_error += delta[l + 1][k] * W[l + 1][k][j + 1];
                        }
                        delta[l][j] = suma_error * derivada_sigmoide(a[l + 1][j + 1]);
                    }
                }

                for (int l = 0; l < n_capas - 1; ++l) {
                    int n_out = capa_sizes[l + 1];
                    int n_in = capa_sizes[l];

                    for (int i = 0; i < n_out; ++i) {
                        for (int j = 0; j <= n_in; ++j) {
                            #pragma omp atomic
                            W[l][i][j] += eta * delta[l][i] * a[l][j];
                        }
                    }
                }
            }
        }
        std::cout << " -> Errores: " << errores_en_epoca 
                  << " (" << (double)errores_en_epoca / X_train.size() * 100.0 << "%)\n";
    }

    int probar(const std::vector<double>& x) const {
        std::vector<std::vector<double>> a(n_capas);
        for (int l = 0; l < n_capas; ++l) {
            a[l].resize(capa_sizes[l] + 1);
        }
        return forward(x, a);
    }
};

// =====================================================================
// CONSTRUCTOR DE ARQUITECTURA FLEXIBLE
// =====================================================================
// Permite definir cualquier número de capas ocultas (1, 2, o más) y
// cuántas neuronas tiene cada una, sin tocar el resto del código.
//
// Ejemplos de uso:
//   construir_arquitectura(784, {128}, 10)        -> 1 capa oculta de 128 neuronas
//   construir_arquitectura(784, {256, 128}, 10)    -> 2 capas ocultas (256 y 128)
//   construir_arquitectura(784, {}, 10)            -> 0 capas ocultas (regresión logística)
std::vector<int> construir_arquitectura(int n_entrada,
                                         const std::vector<int>& neuronas_capas_ocultas,
                                         int n_salida) {
    std::vector<int> arquitectura;
    arquitectura.push_back(n_entrada);
    for (int n : neuronas_capas_ocultas) {
        arquitectura.push_back(n);
    }
    arquitectura.push_back(n_salida);
    return arquitectura;
}

// Imprime la arquitectura resultante de forma legible, ej: 784 -> 256 -> 128 -> 10
void imprimir_arquitectura(const std::vector<int>& arquitectura) {
    std::cout << "Arquitectura: ";
    for (size_t i = 0; i < arquitectura.size(); ++i) {
        std::cout << arquitectura[i];
        if (i + 1 < arquitectura.size()) std::cout << " -> ";
    }
    std::cout << "  (" << (arquitectura.size() - 2) << " capa(s) oculta(s))\n";
}

// =====================================================================
// LECTOR DE ARCHIVOS CSV DE MNIST
// =====================================================================
bool cargar_dataset_csv(const std::string& ruta_archivo, 
                        std::vector<std::vector<double>>& X_data, 
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

        std::vector<double> features;
        features.push_back(1.0); // Bias x0

        while (std::getline(ss, valor, ',')) {
            features.push_back((double)std::stoi(valor) / 255.0);
        }

        if (features.size() == 785) X_data.push_back(features);
        else y_data.pop_back();
    }
    archivo.close();
    return true;
}

// =====================================================================
// MAIN CONTROLADOR
// =====================================================================
int main() {
    std::string archivo_train = "mnist_train.csv"; 
    std::string archivo_test  = "mnist_test.csv";

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    std::cout << "Cargando datasets desde archivos CSV...\n";
    if (!cargar_dataset_csv(archivo_train, X_train, y_train)) return -1;
    if (!cargar_dataset_csv(archivo_test, X_test, y_test)) return -1;

    // =========================================================
    // ZONA DE CONFIGURACIÓN — MODIFICA SOLO ESTO
    // =========================================================
    // Define aquí cuántas capas ocultas quieres y cuántas neuronas
    // tiene cada una. El tamaño del vector = número de capas ocultas.

    std::vector<int> neuronas_ocultas = {256, 128};   // <-- CAMBIA AQUÍ
    double tasa_aprendizaje = 0.5;                   // <-- Y AQUÍ SI HACE FALTA
    int epocas = 20;                                  // <-- Y AQUÍ

    // Con 1 sola capa oculta suele funcionar mejor una tasa de
    // aprendizaje un poco más alta (ej. 0.1) porque hay menos capas
    // por las que el gradiente se tiene que propagar.
    // Con 2 capas ocultas, 0.03-0.05 suele ser un rango seguro.
    // =========================================================

    std::vector<int> mi_arquitectura = construir_arquitectura(784, neuronas_ocultas, 10);
    imprimir_arquitectura(mi_arquitectura);

    MNISTDeepMLP red(mi_arquitectura, tasa_aprendizaje);

    std::cout << "=== INICIANDO ENTRENAMIENTO ===\n";
    double tiempo_inicio = omp_get_wtime();

    for (int e = 1; e <= epocas; ++e) {
        std::cout << "Epoca " << e;
        red.entrenar_epoca(X_train, y_train);
    }

    double tiempo_final = omp_get_wtime();
    std::cout << "\nTiempo de entrenamiento: " << (tiempo_final - tiempo_inicio) << " segundos.\n";

    std::cout << "\n=== EVALUANDO EN EL TEST-SET ===\n";
    int aciertos = 0;
    
    #pragma omp parallel for reduction(+:aciertos)
    for (size_t i = 0; i < X_test.size(); ++i) {
        if (red.probar(X_test[i]) == y_test[i]) {
            aciertos++;
        }
    }

    std::cout << "Resultados Finales:\n";
    std::cout << "-> Clasificaciones correctas: " << aciertos << " de " << X_test.size() << "\n";
    std::cout << "-> Precision global en Test: " << ((double)aciertos / X_test.size()) * 100.0 << "%\n";

    return 0;
}