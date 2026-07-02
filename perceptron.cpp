#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

// =====================================================================
// ESTRUCTURA MODULAR DEL PERCEPTRÓN
// =====================================================================
class MNISTPerceptron {
private:
    std::vector<std::vector<double>> W; 
    double eta;

public:
    MNISTPerceptron(double tasa_aprendizaje = 0.01) {
        // 10 neuronas (0-9), cada una con 785 pesos (784 píxeles + 1 Bias)
        W.assign(10, std::vector<double>(784 + 1, 0.0)); 
        eta = tasa_aprendizaje;
    }

    double regla_propagacion_individual(const std::vector<double>& x, int k) {
        double u = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            u += x[j] * W[k][j];
        }
        return u;
    }

    int calcular_prediccion(const std::vector<double>& x) {
        double max_u = -1e9;
        int clase_ganadora = 0;
        for (int k = 0; k < 10; ++k) {
            double u = regla_propagacion_individual(x, k);
            if (u > max_u) {
                max_u = u;
                clase_ganadora = k;
            }
        }
        return clase_ganadora;
    }

    void entrenar_epoca(const std::vector<std::vector<double>>& X_train, const std::vector<int>& labels_train) {
        int errores_en_epoca = 0;
        for (size_t i = 0; i < X_train.size(); ++i) {
            const std::vector<double>& x = X_train[i]; 
            int etiqueta_real = labels_train[i];
            int prediccion = calcular_prediccion(x);

            if (prediccion != etiqueta_real) {
                errores_en_epoca++;
            }

            // Actualización individual de las 10 neuronas bajo la regla del gradiente
            for (int k = 0; k < 10; ++k) {
                double u = regla_propagacion_individual(x, k);
                int y_k = (u > 0.0) ? 1 : 0; 
                int d_k = (etiqueta_real == k) ? 1 : 0;

                if (d_k != y_k) {
                    double error_local = d_k - y_k;
                    for (size_t j = 0; j < W[k].size(); ++j) {
                        W[k][j] += eta * error_local * x[j];
                    }
                }
            }
        }
        std::cout << " -> Errores: " << errores_en_epoca 
                  << " (" << (double)errores_en_epoca / X_train.size() * 100.0 << "%)\n";
    }
};

// =====================================================================
// MÓDULO AUXILIAR: LECTOR DE ARCHIVOS CSV DE MNIST
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
        
        // 1. El primer valor siempre es la etiqueta (label)
        if (std::getline(ss, valor, ',')) {
            y_data.push_back(std::stoi(valor));
        }

        // 2. Los siguientes 784 valores son los píxeles de la imagen
        std::vector<double> features;
        features.push_back(1.0); // MÓDULO 1: Insertamos el Bias (+1) en la posición x0

        while (std::getline(ss, valor, ',')) {
            int pixel_val = std::stoi(valor);
            // MÓDULO 1: Normalizamos el píxel [0, 255] al rango [0.0, 1.0]
            features.push_back((double)pixel_val / 255.0); 
        }

        // Validación rápida por seguridad (Debe tener 1 Bias + 784 píxeles)
        if (features.size() == 785) {
            X_data.push_back(features);
        } else {
            std::cerr << "Advertencia: Linea ignorada por formato incorrecto de pixeles (" << features.size() - 1 << " detectados).\n";
            y_data.pop_back(); // Descartamos la etiqueta correspondiente
        }
    }

    archivo.close();
    return true;
}

// =====================================================================
// MÓDULO PRINCIPAL: CONTROLADOR DE LA EJECUCIÓN (MAIN)
// =====================================================================
int main() {
    // IMPORTANTE: Asegúrate de colocar los nombres exactos de tus archivos CSV aquí
    std::string archivo_train = "mnist_train.csv"; 
    std::string archivo_test  = "mnist_test.csv";

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;

    std::cout << "Cargando datasets desde archivos CSV...\n";
    
    if (!cargar_dataset_csv(archivo_train, X_train, y_train)) return -1;
    std::cout << "-> Datos de Entrenamiento cargados con exito: " << X_train.size() << " ejemplos.\n";

    if (!cargar_dataset_csv(archivo_test, X_test, y_test)) return -1;
    std::cout << "-> Datos de Prueba cargados con exito: " << X_test.size() << " ejemplos.\n\n";

    // Instanciar el clasificador modular con una tasa de aprendizaje (eta) de 0.05
    MNISTPerceptron clasificador(0.05);

    // Bucle de épocas de entrenamiento
    int epocas = 10; 
    std::cout << "=== INICIANDO ENTRENAMIENTO MULTICLASE (" << epocas << " EPOCAS) ===\n";
    for (int e = 1; e <= epocas; ++e) {
        std::cout << "Epoca " << e;
        clasificador.entrenar_epoca(X_train, y_train);
    }

    // ==========================================
    // EJECUCIÓN DEL MÓDULO 4: EVALUACIÓN / PRUEBA
    // ==========================================
    std::cout << "\n=== EVALUANDO EN EL TEST-SET ===\n";
    int aciertos = 0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        int prediccion = clasificador.calcular_prediccion(X_test[i]);
        if (prediccion == y_test[i]) {
            aciertos++;
        }
    }

    double precision = ((double)aciertos / X_test.size()) * 100.0;
    std::cout << "Resultados Finales:\n";
    std::cout << "-> Clasificaciones correctas: " << aciertos << " de " << X_test.size() << "\n";
    std::cout << "-> Precision global: " << precision << "%\n";

    return 0;
}