#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <thread>
#include <algorithm>

using namespace std;

// Función de activación y su derivada
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double d_sigmoid(double y) {
    // y ya es la salida de la función sigmoid, es decir y = sigmoid(x)
    return y * (1.0 - y);
}

// Arquitectura de la red (1 capa oculta)
const int INPUT_SIZE = 784;
const int HIDDEN_SIZE = 64; // Puedes cambiar la cantidad de neuronas en la capa oculta
const int OUTPUT_SIZE = 10;

struct RedNeuronal {
    vector<vector<double>> W1; // Pesos Capa Oculta (INPUT_SIZE x HIDDEN_SIZE)
    vector<double> b1;         // Sesgos Capa Oculta (HIDDEN_SIZE)
    vector<vector<double>> W2; // Pesos Capa Salida (HIDDEN_SIZE x OUTPUT_SIZE)
    vector<double> b2;         // Sesgos Capa Salida (OUTPUT_SIZE)
};

struct Gradientes {
    vector<vector<double>> dW1;
    vector<double> db1;
    vector<vector<double>> dW2;
    vector<double> db2;
};

void inicializar_red(RedNeuronal& red) {
    red.W1.assign(INPUT_SIZE, vector<double>(HIDDEN_SIZE));
    red.b1.assign(HIDDEN_SIZE, 0.0);
    red.W2.assign(HIDDEN_SIZE, vector<double>(OUTPUT_SIZE));
    red.b2.assign(OUTPUT_SIZE, 0.0);

    // Inicialización aleatoria pequeña (entre -0.05 y 0.05)
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            red.W1[i][j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        red.b1[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            red.W2[i][j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++)
        red.b2[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
}

void acumular_gradientes(int desde, int hasta,
                         const vector<vector<double>>& matriz_input,
                         const vector<vector<int>>& matriz_clase,
                         const RedNeuronal& red,
                         Gradientes& grad) {
    
    grad.dW1.assign(INPUT_SIZE, vector<double>(HIDDEN_SIZE, 0.0));
    grad.db1.assign(HIDDEN_SIZE, 0.0);
    grad.dW2.assign(HIDDEN_SIZE, vector<double>(OUTPUT_SIZE, 0.0));
    grad.db2.assign(OUTPUT_SIZE, 0.0);

    for (int idx = desde; idx < hasta; idx++) {
        // --- FORWARD PASS ---
        // 1. Capa Oculta
        vector<double> a1(HIDDEN_SIZE, 0.0);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double suma = red.b1[j];
            for (int i = 0; i < INPUT_SIZE; i++)
                suma += matriz_input[idx][i + 1] * red.W1[i][j];
            a1[j] = sigmoid(suma);
        }

        // 2. Capa de Salida
        vector<double> a2(OUTPUT_SIZE, 0.0);
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            double suma = red.b2[k];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                suma += a1[j] * red.W2[j][k];
            a2[k] = sigmoid(suma);
        }

        // --- BACKWARD PASS (MSE Loss) ---
        // Función de Costo: MSE = 1/2 * (Predicción - Real)^2
        // Derivada de MSE respecto a la activación = (Predicción - Real)
        
        // Error en la capa de salida (Delta 2)
        vector<double> delta2(OUTPUT_SIZE, 0.0);
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            double error = a2[k] - matriz_clase[idx][k]; 
            delta2[k] = error * d_sigmoid(a2[k]);
            
            grad.db2[k] += delta2[k];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                grad.dW2[j][k] += delta2[k] * a1[j];
        }

        // Error en la capa oculta (Delta 1)
        vector<double> delta1(HIDDEN_SIZE, 0.0);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double suma_errores = 0.0;
            for (int k = 0; k < OUTPUT_SIZE; k++)
                suma_errores += delta2[k] * red.W2[j][k];
            
            delta1[j] = suma_errores * d_sigmoid(a1[j]);
            
            grad.db1[j] += delta1[j];
            for (int i = 0; i < INPUT_SIZE; i++)
                grad.dW1[i][j] += delta1[j] * matriz_input[idx][i + 1];
        }
    }
}

void aplicar_gradientes(RedNeuronal& red, const Gradientes& total, double tasa, int batch_size) {
    // Aplicamos los gradientes promediados en todo el batch
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        red.b2[k] -= (tasa / batch_size) * total.db2[k];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            red.W2[j][k] -= (tasa / batch_size) * total.dW2[j][k];
    }
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        red.b1[j] -= (tasa / batch_size) * total.db1[j];
        for (int i = 0; i < INPUT_SIZE; i++)
            red.W1[i][j] -= (tasa / batch_size) * total.dW1[i][j];
    }
}

void entrenar_paralelo(const vector<vector<double>>& matriz_input,
                       const vector<vector<int>>& matriz_clase,
                       RedNeuronal& red,
                       int epocas,
                       double tasa,
                       int num_hilos,
                       int tam_lote) {
    const int n = matriz_input.size();
    if (num_hilos < 1) num_hilos = 1;

    for (int ep = 0; ep < epocas; ep++) {
        for (int inicio = 0; inicio < n; inicio += tam_lote) {
            const int fin = min(inicio + tam_lote, n);
            const int muestras = fin - inicio;
            const int hilos_activos = min(num_hilos, muestras);

            vector<Gradientes> grads(hilos_activos);
            vector<thread> hilos;
            hilos.reserve(hilos_activos);

            for (int t = 0; t < hilos_activos; t++) {
                const int desde = inicio + (muestras * t) / hilos_activos;
                const int hasta = inicio + (muestras * (t + 1)) / hilos_activos;
                hilos.emplace_back([&, desde, hasta, t]() {
                    acumular_gradientes(desde, hasta, matriz_input, matriz_clase, red, grads[t]);
                });
            }
            for (auto& h : hilos) h.join();

            Gradientes total;
            total.dW1.assign(INPUT_SIZE, vector<double>(HIDDEN_SIZE, 0.0));
            total.db1.assign(HIDDEN_SIZE, 0.0);
            total.dW2.assign(HIDDEN_SIZE, vector<double>(OUTPUT_SIZE, 0.0));
            total.db2.assign(OUTPUT_SIZE, 0.0);

            for (const auto& g : grads) {
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    total.db2[k] += g.db2[k];
                    for (int j = 0; j < HIDDEN_SIZE; j++) total.dW2[j][k] += g.dW2[j][k];
                }
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    total.db1[j] += g.db1[j];
                    for (int i = 0; i < INPUT_SIZE; i++) total.dW1[i][j] += g.dW1[i][j];
                }
            }
            aplicar_gradientes(red, total, tasa, muestras);
        }
        
        if (ep % 5 == 0) {
            // Evaluamos algunos aciertos para ver progreso rápido usando el mayor valor para la respuesta (argmax)
            int correctos = 0;
            int total_eval = min(n, 1000); // Evaluar con los primeros 1000 para no hacer lento el log
            for(int idx = 0; idx < total_eval; idx++){
                vector<double> a1(HIDDEN_SIZE, 0.0);
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    double suma = red.b1[j];
                    for (int i = 0; i < INPUT_SIZE; i++) suma += matriz_input[idx][i + 1] * red.W1[i][j];
                    a1[j] = sigmoid(suma);
                }
                vector<double> a2(OUTPUT_SIZE, 0.0);
                for (int k = 0; k < OUTPUT_SIZE; k++) {
                    double suma = red.b2[k];
                    for (int j = 0; j < HIDDEN_SIZE; j++) suma += a1[j] * red.W2[j][k];
                    a2[k] = sigmoid(suma);
                }
                int predicho = max_element(a2.begin(), a2.end()) - a2.begin();
                if(matriz_clase[idx][predicho] == 1) correctos++;
            }
            cout << "Epoca " << ep << " finalizada. Precision en muestra: " << correctos << "/" << total_eval << endl;
        }
    }

    // Guardar pesos y sesgos
    ofstream archivo1("mlp_w1.txt");
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) archivo1 << red.W1[i][j] << " ";
        archivo1 << "\n";
    }
    archivo1.close();
    
    ofstream archivo2("mlp_b1.txt");
    for (int j = 0; j < HIDDEN_SIZE; j++) archivo2 << red.b1[j] << " ";
    archivo2.close();

    ofstream archivo3("mlp_w2.txt");
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int k = 0; k < OUTPUT_SIZE; k++) archivo3 << red.W2[j][k] << " ";
        archivo3 << "\n";
    }
    archivo3.close();
    
    ofstream archivo4("mlp_b2.txt");
    for (int k = 0; k < OUTPUT_SIZE; k++) archivo4 << red.b2[k] << " ";
    archivo4.close();
    
    cout << "Entrenamiento completado. Pesos y sesgos guardados." << endl;
}

vector<double> split(string texto, char delimitador){
    vector<double> vec;
    stringstream entrada(texto);
    string porcion;
    while(getline(entrada,porcion,delimitador)){
        vec.push_back(stoi(porcion));
    }
    return vec;
}

vector<vector<int>> build_matriz_clase(int cant_clases, const vector<vector<double>>& entrada_matriz){
    vector<vector<int>> M(entrada_matriz.size(), vector<int>(cant_clases, 0));
    for(size_t i = 0; i < entrada_matriz.size(); i++){
        M[i][(int)entrada_matriz[i][0]] = 1;
    }
    return M;
}

vector<vector<double>> normalizar(vector<vector<double>> matriz, int offset){
    vector<vector<double>> matriz_norma(matriz.size(), vector<double>(matriz[0].size(), 0));
    for (size_t i = 0; i < matriz.size(); i++) {
        matriz_norma[i][0] = matriz[i][0];  
        for (size_t j = offset; j < matriz[0].size(); j++) {
            if(matriz[i][j] > 0)
                matriz_norma[i][j] = matriz[i][j] / 255.0; 
        }
    }
    return matriz_norma;
}

int main(){
    srand(time(0));

    RedNeuronal red;
    inicializar_red(red);

    cout << "Cargando mnist_train.csv..." << endl;
    ifstream archivo("mnist_train.csv");
    string linea;

    vector<vector<double>> entrada_matriz;
    while (getline(archivo,linea)){
        vector<double> vec_num = split(linea,',');
        entrada_matriz.push_back(vec_num);
    }
    archivo.close();
    
    cout << "Normalizando..." << endl;
    entrada_matriz = normalizar(entrada_matriz, 1);
    vector<vector<int>> matriz_clase = build_matriz_clase(10, entrada_matriz);

    const int num_hilos = thread::hardware_concurrency();
    const int tam_lote  = 600;  

    cout << "Entrenando con " << num_hilos << " hilos, lotes de "
         << tam_lote << " muestras..." << endl;
         
    // 50 epocas, y tasa de aprendizaje de 0.5 (puede variar segun los resultados)
    entrenar_paralelo(entrada_matriz, matriz_clase, red, 50, 0.5, num_hilos, tam_lote);

    return 0;
}
