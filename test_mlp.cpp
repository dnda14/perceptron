#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

// Función de activación
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Arquitectura (Debe coincidir con la de main_mlp.cpp)
const int INPUT_SIZE = 784;
const int HIDDEN_SIZE = 64; 
const int OUTPUT_SIZE = 10;

// Variables para almacenar los pesos y sesgos
vector<vector<double>> W1(INPUT_SIZE, vector<double>(HIDDEN_SIZE));
vector<double> b1(HIDDEN_SIZE);
vector<vector<double>> W2(HIDDEN_SIZE, vector<double>(OUTPUT_SIZE));
vector<double> b2(OUTPUT_SIZE);

// Funciones utilitarias para leer los archivos
bool cargar_matriz(const string& nombre_archivo, vector<vector<double>>& matriz) {
    ifstream archivo(nombre_archivo);
    if (!archivo.is_open()) return false;
    for (size_t i = 0; i < matriz.size(); i++) {
        for (size_t j = 0; j < matriz[0].size(); j++) {
            archivo >> matriz[i][j];
        }
    }
    return true;
}

bool cargar_vector(const string& nombre_archivo, vector<double>& vec) {
    ifstream archivo(nombre_archivo);
    if (!archivo.is_open()) return false;
    for (size_t i = 0; i < vec.size(); i++) {
        archivo >> vec[i];
    }
    return true;
}

vector<double> split(string texto, char delimitador) {
    vector<double> vec;
    stringstream entrada(texto);
    string porcion;
    while (getline(entrada, porcion, delimitador)) {
        vec.push_back(stod(porcion));
    }
    return vec;
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

void test(const vector<vector<double>>& matriz_input) {
    int total = matriz_input.size();
    int correctos = 0;
    
    // Contadores para desglosar la precision por clase
    vector<int> por_clase_total(10, 0);
    vector<int> por_clase_ok(10, 0);

    for (int idx = 0; idx < total; idx++) {
        int etiqueta = (int)matriz_input[idx][0];
        por_clase_total[etiqueta]++;

        // 1. Forward en la Capa Oculta
        vector<double> a1(HIDDEN_SIZE, 0.0);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double suma = b1[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                suma += matriz_input[idx][i + 1] * W1[i][j];
            }
            a1[j] = sigmoid(suma);
        }

        // 2. Forward en la Capa de Salida
        vector<double> a2(OUTPUT_SIZE, 0.0);
        for (int k = 0; k < OUTPUT_SIZE; k++) {
            double suma = b2[k];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                suma += a1[j] * W2[j][k];
            }
            a2[k] = sigmoid(suma);
        }

        // Predicción: tomar el índice con el valor máximo
        int predicho = max_element(a2.begin(), a2.end()) - a2.begin();

        if (predicho == etiqueta) {
            correctos++;
            por_clase_ok[etiqueta]++;
        }
    }

    cout << "--------------------------------------------------\n";
    cout << "Resultados Totales:\n";
    cout << "Precision Total: " << correctos << " / " << total 
         << " (" << (double)correctos / total * 100.0 << "%)\n\n";

    cout << "Resultados Por Clase:\n";
    for (int i = 0; i < 10; i++) {
        double porc = (por_clase_total[i] > 0) ? (double)por_clase_ok[i] / por_clase_total[i] * 100.0 : 0.0;
        cout << "Clase " << i << ": " << por_clase_ok[i] << " / " << por_clase_total[i] 
             << " (" << porc << "%)\n";
    }
    cout << "--------------------------------------------------\n";
}

int main() {
    cout << "Cargando parametros del modelo..." << endl;
    if (!cargar_matriz("mlp_w1.txt", W1) || !cargar_vector("mlp_b1.txt", b1) ||
        !cargar_matriz("mlp_w2.txt", W2) || !cargar_vector("mlp_b2.txt", b2)) {
        cout << "Error: No se pudieron cargar los archivos de pesos/sesgos." << endl;
        return 1;
    }

    cout << "Cargando mnist_test.csv..." << endl;
    ifstream archivo("mnist_test.csv");
    if (!archivo.is_open()) {
        cout << "Error al abrir mnist_test.csv" << endl;
        return 1;
    }

    string linea;
    vector<vector<double>> entrada_matriz;
    while (getline(archivo, linea)) {
        vector<double> vec_num = split(linea, ',');
        entrada_matriz.push_back(vec_num);
    }
    archivo.close();

    cout << "Normalizando datos de prueba..." << endl;
    entrada_matriz = normalizar(entrada_matriz, 1);

    cout << "Evaluando " << entrada_matriz.size() << " muestras..." << endl;
    test(entrada_matriz);

    return 0;
}
