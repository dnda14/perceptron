#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace std;

// Funcion de activacion sigmoide
double f_activacion(double valor) {
    return 1.0 / (1.0 + exp(-valor));
}

// Funcion para separar strings por delimitador (segura contra espacios vacios)
vector<double> split(string texto, char delimitador) {
    vector<double> vec;
    stringstream entrada(texto);
    string porcion;
    while (getline(entrada, porcion, delimitador)) {
        // Ignorar espacios en blanco o retornos de carro
        if (!porcion.empty() && porcion != "\r" && porcion != " ") {
            try {
                vec.push_back(stod(porcion));
            } catch (...) {}
        }
    }
    return vec;
}

int main(int argc, char* argv[]) {
    // Por defecto lee imagen.txt, pero el usuario puede pasar otro archivo
    string archivo_imagen = "imagen.txt";
    if (argc > 1) {
        archivo_imagen = argv[1];
    }

    // 1. Cargar la inteligencia del modelo (pesos)
    ifstream f1("pesos.txt");
    if (!f1.is_open()) {
        cout << "Error: No se encontro el archivo pesos.txt. Entrena el modelo primero." << endl;
        return 1;
    }
    vector<vector<double>> pesos;
    string linea;
    while (getline(f1, linea)) {
        if (linea.empty() || linea == "\r") continue;
        pesos.push_back(split(linea, ' '));
    }
    f1.close();

    // 2. Cargar los sesgos (bias)
    ifstream f3("bias.txt");
    if (!f3.is_open()) {
        cout << "Error: No se encontro el archivo bias.txt. Entrena el modelo primero." << endl;
        return 1;
    }
    vector<double> bias;
    if (getline(f3, linea)) {
        bias = split(linea, ' ');
    }
    f3.close();

    // 3. Cargar UN SOLO mapa de pixeles (la imagen del usuario)
    ifstream f2(archivo_imagen);
    if (!f2.is_open()) {
        cout << "Error: No se encontro la imagen " << archivo_imagen << endl;
        return 1;
    }
    vector<double> imagen;
    if (getline(f2, linea)) {
        // Separamos los 784 pixeles por coma (asumiendo que vienen en CSV)
        // Puedes cambiar ',' por ' ' si tu imagen viene separada por espacios
        vector<double> raw_pixels = split(linea, ',');
        
        // Normalizamos los pixeles (igual que en el entrenamiento)
        for (double p : raw_pixels) {
            if (p > 0) imagen.push_back(p / 256.0);
            else imagen.push_back(0.0);
        }
    }
    f2.close();

    // Validar que la imagen tenga el tamaño correcto
    if (imagen.size() != 784) {
        cout << "Error: La imagen debe tener exactamente 784 pixeles. Tu archivo tiene " << imagen.size() << endl;
        return 1;
    }

    // 4. Calcular el resultado (Feedforward)
    vector<double> salidas(10, 0.0);
    for (int k = 0; k < 10; k++) {
        double suma = bias[k];
        // Multiplicar los 784 pixeles de la imagen por los 784 pesos del numero k
        for (int j = 0; j < 784; j++) {
            suma += imagen[j] * pesos[j][k];
        }
        // Pasar por la funcion sigmoide para tener un porcentaje de probabilidad
        salidas[k] = f_activacion(suma);
    }

    // 5. Elegir el numero con la probabilidad mas alta
    int prediccion = max_element(salidas.begin(), salidas.end()) - salidas.begin();

    // 6. Mostrar el resultado
    cout << "\n=======================================" << endl;
    cout << "  PREDICCION DEL MODELO DE PERCEPTRON  " << endl;
    cout << "=======================================" << endl;
    cout << "  El numero en la imagen es: -> " << prediccion << " <-" << endl;
    cout << "=======================================\n" << endl;
    
    cout << "Probabilidades que penso el modelo para cada numero:" << endl;
    for (int k = 0; k < 10; k++) {
        cout << " - Numero " << k << ": " << salidas[k] * 100.0 << " %" << endl;
    }
    cout << endl;

    return 0;
}
