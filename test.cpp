#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>  

using namespace std;

double f_activacion(double valor) {
    return 1.0 / (1.0 + exp(-valor));
}

vector<vector<double>> normalizar(vector<vector<double>> matriz, int offset) {
    vector<vector<double>> out(matriz.size(), vector<double>(matriz[0].size(), 0));
    for (int i = 0; i < matriz.size(); i++)
    {
        out[i][0] = matriz[i][0];
        for (int j = offset; j < matriz[0].size(); j++)
            if (matriz[i][j] > 0)
                out[i][j] = matriz[i][j] / 256.0;
        }
    return out;
}

vector<double> split(string texto, char delimitador) {
    vector<double> vec;
    stringstream entrada(texto);
    string porcion;
    while (getline(entrada, porcion, delimitador))
        vec.push_back(stod(porcion));
    return vec;
}

void test(vector<vector<double>> pesos,
          vector<vector<double>> matriz_input,
          vector<double> bias) {

    int total     = matriz_input.size();
    int correctos = 0;

    vector<int> por_clase_total(10, 0);
    vector<int> por_clase_ok   (10, 0);

    for (int i = 0; i < total; i++) {
        int etiqueta = (int)matriz_input[i][0];   

        // Forward pass con sigmoid
        vector<double> salidas(10, 0.0);
        for (int k = 0; k < 10; k++) {
            double suma = bias[k];
            for (int j = 0; j < 784; j++)
                suma += matriz_input[i][j + 1] * pesos[j][k];
            salidas[k] = f_activacion(suma);     
        }

        int predicho = max_element(salidas.begin(), salidas.end())
                       - salidas.begin();         

        por_clase_total[etiqueta]++;
        if (predicho == etiqueta) {
            correctos++;
            por_clase_ok[etiqueta]++;
        }
    }

    cout << "\n=== Resultados ===\n";
    cout << "Correctos : " << correctos << " / " << total << "\n";
    cout << "Accuracy  : " << 100.0 * correctos / total << " %\n\n";

    cout << "Accuracy por clase:\n";
    for (int c = 0; c < 10; c++) {
        double acc = por_clase_total[c] > 0
                     ? 100.0 * por_clase_ok[c] / por_clase_total[c]
                     : 0.0;
        cout << "  Clase " << c << " : "
             << por_clase_ok[c] << "/" << por_clase_total[c]
             << "  (" << acc << " %)\n";
    }
}

int main() {
    ifstream f1("pesos.txt");
    vector<vector<double>> pesos;
    string linea;
    while (getline(f1, linea))
        pesos.push_back(split(linea, ' '));
    f1.close();

    ifstream f2("mnist_test.csv");
    vector<vector<double>> matriz_input;
    while (getline(f2, linea))
        matriz_input.push_back(split(linea, ','));
    f2.close();

    ifstream f3("bias.txt");
    vector<double> bias;
    while (getline(f3, linea))
        bias = split(linea, ' ');
    f3.close();

    matriz_input = normalizar(matriz_input, 1);

    test(pesos, matriz_input, bias);
    return 0;
}