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

double f_activacion(double valor){
    return (valor >= 0.0) ? 1.0 : 0.0;
}

vector<vector<double>> pesos(784,vector<double>(10));
vector<double> bias(10,0);

struct Gradientes {
    vector<vector<double>> dpesos;
    vector<double> dbias;
};

void acumular_gradientes(int desde, int hasta,
                         const vector<vector<double>>& matriz_input,
                         const vector<vector<int>>& matriz_clase,
                         const vector<vector<double>>& pesos_loc,
                         const vector<double>& bias_loc,
                         Gradientes& grad) {
    grad.dpesos.assign(784, vector<double>(10, 0.0));
    grad.dbias.assign(10, 0.0);

    for (int j = desde; j < hasta; j++) {
        vector<double> y(10);
        for (int m = 0; m < 10; m++) {
            double suma = bias_loc[m];
            for (int k = 0; k < 784; k++)
                suma += matriz_input[j][k + 1] * pesos_loc[k][m];
            y[m] = f_activacion(suma);
        }
        
        for (int m = 0; m < 10; m++) {
            double error = matriz_clase[j][m] - y[m];
            
            // 2. CAMBIO: Regla Clásica del Perceptrón (sin derivada)
            // En vez de: double delta = error * y[m] * (1.0 - y[m]);
            double delta = error; 
            
            grad.dbias[m] += delta;
            for (int k = 0; k < 784; k++)
                grad.dpesos[k][m] += delta * matriz_input[j][k + 1];
        }
    }
}

void aplicar_gradientes(const Gradientes& grad, double tasa) {
    for (int m = 0; m < 10; m++) {
        bias[m] += tasa * grad.dbias[m];
        for (int k = 0; k < 784; k++)
            pesos[k][m] += tasa * grad.dpesos[k][m];
    }
}

void entrenar_paralelo(const vector<vector<double>>& matriz_input,
                       const vector<vector<int>>& matriz_clase,
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
                    acumular_gradientes(desde, hasta, matriz_input, matriz_clase,
                                        pesos, bias, grads[t]);
                });
            }
            for (auto& h : hilos) h.join();

            Gradientes total;
            total.dpesos.assign(784, vector<double>(10, 0.0));
            total.dbias.assign(10, 0.0);
            for (const auto& g : grads) {
                for (int m = 0; m < 10; m++) {
                    total.dbias[m] += g.dbias[m];
                    for (int k = 0; k < 784; k++)
                        total.dpesos[k][m] += g.dpesos[k][m];
                }
            }
            aplicar_gradientes(total, tasa);
        }
        if(ep % 10 == 0) cout << "Epoca " << ep << " finalizada." << endl;
    }

    ofstream archivo("pesos_step.txt");
    for (int i = 0; i < 784; i++) {
        for (int j = 0; j < 10; j++) {
            archivo << pesos[i][j] << " ";
        }
        archivo << endl;
    }
    archivo.close();

    ofstream archivo2("bias_step.txt");
    for (int i = 0; i < 10; i++)
        archivo2 << bias[i] << " ";
    archivo2.close();
    
    cout << "Entrenamiento completado. Guardado en pesos_step.txt y bias_step.txt" << endl;
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

vector<vector<int>> build_matriz_clase(int cant_clases,vector<vector<double> >entrada_matriz){
    vector<vector<int>> M(60000,vector<int>(cant_clases,0));
    for(int i=0;i<60000;i++){
        M[i][entrada_matriz[i][0]]=1;
    }
    return M;
}

vector<vector<double>> normalizar(vector<vector<double>> matriz,int offset){
    vector<vector<double>> matriz_norma(matriz.size(),vector<double>(matriz[0].size(),0));
    for (int i = 0; i < matriz.size(); i++)
    {
        matriz_norma[i][0] = matriz[i][0];  
        for (int j = 0 + offset; j < matriz[0].size(); j++)
        {
            if(matriz[i][j]>0)
                matriz_norma[i][j] = matriz[i][j]/256;
        }
    }
    return matriz_norma;
}

int main(){
    srand(time(0));

    for(int i=0; i<784; i++) {
        for(int j=0; j<10; j++) {
            pesos[i][j] = (double)(rand() % 1000) / 50000.0; 
        }
    }
    for(int j=0; j<10; j++) {
        bias[j]= (double)(rand() % 1000) / 50000.0; 
    }

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
    entrada_matriz = normalizar(entrada_matriz,1);
    vector<vector<int>> matriz_clase = build_matriz_clase(10,entrada_matriz);

    const int num_hilos = thread::hardware_concurrency();
    const int tam_lote  = 600;  

    cout << "Entrenando con " << num_hilos << " hilos, lotes de "
         << tam_lote << " muestras..." << endl;
         
    entrenar_paralelo(entrada_matriz, matriz_clase, 50, 0.01, num_hilos, tam_lote);

    return 0;
}
