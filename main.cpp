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
    return 1.0 / (1.0 + exp(-valor));
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
            double delta = error * y[m] * (1.0 - y[m]);
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
    }

    ofstream archivo("pesos.txt");
    for (int i = 0; i < 784; i++) {
        for (int j = 0; j < 10; j++) {
            archivo << pesos[i][j] << " ";
        }
        archivo << endl;
    }
    archivo.close();

    ofstream archivo2("bias.txt");
    for (int i = 0; i < 10; i++)
        archivo2 << bias[i] << " ";
    archivo2.close();
}

void entrenar(vector<vector<double>> matriz_input, 
             vector<vector<int>> matriz_clase,
             int epocas, 
             double tasa){
    
    for(int i=0;i<epocas;i++){
        vector<double> y(10,0.0);
        for (int j = 0; j < 60000; j++)
        {
            vector<double> total_peso_input(10,0);

            for (int m = 0; m < 10; m++)
            {
                double suma = bias[m];
                for (int k = 0; k < 784; k++)
                    suma += matriz_input[j][k+1] * pesos[k][m];
                y[m] = f_activacion(suma);
            }
                
            for (int m = 0; m < 10; m++) {
                double error = matriz_clase[j][m] - y[m];
                double delta = error * y[m] * (1.0 - y[m]);
                bias[m] += tasa * delta;

                for (int k = 0; k < 784; k++)
                    pesos[k][m] += tasa * delta * matriz_input[j][k+1];
            }

            
        }
        
    }
    cout<<"dsds"<<endl;
    ofstream archivo("pesos.txt");
    for (int i = 0; i < 784; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            
            archivo<<pesos[i][j];
            archivo<<" ";
        }
        archivo<<endl;
        
    }
    archivo.close();

    ofstream archivo2("bias.txt");
    for (int i = 0; i < 10; i++)
    {
        archivo2<<bias[i];
        archivo2<<" ";
    }
    
    archivo2.close();
    
    
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
    cout<<"hola"<<endl;
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

    ifstream archivo("mnist_train.csv");
    string linea;

    vector<vector<double>> entrada_matriz;
    while (getline(archivo,linea)){
        //texto.erase(remove(texto.begin(), texto.end(), ','), texto.end());
        vector<double> vec_num = split(linea,',');
        int sz = vec_num.size();
        entrada_matriz.push_back(vec_num);
        
    }
    entrada_matriz = normalizar(entrada_matriz,1);
    vector<vector<int>> matriz_clase = build_matriz_clase(10,entrada_matriz);
    cout<<matriz_clase[0][0]<<endl;
    archivo.close();

    const int num_hilos = thread::hardware_concurrency();
    const int tam_lote  = 600;  

    cout << "Entrenando con " << num_hilos << " hilos, lotes de "
         << tam_lote << " muestras..." << endl;

    entrenar_paralelo(entrada_matriz, matriz_clase, 100, 0.002, num_hilos, tam_lote);
    // entrenar(entrada_matriz, matriz_clase, 20, 0.001);  // version secuencial

    return 0;
}