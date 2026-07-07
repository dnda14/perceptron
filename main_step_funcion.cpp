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


vector<double> split_linea(string linea){
    stringstream linea_input(linea);
    string numero_string;
    vector<double> lista_numeros;

    while(getline(linea_input,numero_string,',')){
        lista_numeros.push_back(stod(numero_string));
    }
    return lista_numeros;
}
void normalizar(vector<double> &lista){
    for(int i=1; i < lista.size(); i++){
        lista[i] = lista[i] / 255.0;
    }
}

void train(vecto<vector<double>> input_data,
     vector<vector<double>> pesos,
    int epocas,
    double tasa_ap){


        
    }

int main(){
    srand(time(0));
    
    vector<vecto<double>>pesos(784,vector<double>(10,0));
    vector<double bias(10,0);

    for(int i=0;i<784;i++){
        for(int j=0;j<10;j++){
            pesos[i][j]=(double)(rand() % 1000)/50000;
        }
    }

    for(int i=0;i<10;i++){
            bias[i]=(double)(rand() % 1000)/50000;

    }


    ifstream archivo("mnist_train.csv");

    vector<vector<double>> input_data;
    string linea;

    while(getline(archivo, linea)){
        vector<double> pesos_linea = split_linea(linea);
        normalizar(pesos_linea);
        input_data.push_back(pesos_linea);
    }



    train(input_data,pesos,bias,50,0.01)




    return 0;
}