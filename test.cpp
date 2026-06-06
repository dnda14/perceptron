#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace std;

vector<vector<double>> normalizar(vector<vector<double>> matriz,int offset){
    vector<vector<double>> matriz_norma(matriz.size(),vector<double>(matriz[0].size(),0));
    for (int i = 0; i < matriz.size(); i++)
    {
        for (int j = 0 + offset; j < matriz[0].size(); j++)
        {
            if(matriz[i][j]>0)
                matriz_norma[i][j] = matriz[i][j]/256;
        }
        
    }
    return matriz_norma;
    
}
vector<double> split(string texto, char delimitador){
    vector<double> vec;
    stringstream entrada(texto);
    string porcion;

    while(getline(entrada,porcion,delimitador)){

        vec.push_back(stod(porcion));
        
    }
    return vec;
}

void test(vector<vector<double>> pesos,vector<vector<double>> matriz_input, vector<double> bias){
    vector<vector<double>> total(1000,vector<double>(10,0));
    for (int i = 0; i < 1000; i++)
    {
        int contador_aciertos=0;
        for (int k = 0; k < 10; k++)
        {
            for(int j=0;j<784;j++){
                
                total[i][k] += matriz_input[i][j+1]*pesos[j][k];
                
            }
            total[i][k]+=bias[k];
        }
    }

    ofstream archivo("resultados.txt",std::ios::app);
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            
            archivo<<total[i][j];
            archivo<<" ";
        }
        archivo<<endl;
        
    }
    archivo.close();

    
}

int main(){

    ifstream archivo("pesos.txt");

    vector<vector<double>> pesos;
    string linea;
    while(getline(archivo,linea)){
        vector<double> vec_num = split(linea,' ');
        pesos.push_back(vec_num);
    }
    archivo.close();

    ifstream archivo2("mnist_test.csv");

    vector<vector<double>> matriz_input;
    string linea2;
    while(getline(archivo2,linea2)){
        vector<double> vec_num = split(linea2,',');
        matriz_input.push_back(vec_num);
    }
    archivo2.close();

    ifstream archivo3("bias.txt");

    

    vector<double> bias ;
    string linea3;
    while(getline(archivo3,linea3)){
        bias = split(linea3,' ');
    }
    archivo3.close();


    test(pesos,matriz_input,bias);

    return 0;


}