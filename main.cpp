#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace std;

int estado_activacion(double valor){
    return valor>=0?1:0;
}

int entrenar(vector<vector<int>> matriz_muestra,vector<vector<int>> matriz_etiquetas, int epocas, float tasa)
vector<int> split(string texto, char delimitador){
    vector<int> vec;
    stringstream entrada(texto);
    string porcion;

    while(getline(entrada,porcion,delimitador)){

        vec.push_back(stoi(porcion));
        
    }
    return vec;
}
vector<vector<int>> build_matriz_etiquetas(int cant_clases,vector<vector<int> >muestras){
    vector<vector<int>> M(60000,vector<int>(cant_clases,0));
    cout<<"hola"<<endl;
    for(int i=0;i<60000;i++){
        for(int j=0;j<cant_clases;j++){
            if(j==muestras[i][0]){
                M[i][j]=1;
                break;
            }
        }
    }
    return M;
}

int main(){
    ifstream archivo("mnist_train.csv");
    string linea;

    vector<vector<int>> muestras;
    while (getline(archivo,linea)){
        //texto.erase(remove(texto.begin(), texto.end(), ','), texto.end());
        vector<int> vec_num = split(linea,',');
        int sz = vec_num.size();
        muestras.push_back(vec_num);
        
    }
    vector<vector<int>> matriz_etiquetas = build_matriz_etiquetas(10,muestras);
    cout<<matriz_etiquetas[0][0]<<endl;
    archivo.close();

    return 0;
}