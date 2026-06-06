#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>

using namespace std;

double f_activacion(double valor){
    return 1.0 / (1.0 + exp(-valor));
}
vector<vector<double>> pesos(784,vector<double>(10));

vector<double> bias(10,0);


void entrenar(vector<vector<double>> matriz_input, 
             vector<vector<int>> matriz_clase,
             int epocas, 
             double tasa){
    
    for(int i=0;i<epocas;i++){
        vector<bool> y(10,0);
        for (int j = 0; j < 60000; j++)
        {
            vector<double> total_peso_input(10,0);

            for (int m = 0; m < 10; m++)
            {
                for (int k = 1; k < 785; k++)
                {
                    //pesos tiene el orden de input x clase 
                    total_peso_input[m] += matriz_input[j][k]*pesos[k-1][m];
                    
                }

                
                y[m]=f_activacion(total_peso_input[m] + bias[m]);

                double error =  matriz_clase[j][m] - y[m] ;

                bias[m] +=  tasa * error;

                for (int k = 0; k < 784; k++)
                {
                    pesos[k][m]+= error*tasa*matriz_input[j][k+1];

                }
            }
            

            
        }
        
    }
    cout<<"dsds"<<endl;
    ofstream archivo("pesos.txt",std::ios::app);
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

    ofstream archivo2("bias.txt",std::ios::app);
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

    entrenar(entrada_matriz,matriz_clase,1,0.001);

    return 0;
}