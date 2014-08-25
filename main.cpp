/*
||=========================================================================||
|| Tarea 1 - CC5509 Reconocimiento de Patrones                             ||
|| Autor: Sebastian Gonzalez                                               ||
|| - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ||
|| El programa consiste en extraer caracteristicas de concavidad, sobre    ||
|| imagenes de digitos impresos. Usando los metodos: 13bins, 4CC y 8C.     ||
|| El clasificador a utilizar sera KNN con distancias Manhattan y Eucli-   ||
|| diana.                                                                  ||
||                                                                         ||
||=========================================================================||
*/
#ifndef _opencv
#define _opencv
#include <opencv2/opencv.hpp>
#endif
#include <opencv2/ml/ml.hpp>

#ifndef _string
#define _string
#include <string>
#endif
#include <cstring>

#include <iostream>
#include <fstream>

#include "concavity.cpp"
#include "knn.cpp"

void calculateConfusionMatrix(std::vector<std::vector <ImageResults> >* results,int k,const char *filename){
    float matC13[10][10];
    float matCC4[10][10];
    float matCC8[10][10];
    for(int i=0;i<(*results).size();i++){
        for(int j=0;j<k;j++){
            ImageResults img = (*results).at(i).at(j);
            
            matC13[img.actualClass][img.classC13]++;
            matCC4[img.actualClass][img.classCC4]++;
            matCC8[img.actualClass][img.classCC8]++;
        }
    }
    for(int i=0;i<10;i++){
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
        for(int j=0;j<10;j++){
            sum1 += matC13[i][j];
            sum2 += matCC4[i][j];
            sum3 += matCC8[i][j];
        }
        for(int j=0;j<10;j++){
            matC13[i][j] /= sum1;
            matCC4[i][j] /= sum2;
            matCC8[i][j] /= sum3;
        }
    }
    //guardar los resultados
    std::ofstream out(filename, std::ios::out);

    //out<<"Confusion C13\n";
    out<<"Actual-Calculated";
    for(int i=0;i<10;i++){
        out<<","<<i;
    }
    out<<"\n";
    for(int i=0;i<10;i++){
        out<<i;
        for(int j=0;j<10;j++){
            out<<","<<matC13[i][j];
        }
        out<<"\n";
    }
    //out<<"Confusion CC4\n";
    out<<"Actual-Calculated";
    for(int i=0;i<10;i++){
        out<<","<<i;
    }
    out<<"\n";
    for(int i=0;i<10;i++){
        out<<i;
        for(int j=0;j<10;j++){
            out<<","<<matCC4[i][j];
        }
        out<<"\n";
    }
    //out<<"Confusion CC8\n";
    out<<"Actual-Calculated";
    for(int i=0;i<10;i++){
        out<<","<<i;
    }
    out<<"\n";
    for(int i=0;i<10;i++){
        out<<i;
        for(int j=0;j<10;j++){
            out<<","<<matCC8[i][j];
        }
        out<<"\n";
    }
    out.close();
}

int main (int argc, char** argv){
    int error=0;
    std::vector<ImageInfo> imageIndex;
    //Train index creation:
    for(int i=1;i<argc;i++){
        if(strcmp("-t",argv[i])==0){
            loadTrainPaths(&imageIndex,"train.txt");
        }
        else if(strcmp("-l",argv[i])==0){
            readIndexFromFile(&imageIndex,"Index");
        }
        else if(strcmp("-w",argv[i])==0){
            writeIndexToFile(&error,imageIndex,"Index");
        }else{
            std::cout << "Use:\n\tTo load training images: -t\n\tTo use index file: -l\n\tTo write index file: -w"<<std::endl;
        }
    }
    
    //Actual program:
    KNearestNeighbors knn;
    knn.train(&imageIndex);
    //get_nearest(ImageInfo* image,int k,ImageResults* resultsManhattan,ImageResults* resultsEuclidian)
    
    std::vector<ImageInfo> testIndex;
    loadTrainPaths(&testIndex,"test.txt");
    
    std::vector<std::vector <ImageResults> > man;
    std::vector<std::vector <ImageResults> > euc;
    
    int k=1;
    
    std::cout << "Beginning work on KNN." << std::endl;
    for(int i=0;i<testIndex.size();i++){
        if(i%500==0) std::cout << "\tHad processed " << i << " images." << std::endl;
        ImageInfo img = testIndex.at(i);
        
        std::vector<ImageResults> manR;
        std::vector<ImageResults> eucR;
        
        knn.get_nearest(&img,k,&manR,&eucR);
        
        man.push_back(manR);
        euc.push_back(eucR);
    }
    std::cout << "Finished work on KNN." << std::endl;
    
    std::cout << "Calculating Confusion Matrix." << std::endl;
    std::cout << "\tManhattan." << std::endl;
    calculateConfusionMatrix(&man,k,"Manhattan.csv");
    std::cout << "\tEuclidian." << std::endl;
    calculateConfusionMatrix(&euc,k,"Euclidian.csv");
    std::cout << "Finished Confusion Matrix." << std::endl;
    
    return error;
}
