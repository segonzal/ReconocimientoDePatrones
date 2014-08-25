#include <queue>
#include <iterator>

struct ImageInfo{
    int class_name;
    float C13[13];
    float CC4[16];
    float CC8[16];
    char path[42];
};

void createDescriptors(std::string path,std::vector<ImageInfo>* index){
    //Our color image
    cv::Mat imageMat = cv::imread(path,CV_LOAD_IMAGE_COLOR);
    if(imageMat.empty()){
        std::cerr << "ERROR: Could not read image " << path << std::endl;
    }
    
    cv::Mat binaryMat = toBinary(imageMat);
    float C13[13];
    float CC4[16];
    float CC8[16];
    
    concavity13C(binaryMat,C13);
    concavity4CC(binaryMat,CC4);
    concavity8CC(binaryMat,CC8);
    
    ImageInfo info;
    info.class_name = (int)(path.at(23)-'0');
    std::copy(C13,C13+13,info.C13);
    std::copy(CC4,CC4+16,info.CC4);
    std::copy(CC8,CC8+16,info.CC8);
    
    std::strcpy(info.path,path.c_str());
    (*index).push_back(info);
}

void loadTrainPaths(std::vector<ImageInfo>* index,const char* filename){
    std::cout << "Loading images... (" << filename <<")" <<std::endl;
    std::ifstream in(filename);
    std::string line;
    int i=0;
    
    while (std::getline(in, line)){
        if(i%500==0){
            std::cout << "\tHad processed " << i << " images."<< std::endl;
        }
        createDescriptors(line,index);
        i++;
    }
    in.close();
    std::cout << "Done loading training images. (total: " << i << ")" << std::endl;
}

void writeIndexToFile(int *error,std::vector<ImageInfo> index,const char* filename){
    std::ofstream out(filename,std::ios::out | std::ios::binary);
    if(!out){
        std::cout << "Cannot open file.";
        (*error) = 1;
    }
    
    //escribir el numero de elementos en el vector.
    int N = index.size();
    std::cout << "Saving\nQuantity of elements to save: " << N << std::endl;
    out.write((char *) &N,sizeof N);
    //iterar por cada elemento del vector.
    for(int i=0;i<N;i++){
        //escribir la estructura.
        ImageInfo info = index.at(i);
        out.write((char *) &info,sizeof info);
    }
    std::cout << "Done saving." << std::endl;
    out.close();
}

void readIndexFromFile(std::vector<ImageInfo>* index,const char* filename){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    
    std::cout << "Loading Index from file." << std::endl;
    
    int N;
    in.read((char *) &N, sizeof N);
    
    for(int i=0;i<N;i++){
        ImageInfo info;
        
        in.read((char *) &info, sizeof info);
        
        (*index).push_back(info);
    }
    
    std::cout << "Done loading Index from file." << std::endl;
    
    in.close();
}

float distanceManhattan(float* arr1,float* arr2,int n){
    float sum=0;
    for(int i=0;i<n;i++){
        sum += (arr1[i]-arr2[i])>0?(arr1[i]-arr2[i]):(arr2[i]-arr1[i]);
    }
    return sum;
}

float distanceEuclidian(float* arr1,float* arr2,int n){
    float sum=0;
    for(int i=0;i<n;i++){
        float a = arr1[i]-arr2[i];
        sum += (a*a);
    }
    return std::sqrt(sum);
}

struct ImageResults{
    char path[42];//name of image
    int actualClass;
    
    float dc13;
    int classC13;
    char pc13[42];
    
    float dcc4;
    int classCC4;
    char pcc4[42];
    
    float dcc8;
    int classCC8;
    char pcc8[42];
};

class KNearestNeighbors{
    private:
        std::vector<ImageInfo>* trainSet;
    public:
        void train(std::vector<ImageInfo>*);
        void get_nearest(ImageInfo*,int,std::vector<ImageResults>*,std::vector<ImageResults>*);
};

void KNearestNeighbors::train(std::vector<ImageInfo>* image){
    trainSet = image;
}

struct ElementData{
    char path[42];
    int cname;
    float distance;
};

class CompareElementData {
public:
    bool operator()(ElementData& t1, ElementData& t2){
       return (t1.distance > t2.distance);
    }
};

void KNearestNeighbors::get_nearest(ImageInfo* image,int k,std::vector<ImageResults>* resultsManhattan,std::vector<ImageResults>* resultsEuclidian){
    std::priority_queue<ElementData, std::vector<ElementData>, CompareElementData> ManC13;
    std::priority_queue<ElementData, std::vector<ElementData>, CompareElementData> ManCC4;
    std::priority_queue<ElementData, std::vector<ElementData>, CompareElementData> ManCC8;
    std::priority_queue<ElementData, std::vector<ElementData>, CompareElementData> EucC13;
    std::priority_queue<ElementData, std::vector<ElementData>, CompareElementData> EucCC4;
    std::priority_queue<ElementData, std::vector<ElementData>, CompareElementData> EucCC8;
    //por cada elemento en el trainSet
    for(int i=0;i<(*trainSet).size();i++){
        ImageInfo info = (*trainSet).at(i);
        //  calcular distancia a image en C13, CC4 y CC8
        //---Manhattan---
        //>C13
        ElementData emc13 = {};
        std::copy(info.path, info.path + 42, emc13.path);
        emc13.cname = info.class_name;
        emc13.distance = distanceManhattan(image->C13,info.C13,13);
        ManC13.push(emc13);
        //>CC4
        ElementData emcc4 = {};
        std::copy(info.path, info.path + 42, emcc4.path);
        emcc4.cname = info.class_name;
        emcc4.distance = distanceManhattan(image->CC4,info.CC4,16);
        ManCC4.push(emcc4);
        //>CC8
        ElementData emcc8 = {};
        std::copy(info.path, info.path + 42, emcc8.path);
        emcc8.cname = info.class_name;
        emcc8.distance = distanceManhattan(image->CC8,info.CC8,16);
        ManCC8.push(emcc8);
        //---Euclidian---
        ElementData eec13 = {};
        std::copy(info.path, info.path + 42, eec13.path);
        eec13.cname = info.class_name;
        eec13.distance = distanceEuclidian(image->C13,info.C13,13);
        EucC13.push(eec13);
        //>CC4
        ElementData eecc4 = {};
        std::copy(info.path, info.path + 42, eecc4.path);
        eecc4.cname = info.class_name;
        eecc4.distance = distanceEuclidian(image->CC4,info.CC4,16);
        EucCC4.push(eecc4);
        //>CC8
        ElementData eecc8 = {};
        std::copy(info.path, info.path + 42, eecc8.path);
        eecc8.cname = info.class_name;
        eecc8.distance = distanceEuclidian(image->CC8,info.CC8,16);
        EucCC8.push(eecc8);
        //  pushear cada elemento calculado en la cola respectiva
    }
    //sacar un elemento de cada cola y colocarlo en el arreglo de salida
    for(int i=0;i<k;i++){
        ElementData emc13 = ManC13.top();
        ElementData emcc4 = ManCC4.top();
        ElementData emcc8 = ManCC8.top();

        ImageResults man = {};
        std::copy(image->path, image->path + 42, man.path);
        man.actualClass = image->class_name;
        man.dc13 = emc13.distance;
        man.classC13 = emc13.cname;
        std::copy(emc13.path, emc13.path + 42, man.pc13);
        man.dcc4 = emcc4.distance;
        man.classCC4 = emcc4.cname;
        std::copy(emcc4.path, emcc4.path + 42, man.pcc4);
        man.dcc8 = emcc8.distance;
        man.classCC8 = emcc8.cname;
        std::copy(emcc8.path, emcc8.path + 42, man.pcc8);
        
        ManC13.pop();
        ManCC4.pop();
        ManCC8.pop();
        
        (*resultsManhattan).push_back(man);
        
        ElementData eec13 = EucC13.top();
        ElementData eecc4 = EucCC4.top();
        ElementData eecc8 = EucCC8.top();
        
        ImageResults euc = {};
        std::copy(image->path, image->path + 42, euc.path);
        euc.actualClass= image->class_name;
        euc.dc13 = eec13.distance;
        euc.classC13 = eec13.cname;
        std::copy(eec13.path, eec13.path + 42, euc.pc13);
        euc.dcc4 = eecc4.distance;
        euc.classCC4 = eecc4.cname;
        std::copy(eecc4.path, eecc4.path + 42, euc.pcc4);
        euc.dcc8 = eecc8.distance;
        euc.classCC8 = eecc8.cname;
        std::copy(eecc8.path, eecc8.path + 42, euc.pcc8);
        EucC13.pop();
        EucCC4.pop();
        EucCC8.pop();
                            
        (*resultsEuclidian).push_back(euc);
    }
}
