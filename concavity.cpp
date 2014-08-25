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

/**
  *Transforms the input image into a binary image.
  */
cv::Mat toBinary(cv::Mat& imageMat){
    //Grayscale matrix
    cv::Mat grayscaleMat(imageMat.size(),CV_8U);
    
    //Convert BGR to Gray
    cv::cvtColor(imageMat,grayscaleMat,CV_BGR2GRAY);
    
    //Binary image
    cv::Mat binaryMat(grayscaleMat.size(),grayscaleMat.type());
    
    //Apply thresholding
    cv::threshold(grayscaleMat,binaryMat,100,255,cv::THRESH_BINARY);
    return binaryMat;
}

float Q_rsqrt( float number ){
	long i;
	float x2, y;
	const float threehalfs = 1.5F;
 
	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//      y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
 
	return y;
}

float sum_square(float* array,int n){
    float res=0;
    for(int i=0;i<n;i++){
        res+=(array[i]*array[i]);
    }
    return res;
}

void normalize(float* array,int n){
    float inv_sqrt = Q_rsqrt(sum_square(array,n));
    for(int i=0;i<n;i++){
        array[i]*=inv_sqrt;
    }
}

/**
  * Calculates the histogram, from a given image onto array pointer.
  * Uses 13 bins method.
  */
void concavity13C(cv::Mat& image,float array[]){
    /* Toma la imagen BINARIA, e itera por cada pixel del foreground.
     * Cuenta los topes arriba,abajo,izquierda,derecha.
     *    -un tope: descarta
     *    -cuatro topes.
     *       del pixel mas alto y el mas bajo:
     *           itero por los pixeles a izquierda y derecha de cada uno.
     *              los clasifico por cuantos auxiliares no topan, si topa por todos, hay una concavidad y se cuenta a parte.
     */
     cv::Mat topes(image.size(),CV_8UC4);
     int prev;
     //255: topa por arriba, 0:escapa por arriba
     for(int j=0;j<image.cols;j++){
         prev=0;
         for(int i=0;i<image.rows;i++){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[0]=0;
            else if(pix==0) {prev=1;topes.at<cv::Vec4b>(i,j)[0]=0;}
            else if(pix==255 && prev!=0)topes.at<cv::Vec4b>(i,j)[0]=prev++;
        }
     }
     //255: topa por la derecha, 0:escapa por la derecha
     for(int i=0;i<image.rows;i++){
        prev=0;
        for(int j=image.cols-1;0<=j;j--){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[1]=0;
            else if(pix==0) {prev=1;topes.at<cv::Vec4b>(i,j)[1]=0;}
            else if(pix==255 && prev!=0)topes.at<cv::Vec4b>(i,j)[1]=prev++;
        }
     }
     //255: topa por abajo, 0:escapa por abajo
     for(int j=0;j<image.cols;j++){
        prev=0;
        for(int i=image.rows-1;0<=i;i--){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[2]=0;
            else if(pix==0) {prev=1;topes.at<cv::Vec4b>(i,j)[2]=0;}
            else if(pix==255 && prev!=0)topes.at<cv::Vec4b>(i,j)[2]=prev++;
        }
     }
     //255:topa por la izquierda, 0:escapa por la izquierda
     for(int i=0;i<image.rows;i++){
        prev=0;
        for(int j=0;j<image.cols;j++){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[3]=0;
            else if(pix==0) {prev=1;topes.at<cv::Vec4b>(i,j)[3]=0;}
            else if(pix==255 && prev!=0)topes.at<cv::Vec4b>(i,j)[3]=prev++;
        }
     }
     for(int i=0;i<13;i++)
        array[i]=0;
     
     for(int i=0;i<topes.rows;i++){
        for(int j=0;j<topes.cols;j++){
            if(image.at<uchar>(i,j)==0) continue;
            cv::Vec4b pix = topes.at<cv::Vec4b>(i,j);
            char MASK=15;
            if(pix[0]!=0) MASK &= 14;//arriba
            if(pix[1]!=0) MASK &= 13;//derecha
            if(pix[2]!=0) MASK &= 11;//abajo
            if(pix[3]!=0) MASK &=  7;//izquierda
            switch(MASK){
                case 3:
                    //escapa 0 1
                    array[0]++;
                    break;
                case 6:
                    //escapa 1 2
                    array[1]++;
                    break;
                case 12:
                    //escapa 2 3
                    array[2]++;
                    break;
                case 9:
                    //escapa 3 0
                    array[3]++;
                    break;
                case 1:
                    //escapa 0
                    array[4]++;
                    break;
                case 2:
                    //escapa 1
                    array[5]++;
                    break;
                case 4:
                    //escapa 2
                    array[6]++;
                    break;
                case 8:
                    //escapa 3
                    array[7]++;
                    break;
                case 0://no escapa
                    {
                    cv::Vec4b top = topes.at<cv::Vec4b>(i,j-pix[0]+1);//arriba
                    cv::Vec4b bottom = topes.at<cv::Vec4b>(i,j+pix[2]-1);//abajo
                    int MASK2=15;
                    if(top[1]!=0)    MASK2 &= 14;//S1
                    if(top[3]!=0)    MASK2 &= 13;//S2
                    if(bottom[1]!=0) MASK2 &= 11;//S3
                    if(bottom[3]!=0) MASK2 &=  7;//S4
                    switch(MASK2){
                        case 14://s1
                            array[9]++;
                            break;
                        case 13://s2
                            array[10]++;
                            break;
                        case 11://s3
                            array[11]++;
                            break;
                        case 7://s4
                            array[12]++;
                            break;
                        case 0:
                            array[8]++;
                            break;
                        default:
                            //descartar
                            break;
                    }
                    }
                    break;
                default:
                    //descartar
                    break;
            }
        }
     }
     normalize(array,13);
}

/**
  * Calculates the histogram, from a given image onto array pointer.
  * Uses 4-Connected method.
  */
void concavity4CC(cv::Mat image,float *array){
     cv::Mat topes(image.size(),CV_8UC4);
     int prev;
     //255: topa por arriba, 0:escapa por arriba
     for(int j=0;j<image.cols;j++){
         prev=0;
         for(int i=0;i<image.rows;i++){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[0]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[0]=255;}
        }
     }
     //255: topa por la derecha, 0:escapa por la derecha
     for(int i=0;i<image.rows;i++){
        prev=0;
        for(int j=image.cols-1;0<=j;j--){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[1]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[1]=255;}
        }
     }
     //255: topa por abajo, 0:escapa por abajo
     for(int j=0;j<image.cols;j++){
        prev=0;
        for(int i=image.rows-1;0<=i;i--){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[2]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[2]=255;}
        }
     }
     //255:topa por la izquierda, 0:escapa por la izquierda
     for(int i=0;i<image.rows;i++){
        prev=0;
        for(int j=0;j<image.cols;j++){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[3]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[3]=255;}
        }
     }
     for(int i=0;i<16;i++)
        array[i]=0;
     
     for(int i=0;i<topes.rows;i++){
        for(int j=0;j<topes.cols;j++){
            if(image.at<uchar>(i,j)==0) continue;
            cv::Vec4b pix = topes.at<cv::Vec4b>(i,j);
            char MASK=15;
            if(pix[0]!=0) MASK &= 14;//arriba
            if(pix[1]!=0) MASK &= 13;//derecha
            if(pix[2]!=0) MASK &= 11;//abajo
            if(pix[3]!=0) MASK &=  7;//izquierda
            array[MASK]++;
        }
    }
    normalize(array,16);
}

/**
  * Calculates the histogram, from a given image onto array pointer.
  * Uses 8-Connected method.
  */
void concavity8CC(cv::Mat image,float* array){
    cv::Mat topes(image.size(),CV_8UC4);
    int prev;
    int rows=image.rows;
    int cols=image.cols;
    //direecion: hacia derecha
    for(int k=0;k<rows;k++){
        int i,j;
        //derecha, abajo
        prev=0;
        for(i=k,j=0;i<rows && j<cols;i++,j++){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[2]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[0]=255;}
        }
        //derecha, arriba
        prev=0;
        for(i=k,j=cols-1;i<rows && j>=0;i++,j--){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[2]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[1]=255;}
        }
        //izquierda, abajo
        prev=0;
        for(i=k,j=0;i>=0 && j<cols;i--,j++){
        uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[2]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[2]=255;}
        }
        //izquierda, arriba
        prev=0;
        for(i=k,j=cols-1;i>=0 && j>=0;i--,j--){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[2]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[3]=255;}
        }
    }
    //direecion: hacia abajo
    for(int k=0;k<cols;k++){
        int i,j;
        //derecha, abajo
        prev=0;
        for(i=0,j=k;i<rows && j<cols;i++,j++){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[0]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[0]=255;}
        }
        //derecha, arriba
        prev=0;
        for(i=0,j=k;i<rows && j>=0;i++,j--){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[1]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[1]=255;}
        }
        //izquierda, abajo
        prev=0;
        for(i=rows-1,j=k;i>=0 && j<cols;i--,j++){
        uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[2]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[2]=255;}
        }
        //izquierda, arriba
        prev=0;
        for(i=rows-1,j=k;i>=0 && j>=0;i--,j--){
            uchar pix = image.at<uchar>(i,j);
            if(pix==255 && prev==0) topes.at<cv::Vec4b>(i,j)[3]=0;
            else if(pix==0 || prev==1) {prev=1;topes.at<cv::Vec4b>(i,j)[3]=255;}
        }
    }
    for(int i=0;i<16;i++)
       array[i]=0;
    
    for(int i=0;i<topes.rows;i++){
       for(int j=0;j<topes.cols;j++){
           if(image.at<uchar>(i,j)==0) continue;
           cv::Vec4b pix = topes.at<cv::Vec4b>(i,j);
           char MASK=15;
           if(pix[0]!=0) MASK &= 14;//arriba
           if(pix[1]!=0) MASK &= 13;//derecha
           if(pix[2]!=0) MASK &= 11;//abajo
           if(pix[3]!=0) MASK &=  7;//izquierda
           array[MASK]++;
       }
    }
    normalize(array,16);
}
