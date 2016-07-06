#include <windows.h>
#include <mmsystem.h>
#include <stdio.h>
#include <stdlib.h>
#include "wininet.h"
#include <direct.h>
#include <string.h>
#include <list>
#include <vector>
#include <io.h>
//#include<highgui.h>
//#include<cvaux.h>
#pragma comment(lib,"Wininet.lib")
#pragma comment(lib,"opencv_calib3d248.lib")
#pragma comment(lib,"opencv_contrib248.lib")
#pragma comment(lib,"opencv_core248.lib")
#pragma comment(lib,"opencv_flann248.lib")
#pragma comment(lib,"opencv_gpu248.lib")

#include "opencv2/objdetect/objdetect.hpp"//�������һЩԤ���������ļ�� (���������۾������ӡ��ˡ�������)��
#include "opencv2/highgui/highgui.hpp"//һ�������õĽӿڣ��ṩ��Ƶ��׽��ͼ�����Ƶ����ȹ��ܣ����м򵥵� UI �ӿڡ�
#include "opencv2/imgproc/imgproc.hpp"//ͼ����ģ�飬�������Ժͷ�����ͼ���˲�������ͼ��ת�� (���š�������͸�ӱ任��һ���Ի��ڱ����ӳ��)����ɫ�ռ�ת����ֱ��ͼ�ȵȡ�
#include "opencv2/ml/ml.hpp"//���ֻ���ѧϰ�㷨���� K ��ֵ��֧���������������硣

#include <fstream> 
#include <iostream>

using namespace std;
using namespace cv;

String cascadeName = "./cascade.xml";


int getFiles( string path, vector<string>& files )  
{  
    //�ļ���� 
    long   hFile   =   0;  
    //�ļ���Ϣ  
    struct _finddata_t fileinfo;  
    string p;  
    if((hFile = _findfirst(p.assign(path).append("/*").c_str(),&fileinfo)) !=  -1)  
    {  
        do  
        {  
            //�����Ŀ¼,����֮  
            //�������,�����б�  
            if((fileinfo.attrib &  _A_SUBDIR))  
            {  
                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
                    getFiles( p.assign(path).append("/").append(fileinfo.name), files );  
            }  
            else  
            {  
                files.push_back(p.assign(path).append("/").append(fileinfo.name) );  
            }  
        }while(_findnext(hFile, &fileinfo)  == 0);  
        _findclose(hFile);  
    }  
	if(files.size()==0)
		return 0;
	else
		return 1;
}  


//int FindImgs(char * pSrcImgPath, char * pRstImgPath, vector<string> img_vec);

int main(string args[] )
{
	CascadeClassifier cascade;//������������������
	vector<string> ImgVec; 
	vector<string>::iterator pImgVecTemp; 
	int count = 0;

	double scale = 1.;
	Mat image;
	double t;
	double times;
	if( !cascade.load( cascadeName ) )//��ָ�����ļ�Ŀ¼�м��ؼ���������
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return 0;
	}
	char* src_path = "../mdpm/";
	int nFlag = getFiles(src_path, ImgVec);	
	if(nFlag != 0)   
	{
		cout<<"Read Image  error !  Input 0 to exit \n";
		exit(0);
	}

	pImgVecTemp = ImgVec.begin();
	String fileName ="./mush_reg.txt";
	ofstream outFile(fileName.c_str(),ios_base::app);//�½��򸲸Ƿ�ʽд�����ݣ�
		//ofstream outFile(fileName.c_str(),ios_base::out);
		if(!outFile.is_open()){
			cout<<"���ļ�ʧ�ܡ���"<<endl;
		}
	for(unsigned int iik = 1; iik<=ImgVec.size(); iik++,pImgVecTemp++)//iik<=ImgList.size()
	{
		
		image = imread(pImgVecTemp->c_str());

		//cout<<pImgListTemp->SrcImgPath<<endl;
		if( !image.empty() )//��ȡͼƬ���ݲ���Ϊ��
		{
			//imshow("image",image);
			Mat gray, smallImg( cvRound (image.rows/scale), cvRound(image.cols/scale), CV_8UC1 );//��ͼƬ��С���ӿ����ٶ�
			cvtColor( image, gray, CV_BGR2GRAY );//��Ϊ�õ�����haar���������Զ��ǻ��ڻҶ�ͼ��ģ�����Ҫת���ɻҶ�ͼ��
			resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );//���ߴ���С��1/scale,�����Բ�ֵ
			equalizeHist( smallImg, smallImg );//ֱ��ͼ����

			
			vector<Rect> rects,rects1;
			vector<Rect>::const_iterator pRect;  
			rects.clear();
			printf( "%03d  ",iik);
			t = (double)cvGetTickCount();//���������㷨ִ��ʱ��
			//cascade.detectMultiScale(image,rects,1.1,3,0|CV_HAAR_SCALE_IMAGE,Size(200,200),Size(400,400));//Size(0,0),Size(30,30)  
			cascade.detectMultiScale(image,rects,1.1,15,0,Size(50,50),Size(480,480));//Ĭ�����ߴ�ΪImage��size��
			//flags:CV_HAAR_DO_CANNY_PRUNING(CANNY��Ե���)��CV_HAAR_SCALE_IMAGE(����ͼ����)��CV_HAAR_FIND_BIGGEST_OBJECT(Ѱ������Ŀ��)��CV_HAAR_DO_ROUGH_SEARCH(����������)
			t = (double)cvGetTickCount() - t;
			times += t/((double)cvGetTickFrequency()*1000.);
			printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

			size_t i, j;  
			for( i = 0; i < rects.size(); i++ )//ȥ���ռ��о������������ϵ�����򣬱������  
			{  
				Rect r = rects[i];  
				for( j = 0; j < rects.size(); j++ )  
					if( j != i && (r & rects[j]) == r)  
						break;  
				if( j == rects.size() )  
					rects1.push_back(r);  
			}  

			 for( i = 0; i < rects1.size(); i++ )  
        {  
            Rect r = rects1[i];  
			count++;
			
            rectangle(image, r.tl(), r.br(), cv::Scalar(0,255,0), 3);// tl:the top-left corner,br: the bottom-right corner 
						int data1 = pRect->x;
						int data2 = pRect->y;
						int data3 = pRect->x+pRect->width;
						int data4 = pRect->y+pRect->height;
						outFile<<data1<<" "<<data2<<" "<<data3<<" "<<data4<<endl;
        }  
		}
	}
	printf( "target_number = %d\n", count );
	printf( "total time = %g ms\n\n", times );
	return 0;
}
