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

#include "opencv2/objdetect/objdetect.hpp"//物体检测和一些预定义的物体的检测 (如人脸、眼睛、杯子、人、汽车等)。
#include "opencv2/highgui/highgui.hpp"//一个简单易用的接口，提供视频捕捉、图像和视频编码等功能，还有简单的 UI 接口。
#include "opencv2/imgproc/imgproc.hpp"//图像处理模块，包括线性和非线性图像滤波、几何图像转换 (缩放、仿射与透视变换、一般性基于表的重映射)、颜色空间转换、直方图等等。
#include "opencv2/ml/ml.hpp"//多种机器学习算法，如 K 均值、支持向量机和神经网络。

#include <fstream> 
#include <iostream>

using namespace std;
using namespace cv;

String cascadeName = "./cascade.xml";


int getFiles( string path, vector<string>& files )  
{  
    //文件句柄 
    long   hFile   =   0;  
    //文件信息  
    struct _finddata_t fileinfo;  
    string p;  
    if((hFile = _findfirst(p.assign(path).append("/*").c_str(),&fileinfo)) !=  -1)  
    {  
        do  
        {  
            //如果是目录,迭代之  
            //如果不是,加入列表  
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
	CascadeClassifier cascade;//创建级联分类器对象
	vector<string> ImgVec; 
	vector<string>::iterator pImgVecTemp; 
	int count = 0;

	double scale = 1.;
	Mat image;
	double t;
	double times;
	if( !cascade.load( cascadeName ) )//从指定的文件目录中加载级联分类器
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
	ofstream outFile(fileName.c_str(),ios_base::app);//新建或覆盖方式写入数据；
		//ofstream outFile(fileName.c_str(),ios_base::out);
		if(!outFile.is_open()){
			cout<<"打开文件失败。。"<<endl;
		}
	for(unsigned int iik = 1; iik<=ImgVec.size(); iik++,pImgVecTemp++)//iik<=ImgList.size()
	{
		
		image = imread(pImgVecTemp->c_str());

		//cout<<pImgListTemp->SrcImgPath<<endl;
		if( !image.empty() )//读取图片数据不能为空
		{
			//imshow("image",image);
			Mat gray, smallImg( cvRound (image.rows/scale), cvRound(image.cols/scale), CV_8UC1 );//将图片缩小，加快检测速度
			cvtColor( image, gray, CV_BGR2GRAY );//因为用的是类haar特征，所以都是基于灰度图像的，这里要转换成灰度图像
			resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );//将尺寸缩小到1/scale,用线性插值
			equalizeHist( smallImg, smallImg );//直方图均衡

			
			vector<Rect> rects,rects1;
			vector<Rect>::const_iterator pRect;  
			rects.clear();
			printf( "%03d  ",iik);
			t = (double)cvGetTickCount();//用来计算算法执行时间
			//cascade.detectMultiScale(image,rects,1.1,3,0|CV_HAAR_SCALE_IMAGE,Size(200,200),Size(400,400));//Size(0,0),Size(30,30)  
			cascade.detectMultiScale(image,rects,1.1,15,0,Size(50,50),Size(480,480));//默认最大尺寸为Image的size。
			//flags:CV_HAAR_DO_CANNY_PRUNING(CANNY边缘检测)、CV_HAAR_SCALE_IMAGE(缩放图像检测)、CV_HAAR_FIND_BIGGEST_OBJECT(寻找最大的目标)、CV_HAAR_DO_ROUGH_SEARCH(做粗略搜索)
			t = (double)cvGetTickCount() - t;
			times += t/((double)cvGetTickFrequency()*1000.);
			printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

			size_t i, j;  
			for( i = 0; i < rects.size(); i++ )//去掉空间中具有内外包含关系的区域，保留大的  
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
