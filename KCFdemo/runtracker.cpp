#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "kcftracker.hpp"

using namespace std;
using namespace cv;

int xMin;
int yMin;
int xMax;
int yMax;


void on_mouse(int event, int x, int y, int flags, void *ustc)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		xMin = x;
		yMin = y;
	}
	if (event == CV_EVENT_RBUTTONDOWN)
	{
		xMax = x;
		yMax = y;
	}

}



//int main(int argc, char* argv[]){
//
//	bool HOG = true;
//	bool FIXEDWINDOW = false;
//	bool MULTISCALE = true;
//	bool SILENT = false;
//	bool LAB = false;
//
//	// Create KCFTracker object
//	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
//
//	// Frame readed
//	Mat frame;
//
//	// Tracker results
//	Rect result;
//	
//	//Using min and max of X and Y for groundtruth rectangle
//	/*float xMin = 192;
//	float yMin = 214;
//	float width = 55;
//	float height = 110;*/
//	int width;
//	int height;
//
//
//
//	string path = "D:\\imglist\\img\\";
//	char imgName[10];
//	string frameName;
//	frame = imread("D:\\imglist\\img\\0001.jpg");
//	imshow("Image", frame);
//	setMouseCallback("Image", on_mouse, 0);
//	waitKey(5000);
//	width = xMax - xMin;
//	height = yMax - yMin;
//	for (int nFrames = 1; nFrames < 725; nFrames++)
//	{
//		if (nFrames < 10)
//		{
//			sprintf_s(imgName, "000%d.jpg", nFrames);
//		}
//		else if (nFrames < 100) { sprintf_s(imgName, "00%d.jpg", nFrames); }
//		else sprintf_s(imgName, "0%d.jpg", nFrames);
//	
//	
//		frameName = path + imgName;
//
//		// Read each frame from the list
//		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
//
//		// First frame, give the groundtruth to the tracker
//		if (nFrames == 1) {
//			tracker.init( Rect(xMin, yMin, width, height), frame );
//			rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );			
//		}
//		// Update
//		else{
//			result = tracker.update(frame);
//			rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );			
//		}
//
//		if (!SILENT){
//			imshow("Image", frame);
//			waitKey(10);
//		}
//	}
//
//}
CascadeClassifier dface;

bool Detect(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	vector<Rect>faces;
	equalizeHist(frame_gray, frame_gray);
	dface.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (!faces.empty())
	{
		xMin = faces[0].x;
		xMax = xMin + faces[0].width;
		yMin = faces[0].y;
		yMax = yMin + faces[0].height;
		return true;
	}
	else return false;
	
}
int main(int argc, char* argv[]){

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = false;
	bool LAB = false;
    if(!dface.load("haarcascade_frontalface_alt.xml")){ cout << "error load xml!"; return -1; }
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	int width;
	int height;

	VideoCapture cap(0);
	bool isfirst = true;
	
		
	if (cap.isOpened())
	{
		while (1)
		{
			cap >> frame;
			if (!frame.empty())
			{
				if (isfirst)
				{
					if (Detect(frame))
						//imshow("image", frame);
						//setMouseCallback("image", on_mouse, 0);
						//waitKey(5000);

					{
						width = xMax - xMin;
						height = yMax - yMin;
						tracker.init(Rect(xMin, yMin, width, height), frame);
						rectangle(frame, Point(xMin, yMin), Point(xMin + width, yMin + height), Scalar(0, 255, 255), 1, 8);
						isfirst = false;
					}
					
				}


				else
				{
					result = tracker.update(frame);
					rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
				}
				if (!SILENT){
					imshow("image", frame);
					waitKey(10);
				}
			}
		}
	}

	return 0;
}
