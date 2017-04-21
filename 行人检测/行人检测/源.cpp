#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <string.h>
#include <ctype.h>

using namespace cv;
using namespace std;
//�������⣬opencvԴ�������û�����ͼ����б����������л���
int main()
{
	Mat frame;
	VideoCapture cap(0);
	
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	//������Ǹ�set��ʹ����opencv���Ѿ�ѵ���õĲ������������������05��һƪhogsvm��ѵ���õ��Ľ��
	vector<Rect>ROI;
	if (cap.isOpened())
	{
		while (1)
		{
			cap >> frame;
			if (!frame.empty())
			{
				hog.detectMultiScale(frame, ROI, 0, Size(8, 8), Size(32, 32));
				for (int i = 0; i < ROI.size(); i++)
				{
					rectangle(frame, ROI[i], Scalar(225, 0, 0), 2);
				}
				imshow("���˼��", frame);
				waitKey(5);
			}
		}
	}
	return 0;
}