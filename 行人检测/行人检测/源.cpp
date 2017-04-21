#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <string.h>
#include <ctype.h>

using namespace cv;
using namespace std;
//很有问题，opencv源代码利用滑窗将图像进行遍历导致运行缓慢
int main()
{
	Mat frame;
	VideoCapture cap(0);
	
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	//上面的那个set，使用了opencv中已经训练好的参数，这个参数可能是05年一篇hogsvm中训练得到的结果
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
				imshow("行人检测", frame);
				waitKey(5);
			}
		}
	}
	return 0;
}