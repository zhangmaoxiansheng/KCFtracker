#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/ml/ml.hpp>
#include<fstream>
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>
#include<string>
#include<stdlib.h>
#include<stdio.h>
//略多。。。
using namespace cv;
using namespace std;

CascadeClassifier dface;
bool Detect(Mat frame, int x)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	vector<Rect>faces;
	equalizeHist(frame_gray, frame_gray);
	dface.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (!faces.empty())
	{
		char temp[50];
		Mat ROI;
		ROI = frame(faces[0]);
		sprintf_s(temp, "D://imglist//pos//%d.jpg", x);
		resize(ROI, ROI, Size(64, 64));
		imwrite(temp, ROI);
		return true;
	}
	else 
	
		
		return false;

}
//若用hogsvm中setsvm需要用到这两个参数，但是cvsvm中没有提供相应的接口，通过继承给导出来
class NEWSVM :public CvSVM                
	
{
public:
	double* get_alpha_vector()
	{
		return this->decision_func->alpha;
	}
	float get_rho()
	{
		return this->decision_func->rho;
	}
};
int main()
{
	HOGDescriptor hog(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	int hogdim = 1764;
	int posnum = 500;
	int negnum = 500;
	Mat data_train;
	Mat data_label;
	Rect rect(0, 0, 480, 480);//为了截负样本的矩形框
	if (!dface.load("haarcascade_frontalface_alt.xml")){ cout << "error load xml!"; return -1; }
	cout << "请输入正样本数量" << endl;
	cin >> posnum;
	cout << "请输入负样本数量" << endl;
	cin >> negnum;
	int imgtotle = posnum + negnum;
	//全自动训练方式，直接从视频流里截取图像生成训练文件，然后马上读取
	Mat frame;
	char img_path[50];
	char flag;
	cout << "请输入flag" << endl;
	cin >> flag;
	VideoCapture cap(0);
	//开视频真麻烦。。。
	if (cap.isOpened() && flag == 'p')
	{

		for (int imgnum = -1; imgnum < posnum; imgnum++)//-1是为了做一个缓冲，给摄像头一个开启时间
		{
			cap >> frame;
			if (!frame.empty()&&imgnum>-1)
			{
				cout << "开始采集正样本图像第" << imgnum << "张" << endl;
				sprintf_s(img_path, "D://imglist//pos//%d.jpg", imgnum);
				imshow("aa", frame);
				imwrite(img_path, frame);
				waitKey(100);
			}
		}
		cout << "正样本采集完毕" << endl;
	}
	cout << "正样本预处理中******" << endl;
	int imgnum = 0;
	for (int i = 0; i < posnum; i++)
	{
		char temp[50];
		sprintf_s(temp, "D://imglist//pos//%d.jpg", i);
		Mat temp_img = imread(temp);
		if (Detect(temp_img, imgnum)) {
			cout << "第" << i << "张处理正常" << endl;
			imgnum++;
		}
		else{
			cout << "第" << i << "张处理异常" << endl;
			remove(temp);

		}

	}
	posnum = imgnum;
	cout << "最终样本数量为" << posnum << endl;
	cout << "请输入flag" << endl;
	cin >> flag;
	if (cap.isOpened() && flag == 'n')
	{

		for (int imgnum = -1; imgnum < negnum; imgnum++)
		{
			cap >> frame;
			if (!frame.empty()&&imgnum>-1 )//-1也是为了做缓冲
			{
				cout << "开始采集负样本图像第" << imgnum << "张" << endl;
				sprintf_s(img_path, "D://imglist//neg//%d.jpg", imgnum);
				imshow("aa", frame);
				Mat ROI;
				ROI = frame(rect);
				resize(ROI, ROI, Size(64, 64));
				imwrite(img_path, ROI);
				waitKey(100);
			}
		}
		cout << "负样本采集处理完毕" << endl;
	}
	cout << "样本工作结束" << endl;
	bool isfirst = true;

	cout << "开始采集正样本hog特征" << endl;
	for (int i = 0; i < imgtotle; i++)
	{
		if (i < posnum){

			sprintf_s(img_path, "D://imglist//pos//%d.jpg", i);
			Mat img_pos = imread(img_path);
			vector<float>des;
			hog.compute(img_pos, des, Size(8, 8));
			cout << "计算正常" << i << endl;
			if (isfirst)
			{
				hogdim = des.size();
				data_train = Mat::zeros(imgtotle, hogdim, CV_32FC1);
				data_label = Mat::zeros(imgtotle, 1, CV_32FC1);
				isfirst = false;
				cout << hogdim << endl;
			}
			for (int j = 0; j < hogdim; j++)
				data_train.at<float>(i, j) = des[j];
			data_label.at<float>(i, 0) = 1;
			cout << "放置正确" << endl;
		}


		else{
			sprintf_s(img_path, "D://imglist//neg//%d.jpg", i - posnum);
			Mat img_neg = imread(img_path);
			vector<float>des2;
			hog.compute(img_neg, des2, Size(8, 8));
			cout << "计算正确" << i << endl;

			for (int j = 0; j < hogdim; j++)
				data_train.at<float>(i , j) = des2[j];
			data_label.at<float>(i , 0) = -1;
			cout << "放置正确" << endl;
		}
	}
	NEWSVM svm;
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	CvSVMParams param(CvSVM::C_SVC, CvSVM::RBF, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	cout << "开始训练******" << endl;
	svm.train(data_train, data_label, Mat(), Mat(), param);
	cout << "训练结束******" << endl;
	svm.save("SVM.xml");
	/*线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha, 有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector, 将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
		如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
		就可以利用你的训练样本训练出来的分类器进行行人检测了。*/
	hogdim = svm.get_var_count();
	int supportvectornum = svm.get_support_vector_count();
	
	Mat alpha = Mat::zeros(1, supportvectornum, CV_32FC1);
	Mat supportVectorMat = Mat::zeros(supportvectornum, hogdim, CV_32FC1);
	Mat resultMat = Mat::zeros(1, hogdim, CV_32FC1);
	for (int i = 0; i < supportvectornum; i++)//把支持向量取到Mat里
	{
		const float * psvm = svm.get_support_vector(i);//第i个支持向量，是一列，取头上的地址然后一个循环把这一列取完
		for (int j = 0; j < hogdim; j++)
		{
			supportVectorMat.at<float>(i, j) = psvm[j];
		}
	}
	double * palf = svm.get_alpha_vector();//这个就是一个向量，同样放到Mat中
	for (int i = 0; i < supportvectornum; i++)
	{
		alpha.at<float>(0, i) = palf[i];
	}
	resultMat = -1 * alpha*supportVectorMat;//转成Mat是为了方便相乘
	vector<float> mydetector;//最后还得转成vector
	for (int i = 0; i < hogdim; i++)
	{
		mydetector.push_back(resultMat.at<float>(0, i));
	}
	mydetector.push_back(svm.get_rho());
	hog.setSVMDetector(mydetector);//这里看出hogsetsvm是一个vector
	ofstream fout("mydectorf_hog_svm.txt");
	for (int i = 0; i < mydetector.size(); i++)
	{
		fout << mydetector[i] << endl;
	}
	
	/********************/
	/*测试部分*/
	cout << "开始测试" << endl;
	Mat img = imread("D://4.jpg");
	Size size = img.size();
	//resize(img, img, Size(64, 64));
	vector<Rect> target;
	hog.detectMultiScale(img, target, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	/*Rect MAX_ROI = target[0];
	for (int i = 0; i < target.size(); i++)
	{
		if ((target[i].width >= MAX_ROI.width) && target[i].height >= MAX_ROI.height)
			MAX_ROI = target[i];
		
	}*/
	for (int i = 0; i < target.size(); i++)
	{
		Rect rec = target[i];
		rectangle(img, rec, Scalar(0, 255, 0), 3);
	}
	//resize(img, img, size);
	imshow("检测结果",img);
	waitKey(0);
	return 0;
}
//读入文件及标签（半自动化）
/*ifstream img_list_pos("D:/svm_data.txt");
ifstream img_list_neg("D:/svm_data_neg.txt");
string img_path;
bool isfirst = true;
int num = 0;*/

/*while (getline(img_list_pos, img_path))
{
Mat img_pos = imread(img_path);
resize(img_pos, img_pos, Size(64, 128));
vector<float>des;
hog.compute(img_pos, des, Size(8, 8));
if (isfirst)
{
hogdim = des.size();
data_train = Mat::zeros(Size(150,hogdim), CV_32FC1);
data_label = Mat::zeros(Size(150, 1), CV_32FC1);
isfirst = false;
}
for (int i = 0; i < hogdim; i++)
data_train.at<float>(num, i) = des[i];
data_label.at<float>(num, 0) = 1;
num++;
}
while (getline(img_list_neg, img_path))
{
Mat img_neg = imread(img_path);
resize(img_neg, img_neg, Size(64, 128));
vector<float>des;
hog.compute(img_neg, des, Size(8, 8));
for (int i = 0; i < hogdim; i++)
data_train.at<float>(num, i) = des[i];
data_label.at<float>(num, 0) = -1;
num++;
}*/