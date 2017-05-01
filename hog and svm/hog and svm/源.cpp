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
//�Զࡣ����
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
//����hogsvm��setsvm��Ҫ�õ�����������������cvsvm��û���ṩ��Ӧ�Ľӿڣ�ͨ���̳и�������
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
	Rect rect(0, 0, 480, 480);//Ϊ�˽ظ������ľ��ο�
	if (!dface.load("haarcascade_frontalface_alt.xml")){ cout << "error load xml!"; return -1; }
	cout << "����������������" << endl;
	cin >> posnum;
	cout << "�����븺��������" << endl;
	cin >> negnum;
	int imgtotle = posnum + negnum;
	//ȫ�Զ�ѵ����ʽ��ֱ�Ӵ���Ƶ�����ȡͼ������ѵ���ļ���Ȼ�����϶�ȡ
	Mat frame;
	char img_path[50];
	char flag;
	cout << "������flag" << endl;
	cin >> flag;
	VideoCapture cap(0);
	//����Ƶ���鷳������
	if (cap.isOpened() && flag == 'p')
	{

		for (int imgnum = -1; imgnum < posnum; imgnum++)//-1��Ϊ����һ�����壬������ͷһ������ʱ��
		{
			cap >> frame;
			if (!frame.empty()&&imgnum>-1)
			{
				cout << "��ʼ�ɼ�������ͼ���" << imgnum << "��" << endl;
				sprintf_s(img_path, "D://imglist//pos//%d.jpg", imgnum);
				imshow("aa", frame);
				imwrite(img_path, frame);
				waitKey(100);
			}
		}
		cout << "�������ɼ����" << endl;
	}
	cout << "������Ԥ������******" << endl;
	int imgnum = 0;
	for (int i = 0; i < posnum; i++)
	{
		char temp[50];
		sprintf_s(temp, "D://imglist//pos//%d.jpg", i);
		Mat temp_img = imread(temp);
		if (Detect(temp_img, imgnum)) {
			cout << "��" << i << "�Ŵ�������" << endl;
			imgnum++;
		}
		else{
			cout << "��" << i << "�Ŵ����쳣" << endl;
			remove(temp);

		}

	}
	posnum = imgnum;
	cout << "������������Ϊ" << posnum << endl;
	cout << "������flag" << endl;
	cin >> flag;
	if (cap.isOpened() && flag == 'n')
	{

		for (int imgnum = -1; imgnum < negnum; imgnum++)
		{
			cap >> frame;
			if (!frame.empty()&&imgnum>-1 )//-1Ҳ��Ϊ��������
			{
				cout << "��ʼ�ɼ�������ͼ���" << imgnum << "��" << endl;
				sprintf_s(img_path, "D://imglist//neg//%d.jpg", imgnum);
				imshow("aa", frame);
				Mat ROI;
				ROI = frame(rect);
				resize(ROI, ROI, Size(64, 64));
				imwrite(img_path, ROI);
				waitKey(100);
			}
		}
		cout << "�������ɼ��������" << endl;
	}
	cout << "������������" << endl;
	bool isfirst = true;

	cout << "��ʼ�ɼ�������hog����" << endl;
	for (int i = 0; i < imgtotle; i++)
	{
		if (i < posnum){

			sprintf_s(img_path, "D://imglist//pos//%d.jpg", i);
			Mat img_pos = imread(img_path);
			vector<float>des;
			hog.compute(img_pos, des, Size(8, 8));
			cout << "��������" << i << endl;
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
			cout << "������ȷ" << endl;
		}


		else{
			sprintf_s(img_path, "D://imglist//neg//%d.jpg", i - posnum);
			Mat img_neg = imread(img_path);
			vector<float>des2;
			hog.compute(img_neg, des2, Size(8, 8));
			cout << "������ȷ" << i << endl;

			for (int j = 0; j < hogdim; j++)
				data_train.at<float>(i , j) = des2[j];
			data_label.at<float>(i , 0) = -1;
			cout << "������ȷ" << endl;
		}
	}
	NEWSVM svm;
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	CvSVMParams param(CvSVM::C_SVC, CvSVM::RBF, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	cout << "��ʼѵ��******" << endl;
	svm.train(data_train, data_label, Mat(), Mat(), param);
	cout << "ѵ������******" << endl;
	svm.save("SVM.xml");
	/*����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha, ��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector, ���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
		��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
		�Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�*/
	hogdim = svm.get_var_count();
	int supportvectornum = svm.get_support_vector_count();
	
	Mat alpha = Mat::zeros(1, supportvectornum, CV_32FC1);
	Mat supportVectorMat = Mat::zeros(supportvectornum, hogdim, CV_32FC1);
	Mat resultMat = Mat::zeros(1, hogdim, CV_32FC1);
	for (int i = 0; i < supportvectornum; i++)//��֧������ȡ��Mat��
	{
		const float * psvm = svm.get_support_vector(i);//��i��֧����������һ�У�ȡͷ�ϵĵ�ַȻ��һ��ѭ������һ��ȡ��
		for (int j = 0; j < hogdim; j++)
		{
			supportVectorMat.at<float>(i, j) = psvm[j];
		}
	}
	double * palf = svm.get_alpha_vector();//�������һ��������ͬ���ŵ�Mat��
	for (int i = 0; i < supportvectornum; i++)
	{
		alpha.at<float>(0, i) = palf[i];
	}
	resultMat = -1 * alpha*supportVectorMat;//ת��Mat��Ϊ�˷������
	vector<float> mydetector;//��󻹵�ת��vector
	for (int i = 0; i < hogdim; i++)
	{
		mydetector.push_back(resultMat.at<float>(0, i));
	}
	mydetector.push_back(svm.get_rho());
	hog.setSVMDetector(mydetector);//���￴��hogsetsvm��һ��vector
	ofstream fout("mydectorf_hog_svm.txt");
	for (int i = 0; i < mydetector.size(); i++)
	{
		fout << mydetector[i] << endl;
	}
	
	/********************/
	/*���Բ���*/
	cout << "��ʼ����" << endl;
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
	imshow("�����",img);
	waitKey(0);
	return 0;
}
//�����ļ�����ǩ�����Զ�����
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