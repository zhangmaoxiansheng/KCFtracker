KCF is a new algorithm to track the object.
KCF can tracker the people in real time,which means that it can be applied on some robots.
This demo can detect your face and track it.Later,the demo may detect the people who is walking and track him.
附带程序hog and svm
实现了从视频流里自动截取人脸作为训练材料，正样本采集简便，然后进行svm和hog训练，正样本的来源也可以是KCF跟踪框，可以在跟踪的过程中优化分类器。（不过opencv自带的svm很垃圾，需要借助其他平台的好分类器）
