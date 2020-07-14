#include "stdafx.h"
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;


Mat fill_frame(Mat src, Mat background) {
	//배경 검출을 위해 영상의 컬러 도메인 변환
	Mat img_hsv;
	cvtColor(src, img_hsv, COLOR_BGR2HSV);
	//비디오 저장을 위한 작업

	//배경 사진에 대해서 원본 만큼 사이즈 잡아주는 함수
	resize(background, background, src.size());

	Mat blue = img_hsv.clone();
	inRange(img_hsv, Scalar(100, 30, 30), Scalar(150, 255, 255), blue);

	Mat dst, dst1, inverted;
	bitwise_not(blue, inverted);
	bitwise_and(src, src, dst, inverted);
	bitwise_or(dst, background, dst1, blue);
	bitwise_or(dst, dst1, dst1);

	imshow("src", src);
	imshow("dst1", dst1);
	return dst1;
}
int main()
{
	VideoCapture cap("../_res/ex2.mp4");
	Mat background = imread("../_res/back.jpg");

	if (!cap.isOpened()) {
		cerr << "Video open failed!" << endl;
		return 1;
	}

	double fps = cap.get(CAP_PROP_FPS);
	int delay = cvRound(100 / fps);
	int width = cap.get(CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CAP_PROP_FRAME_HEIGHT);
	Mat frame;

	Mat result;
	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
	VideoWriter video("../_res/output.mp4", fourcc, fps, Size(width, height));

	while (true) {
		cap >> frame;
		if (frame.empty())
			break;
		result = fill_frame(frame, background);
		video << result;
		if (waitKey(delay) == 27)
			break;
	}
	waitKey();
	return 0;
}
