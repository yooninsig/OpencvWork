#include "stdafx.h"
#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;

Mat src;  //영상을 잡음 제거 필터링을 정의할 Mat객체
Mat buff; //영상을 읽어올 Mat객체 설정
Mat dst;  //최종 영상을 출력할 Mat객체

Point2f srcQuad[4], dstQuad[4]; 

void on_mouse(int event, int x, int y, int flags, void*) {
	static int cnt = 0;

	if (event == EVENT_LBUTTONDOWN) {
		if (cnt < 4) {
			srcQuad[cnt++] = Point2f(x, y);

			circle(buff, Point(x, y), 5, Scalar(0, 0, 255), -1);
			imshow("buff", buff);
		}
	}
}
void on_result(int event, int x, int y, int flags, void*) {
	static int result = 0;

	if (event == EVENT_LBUTTONDOWN) {
		if (result < 4) {
			dstQuad[result++] = Point2f(x, y);

			circle(dst, Point(x, y), 5, Scalar(0, 0, 255), -1);
			imshow("dst", dst);

			if (result == 4) {

				Mat pers = getPerspectiveTransform(srcQuad, dstQuad);

				warpPerspective(src, dst, pers, src.size());
				imshow("dst", dst);
			}
		}
	}
}
int main(void) {

	buff = imread("../_res/card_noise.jpg", IMREAD_GRAYSCALE);
	if (buff.empty()) {
		cerr << "IMAGE LOAD FAILED!" << endl;
		return -1;
	}

	dst = Mat::zeros(buff.size(), CV_8UC3);

	namedWindow("buff");
	namedWindow("dst");

	setMouseCallback("buff", on_mouse);
	bilateralFilter(buff, src, -1, 35, 20);
	setMouseCallback("dst", on_result);

	imshow("buff", buff);
	imshow("dst", dst);
	waitKey(0);

	destroyAllWindows();
	return 0;
}