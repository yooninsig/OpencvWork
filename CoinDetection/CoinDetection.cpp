// HelloOpenCV.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include "func.h"

using namespace cv;
using namespace std;
vector<SCircle> findCircle;

int main()
{
	Mat image = imread("../_res/coin.jpg", IMREAD_GRAYSCALE);
	CV_Assert(!image.empty());

	Mat EdgeImage(image.size(), CV_8U, Scalar(0));
	CannyEdgeDetection(&image, &EdgeImage, 4.5, 30.0, 60.0);
	int c10, c100, c500;
	CircleDetection(&EdgeImage, 35, 36, 1, 0.6, &findCircle);
	c10 = findCircle.size();
	CircleDetection(&EdgeImage, 42, 43, 1, 0.6, &findCircle);
	c100 = findCircle.size() - c10;
	CircleDetection(&EdgeImage, 51, 52, 1, 0.6, &findCircle);
	c500 = findCircle.size() - c10 - c100;

	printf("총액 500짜리 %d개 100개짜리 %d개 10원 짜리 %d개 = %d원", c500, c100, c10, (c500*500) + (c100*100) + (c10*10));

	imshow("원본", image);
	imshow("엣지 찾기", EdgeImage);
	waitKey(0);

    return 0;
}

