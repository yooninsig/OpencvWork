// HelloOpenCV.cpp : �ܼ� ���� ���α׷��� ���� �������� �����մϴ�.
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

	printf("�Ѿ� 500¥�� %d�� 100��¥�� %d�� 10�� ¥�� %d�� = %d��", c500, c100, c10, (c500*500) + (c100*100) + (c10*10));

	imshow("����", image);
	imshow("���� ã��", EdgeImage);
	waitKey(0);

    return 0;
}

