// HelloOpenCV.cpp : �ܼ� ���� ���α׷��� ���� �������� �����մϴ�.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#define PI 3.14159265359
using namespace cv;
using namespace std;

Mat Gaussian(Mat image);
Mat Median(Mat image);
Mat Anisotropic(Mat image);

int main()
{
	Mat image = imread("../_res/image.jpg", IMREAD_GRAYSCALE);
	CV_Assert(!image.empty());
	Mat result(image.size(), CV_8U, Scalar(0));

	int choice;
	printf("���ڸ� �Է� (1.����þ�, 2.�̵��, 3����) : ");
	scanf("%d", &choice);

	if (choice == 1)
		result = Gaussian(image);
	else if (choice == 2)
		result = Median(image);

	else if (choice == 3)
		result = Anisotropic(image);
	else
		printf("�ٽ� �Է��ϼ��� \n");

	imshow("���� �̹���", image);
	imshow("���͸� ���� �̹���", result);
	waitKey(0);

	return 0;
}

Mat Gaussian(Mat image) {
	int filtersize = 0;
	double sigma = 3.0;

	Mat operation(image.size(), CV_8U, Scalar(0));
	Mat buff(image.size(), CV_64F, Scalar(0));

	int dim = static_cast<int>(8 * sigma + 1.0);
	if (dim < 3) dim = 1;
	if (dim % 2 == 0) dim++;
	int dim2 = dim / 2;

	double* Mask = (double*)malloc(sizeof(double) * dim);
	for (int i = 0; i < dim; i++) {
		int x = i - dim2;
		Mask[i] = exp(-(x * x) / (2 * sigma * sigma)) / (sqrt(2 * PI) * sigma);
	}

	double sum;
	for (int i = dim2; i < image.cols - dim2; i++) {
		for (int j = dim2; j < image.rows - dim2; j++) {

			sum = 0;
			for (int k = 0; k < dim; k++) {
				int y = k - dim2 + j;
				sum += Mask[k] * image.at<uchar>(y, i);
			}
			buff.at<double>(j, i) = sum;
		}
	}
	for (int j = dim2; j < image.rows - dim2; j++) {
		for (int i = dim2; i < image.cols - dim2; i++) {

			sum = 0;
			for (int k = 0; k < dim; k++) {
				int x = k - dim2 + i;
				sum += Mask[k] * buff.at<double>(j, x);
			}
			operation.at<uchar>(j, i) = saturate_cast<uchar>(sum);
		}
	}
	free(Mask);
	return operation;
}

Mat Median(Mat image) {
	//�̵�� ���͸�.
	Mat operation(image.size(), CV_8U, Scalar(0));

	for (int i = 1; i < image.rows - 1; i++) {
		for (int j = 1; j < image.cols - 1; j++) {

			uchar* Mask = (uchar*)malloc(sizeof(uchar) * 9); // �߰��� ������ �迭 ����
			uchar tmp;

			Mask[0] = image.at<uchar>(i - 1, j - 1);
			Mask[1] = image.at<uchar>(i - 1, j);
			Mask[2] = image.at<uchar>(i - 1, j + 1);
			Mask[3] = image.at<uchar>(i, j - 1);
			Mask[4] = image.at<uchar>(i, j);
			Mask[5] = image.at<uchar>(i, j + 1);
			Mask[6] = image.at<uchar>(i + 1, j - 1);
			Mask[7] = image.at<uchar>(i + 1, j);
			Mask[8] = image.at<uchar>(i + 1, j + 1);
			//sort ����
			for (int h = 0; h < 9; h++) {
				for (int w = 0; w <= 9 - (h + 1); w++) {
					if (Mask[w] > Mask[w + 1]) {
						tmp = Mask[w];
						Mask[w] = Mask[w + 1];
						Mask[w + 1] = tmp;
					}
				}
			}
			operation.at<uchar>(i, j) = saturate_cast<uchar>(Mask[4]);
		}	//���� ����� �迭�� �߰� �� ����.
	}
	return operation;
}

Mat Anisotropic(Mat image) {
	//���漺 ���͸�
	Mat operation(image.size(), CV_8U, Scalar(0));
	Mat dblimage(image.size(), CV_64F, Scalar(0));
	Mat buff(image.size(), CV_64F, Scalar(0));

	image.convertTo(dblimage, CV_64F);

	double lambda = 0.25;
	double K = 4;
	double K2 = K * K;
	int itr = 10;

	double gradn, grads, grade, gradw;
	double gcn, gcs, gce, gcw;

	for (int i = 0; i < itr; i++) {
		for (int x = 1; x < image.rows - 1; x++) {
			for (int y = 1; y < image.cols - 1; y++) {

				gradn = dblimage.at<double>(x - 1, y) - dblimage.at<double>(x, y);
				grads = dblimage.at<double>(x + 1, y) - dblimage.at<double>(x, y);
				grade = dblimage.at<double>(x, y - 1) - dblimage.at<double>(x, y);
				gradw = dblimage.at<double>(x, y + 1) - dblimage.at<double>(x, y);

				gcn = gradn / (1.0f + gradn * gradn / K2);
				gcs = grads / (1.0f + grads * grads / K2);
				gce = grade / (1.0f + grade * grade / K2);
				gcw = gradw / (1.0f + gradw * gradw / K2);

				buff.at<double>(x, y) = dblimage.at<double>(x, y) + lambda * (gcn + gcs + gce + gcw);
			}
		}
		buff.copyTo(dblimage);
	}
	dblimage.convertTo(operation, CV_8U);

	return operation;
}
