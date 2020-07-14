#pragma once

#include <opencv2\opencv.hpp>

#define	GRAYSCALE		256
#define PI				3.14159265358979323846
#define	PIOVER180		0.01745329251994329576
#define STRONG_EDGE		255
#define	WEAK_EDGE		128
#define	MAX_LABEL		100000

enum DISTRICT { AREA0 = 0, AREA45, AREA90, AREA135, NOAREA };


using namespace cv;
using namespace std;

typedef struct
{
	vector<Point> pixels;
	int cx, cy;
	int minx, miny, maxx, maxy;
	int w, h;
	int area;
} LabelInfo;


struct SCircle
{
	Point _center;
	int _radius;
	int _value;
};

// binary mask with thesholding
void Histogram(Mat img, float histo[GRAYSCALE]);
Mat Threshold(Mat image, int thresh);
Mat ThresholdIterative(Mat image, int* thresh);


// morphology
Mat Erosion(Mat mask, int nSESize);
Mat Dilation(Mat mask, int nSESize);


// edge detection with canny method
void GaussianFiltering(Mat* pSrc, Mat* pDst, double sigma);
void check_weak_edge(Mat* edge, vector<Point>* pvecEdge, int x, int y);
void CannyEdgeDetection(Mat* image, Mat* edgeimg, double sigma, double th_low, double th_high);


// labeling
int Labeling(Mat imgSrc, Mat imgDst, vector<LabelInfo>& labels);


// circle detection with Hough transform
void CircleDetection(Mat* pSrc, int minradius, int maxradius, int radiusoffset, double ratio, vector<SCircle>* pvecCircle);


