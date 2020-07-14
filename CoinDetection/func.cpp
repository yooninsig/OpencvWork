#include "func.h"


Mat Threshold(Mat image, int thresh)
{
	Mat binmask(image.size(), CV_8U, Scalar(0));

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			binmask.at<uchar>(y, x) = (image.at<uchar>(y, x) < thresh) ? 0 : 255;
		}
	}

	return binmask;
}


void Histogram(Mat img, float histo[GRAYSCALE])
{
	memset(histo, 0, sizeof(float) * GRAYSCALE);

	// ������׷����
	int cnt[GRAYSCALE];
	memset(cnt, 0, sizeof(int) * GRAYSCALE);

	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			cnt[img.at<uchar>(y, x)]++;
		}
	}

	// ������׷�����ȭ(histogram normalization)
	for (int i = 0; i< 256; i++)
	{
		histo[i] = static_cast<float>(cnt[i]) / (img.rows*img.cols);
	}
}


Mat ThresholdIterative(Mat image, int* thresh)
{
	Mat binmask;

	float hist[GRAYSCALE] = { 0, };
	Histogram(image, hist); // ����ȭ�� ������׷�. hist �迭�� ������ [0, 1].

							// �ʱ� �Ӱ谪 ���� - �׷��̽����� ���� ��ü ���
	int i, T, Told;
	float sum = 0.f;
	for (i = 0; i < GRAYSCALE; i++)
	{
		sum += (i * hist[i]);
	}

	T = static_cast<int>(sum + .5f);

	// �ݺ��� ���� �Ӱ谪 ����

	float a1, b1, u1, a2, b2, u2;
	do {
		Told = T;

		a1 = b1 = u1 = 0.f;
		for (i = 0; i <= Told; i++)
		{
			a1 += (i*hist[i]);
			b1 += hist[i];
		}

		if (b1 != 0.f)
			u1 = a1 / b1;

		a2 = b2 = u2 = 0.f;
		for (i = Told + 1; i < 256; i++)
		{
			a2 += (i*hist[i]);
			b2 += hist[i];
		}

		if (b2 != 0.f)
			u2 = a2 / b2;

		T = static_cast<int>((u1 + u2) / 2 + 0.5f);
	} while (T != Told);
	*thresh = T;

	binmask = Threshold(image, T);

	return binmask;
}


Mat Erosion(Mat mask, int nSESize)
{
	int w = mask.cols;
	int h = mask.rows;
	int val = 255;
	Mat morph(mask.size(), CV_8U, Scalar(0));
	mask.copyTo(morph);
	int nHalfSESize = nSESize >> 1;
	int nFullSESize = nSESize * nSESize;
	int nFullSEValue = val * nFullSESize;

	for (int y = nHalfSESize; y < h - nHalfSESize; y++)
	{
		for (int x = nHalfSESize; x < w - nHalfSESize; x++)
		{
			int sum = 0;
			for (int m = -nHalfSESize; m <= nHalfSESize; m++)
			{
				for (int l = -nHalfSESize; l <= nHalfSESize; l++)
				{
					int idxx = x + l;
					int idxy = y + m;
					if ((idxx < 0) || (idxx > w - 1) || (idxy < 0) || (idxy > h - 1)) continue;
					sum += mask.at<uchar>(idxy, idxx);
				}
			}
			//
			if (sum != nFullSEValue)
			{
				for (int m = -nHalfSESize; m <= nHalfSESize; m++)
				{
					for (int l = -nHalfSESize; l <= nHalfSESize; l++)
					{
						int idxx = x + l;
						int idxy = y + m;
						if ((idxx < 0) || (idxx > w - 1) || (idxy < 0) || (idxy > h - 1)) continue;
						morph.at<uchar>(idxy, idxx) = 0;
					}
				}
			}
		}
	}
	return morph;
}


Mat Dilation(Mat mask, int nSESize)
{
	int w = mask.cols;
	int h = mask.rows;
	int val = 255;
	Mat morph(mask.size(), CV_8U, Scalar(0));
	int nHalfSESize = nSESize >> 1;

	for (int y = nHalfSESize; y < h - nHalfSESize; y++)
	{
		for (int x = nHalfSESize; x < w - nHalfSESize; x++)
		{
			int sum = 0;
			for (int m = -nHalfSESize; m <= nHalfSESize; m++)
			{
				for (int l = -nHalfSESize; l <= nHalfSESize; l++)
				{
					int idxx = x + l;
					int idxy = y + m;
					if ((idxx < 0) || (idxx > w - 1) || (idxy < 0) || (idxy > h - 1)) continue;
					sum += mask.at<uchar>(idxy, idxx);
				}
			}
			//
			if (sum > 0)
			{
				for (int m = -nHalfSESize; m <= nHalfSESize; m++)
				{
					for (int l = -nHalfSESize; l <= nHalfSESize; l++)
					{
						int idxx = x + l;
						int idxy = y + m;
						if ((idxx < 0) || (idxx > w - 1) || (idxy < 0) || (idxy > h - 1)) continue;
						morph.at<uchar>(idxy, idxx) = val;
					}
				}
			}
		}
	}
	return morph;
}



void GaussianFiltering(Mat* pSrc, Mat* pDst, double sigma)
{
	int dim = static_cast<int>(8 * sigma + 1.0);
	if (dim < 3) dim = 3;
	if (dim % 2 == 0) dim++;
	GaussianBlur(*pSrc, *pDst, Size(dim, dim), sigma, sigma);
}

void check_weak_edge(Mat* edge, vector<Point>* pvecEdge, int x, int y)
{
	if (edge->at<uchar>(y, x) == WEAK_EDGE)
	{
		edge->at<uchar>(y, x) = STRONG_EDGE;
		pvecEdge->push_back(Point(x, y));
	}
}

void CannyEdgeDetection(Mat* image, Mat* edgeimg, double sigma, double th_low, double th_high)
{
	int i, j;

	int w = image->cols;
	int h = image->rows;

	// 1. ����þ� ���͸�
	Mat gauss(image->size(), CV_8U, Scalar(0));
	GaussianFiltering(image, &gauss, sigma);

	// 2. �׷����Ʈ ���ϱ� (ũ�� & ����)
	Mat gradX(image->size(), CV_64F, Scalar(0));
	Mat gradY(image->size(), CV_64F, Scalar(0));
	Mat gradMag(image->size(), CV_64F, Scalar(0));

	for (j = 1; j < h - 1; j++)
	{
		for (i = 1; i < w - 1; i++)
		{
			gradX.at<double>(j, i) = -gauss.at<uchar>(j - 1, i - 1) - 2.0*gauss.at<uchar>(j, i - 1) - gauss.at<uchar>(j + 1, i - 1)
				+ gauss.at<uchar>(j - 1, i + 1) + 2.0*gauss.at<uchar>(j, i + 1) + gauss.at<uchar>(j + 1, i + 1);

			gradY.at<double>(j, i) = -gauss.at<uchar>(j - 1, i - 1) - 2.0*gauss.at<uchar>(j - 1, i) - gauss.at<uchar>(j - 1, i + 1)
				+ gauss.at<uchar>(j + 1, i - 1) + 2.0*gauss.at<uchar>(j + 1, i) + gauss.at<uchar>(j + 1, i + 1);

			double gx = gradX.at<double>(j, i);
			double gy = gradY.at<double>(j, i);
			gradMag.at<double>(j, i) = sqrt(gx*gx + gy * gy);
		}
	}

	// 3. ���ִ� ����
	// ������ �ִ븦 ���԰� ���ÿ� ���� �Ӱ谪�� �����Ͽ� strong edge�� weak edge�� ���Ѵ�.
	//	Mat edge(image->size(), CV_8U, Scalar(0));	
	vector<Point> strong_edges;

	double ang;
	int district;
	bool local_max;
	for (j = 1; j < h - 1; j++)
	{
		for (i = 1; i < w - 1; i++)
		{
			// �׷����Ʈ ũ�Ⱑ th_low���� ū �ȼ��� ���ؼ��� ������ �ִ� �˻�.
			// ������ �ִ��� �ȼ��� ���ؼ��� ���� ���� �Ǵ� ���� ������ ����.
			double mag = gradMag.at<double>(j, i);
			double gx = gradX.at<double>(j, i);
			double gy = gradY.at<double>(j, i);

			if (mag > th_low)
			{
				// �׷����Ʈ ���� ��� (4�� ����)
				if (gx != .0)
				{
					ang = atan2(gy, gx) * 180 / PI;
					if (((ang >= -22.5f) && (ang < 22.5f)) || (ang >= 157.5f) || (ang < -157.5f))
						district = AREA0;
					else if (((ang >= 22.5f) && (ang < 67.5f)) || ((ang >= -157.5f) && (ang < -112.5f)))
						district = AREA45;
					else if (((ang >= 67.5) && (ang < 112.5)) || ((ang >= -112.5) && (ang < -67.5)))
						district = AREA90;
					else
						district = AREA135;
				}
				else
					district = AREA90;

				// ������ �ִ� �˻�
				local_max = false;
				switch (district)
				{
				case AREA0:
					if ((mag >= gradMag.at<double>(j, i - 1)) && (mag > gradMag.at<double>(j, i + 1)))
						local_max = true;
					break;
				case AREA45:
					if ((mag >= gradMag.at<double>(j - 1, i - 1)) && (mag > gradMag.at<double>(j + 1, i + 1)))
						local_max = true;
					break;
				case AREA90:
					if ((mag >= gradMag.at<double>(j - 1, i)) && (mag > gradMag.at<double>(j + 1, i)))
						local_max = true;
					break;
				case AREA135:
				default:
					if ((mag >= gradMag.at<double>(j - 1, i + 1)) && (mag > gradMag.at<double>(j + 1, i - 1)))
						local_max = true;
					break;
				}

				// ���� ������ ���� ���� ����.
				if (local_max)
				{
					if (mag > th_high)
					{
						edgeimg->at<uchar>(j, i) = STRONG_EDGE;
						strong_edges.push_back(Point(i, j));
					}
					else
					{
						edgeimg->at<uchar>(j, i) = WEAK_EDGE;
					}
				}
			}
		}
	}

	// 4. �����׸��ý� ���� Ʈ��ŷ
	while (!strong_edges.empty())
	{
		Point p = strong_edges.back();
		strong_edges.pop_back();
		int x = p.x, y = p.y;

		// ���� ���� �ֺ��� ���� ������ ���� ����(���� ����)�� ����
		check_weak_edge(edgeimg, &strong_edges, x + 1, y);
		check_weak_edge(edgeimg, &strong_edges, x + 1, y + 1);
		check_weak_edge(edgeimg, &strong_edges, x, y + 1);
		check_weak_edge(edgeimg, &strong_edges, x - 1, y + 1);
		check_weak_edge(edgeimg, &strong_edges, x - 1, y);
		check_weak_edge(edgeimg, &strong_edges, x - 1, y - 1);
		check_weak_edge(edgeimg, &strong_edges, x, y - 1);
		check_weak_edge(edgeimg, &strong_edges, x + 1, y - 1);
	}

	for (j = 0; j < h; j++)
	{
		for (i = 0; i < w; i++)
		{
			// ������ ���� ������ �����ִ� �ȼ��� ��� ������ �ƴ� ������ �Ǵ�.
			if (edgeimg->at<uchar>(j, i) == WEAK_EDGE) edgeimg->at<uchar>(j, i) = 0;
		}
	}
}

int Labeling(Mat imgSrc, Mat imgDst, vector<LabelInfo>& labels)
{
	int w = imgSrc.cols;
	int h = imgSrc.rows;

	//-------------------------------------------------------------------------
	// �ӽ÷� ���̺��� ������ �޸� ������ � ���̺� ����
	//-------------------------------------------------------------------------
	Mat imgMap(imgSrc.size(), CV_32S, Scalar(0));
	int eq_tbl[MAX_LABEL][2] = { { 0, }, };

	//-------------------------------------------------------------------------
	// ù ��° ��ĵ - �ʱ� ���̺� ���� �� � ���̺� ����
	//-------------------------------------------------------------------------
	int label = 0, maxl, minl, min_eq, max_eq;
	for (int j = 1; j < h; j++)
	{
		for (int i = 1; i < w; i++)
		{
			if (imgSrc.at<uchar>(j, i) == 255)
			{
				// �ٷ� �� �ȼ��� ���� �ȼ� ��ο� ���̺��� �����ϴ� ���
				if ((imgMap.at<int>(j - 1, i) != 0) && (imgMap.at<int>(j, i - 1) != 0))
				{
					if (imgMap.at<int>(j - 1, i) == imgMap.at<int>(j, i - 1))
					{
						// �� ���̺��� ���� ���� ���
						imgMap.at<int>(j, i) = imgMap.at<int>(j - 1, i);
					}
					else
					{
						// �� ���̺��� ���� �ٸ� ���, ���� ���̺��� �ο�
						maxl = max(imgMap.at<int>(j - 1, i), imgMap.at<int>(j, i - 1));
						minl = min(imgMap.at<int>(j - 1, i), imgMap.at<int>(j, i - 1));
						imgMap.at<int>(j, i) = minl;

						// � ���̺� ����
						min_eq = min(eq_tbl[maxl][1], eq_tbl[minl][1]);
						max_eq = max(eq_tbl[maxl][1], eq_tbl[minl][1]);

						eq_tbl[eq_tbl[max_eq][1]][1] = min_eq;
					}
				}
				else if (imgMap.at<int>(j - 1, i) != 0)
				{
					// �ٷ� �� �ȼ����� ���̺��� ������ ���
					imgMap.at<int>(j, i) = imgMap.at<int>(j - 1, i);
				}
				else if (imgMap.at<int>(j, i - 1) != 0)
				{
					// �ٷ� ���� �ȼ����� ���̺��� ������ ���
					imgMap.at<int>(j, i) = imgMap.at<int>(j, i - 1);
				}
				else
				{
					// �̿��� ���̺��� �������� ������ ���ο� ���̺��� �ο�
					label++;
					imgMap.at<int>(j, i) = label;
					eq_tbl[label][0] = label;
					eq_tbl[label][1] = label;
				}
			}
		}
	}

	//-------------------------------------------------------------------------
	// � ���̺� ����
	//-------------------------------------------------------------------------
	int temp;
	for (int i = 1; i <= label; i++)
	{
		temp = eq_tbl[i][1];
		if (temp != eq_tbl[i][0])
		{
			eq_tbl[i][1] = eq_tbl[temp][1];
		}
	}

	// � ���̺��� ���̺��� 1���� ���ʴ�� ������Ű��
	int* hash = new int[label + 1];
	memset(hash, 0, sizeof(int)*(label + 1));

	for (int i = 1; i <= label; i++)
	{
		hash[eq_tbl[i][1]] = eq_tbl[i][1];
	}

	int label_cnt = 1;
	for (int i = 1; i <= label; i++)
	{
		if (hash[i] != 0)	hash[i] = label_cnt++;
	}

	for (int i = 1; i <= label; i++)
	{
		eq_tbl[i][1] = hash[eq_tbl[i][1]];
	}
	delete[] hash;


	//-------------------------------------------------------------------------
	// �� ��° ��ĵ - � ���̺��� �̿��Ͽ� ��� �ȼ��� ������ ���̺� �ο�
	//-------------------------------------------------------------------------
	int idx;
	for (int j = 1; j < h; j++)
		for (int i = 1; i < w; i++)
		{
			if (imgMap.at<int>(j, i) != 0)
			{
				idx = imgMap.at<int>(j, i);
				imgDst.at<uchar>(j, i) = eq_tbl[idx][1];
			}
		}

	//-------------------------------------------------------------------------
	// IppLabelInfo ���� �ۼ�
	//-------------------------------------------------------------------------
	labels.resize(label_cnt - 1);
	LabelInfo* pLabel;
	for (int i = 0; i < (int)labels.size(); i++)
	{
		labels[i].minx = INT_MAX;
		labels[i].miny = INT_MAX;

		labels[i].maxx = 0;
		labels[i].maxy = 0;
	}

	for (int j = 1; j < h; j++)
	{
		for (int i = 1; i < w; i++)
		{
			if (imgDst.at<uchar>(j, i) != 0)
			{
				pLabel = &labels.at(imgDst.at<uchar>(j, i) - 1);
				pLabel->pixels.push_back(Point(i, j));
				pLabel->cx += i;
				pLabel->cy += j;

				if (i < pLabel->minx) pLabel->minx = i;
				if (i > pLabel->maxx) pLabel->maxx = i;
				if (j < pLabel->miny) pLabel->miny = j;
				if (j > pLabel->maxy) pLabel->maxy = j;
			}
		}
	}

	for (int i = 0; i < (int)labels.size(); i++)
	{
		labels[i].area = labels[i].pixels.size();
		labels[i].cx /= labels[i].pixels.size();
		labels[i].cy /= labels[i].pixels.size();
		labels[i].w = labels[i].maxx - labels[i].minx;
		labels[i].h = labels[i].maxy - labels[i].miny;
	}

	return (label_cnt - 1);
}


bool CompareAccumValDown(SCircle& a, SCircle& b) { return (a._value > b._value); }
void CircleDetection(Mat* pSrc, int minradius, int maxradius, int radiusoffset, double ratio, vector<SCircle>* pvecCircle)
{
	Mat gradient = pSrc->clone();

	// accumulating
	int w = pSrc->cols;
	int h = pSrc->rows;
	int thresh_grad = 30;

	// 1st scan - accumulating
	Mat curmask = Mat::zeros(Size(w, h), CV_32S);
	for (int r = minradius; r <= maxradius; r += radiusoffset)
	{
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				if (gradient.at<uchar>(y, x) < thresh_grad) continue;
				for (int angle = 0; angle < 360; angle++)
				{
					double rad = angle * PIOVER180;
					int a = x - (int)floor(r*cos(rad));
					int b = y - (int)floor(r*sin(rad));
					if ((a < r) || (a >= w - r) || (b < r) || (b >= h - r)) continue;
					curmask.at<int>(b, a)++;
				}
			}
		}
	}

	// 2nd scan - leave only local maximum	
	int ksize = 15;
	int halfksize = ksize >> 1;
	int minarea = (int)(PI*(minradius + maxradius)*ratio + .5);
	for (int y = halfksize; y < h - halfksize; y++)
	{
		for (int x = halfksize; x < w - halfksize; x++)
		{
			int val = curmask.at<int>(y, x);
			bool bMax = true;
			int nMax = 0;

			if (val < minarea) continue;
			for (int n = -halfksize; n <= halfksize; n++)
			{
				for (int m = -halfksize; m <= halfksize; m++)
				{
					if (n == 0 && m == 0) continue;
					if (val < curmask.at<int>(y + n, x + m))
					{
						bMax = false;
						break;
					}
				}
				if (bMax == false) break;
			}
			if (bMax)
			{
				nMax = curmask.at<int>(y, x);
				for (int n = -halfksize; n <= halfksize; n++)
				{
					for (int m = -halfksize; m <= halfksize; m++)
					{
						curmask.at<int>(y + n, x + m) = 0;
					}
				}
				curmask.at<int>(y, x) = nMax;
			}
		}
	}

	// 3rd scan - circle detection
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			if (curmask.at<int>(y, x) > minarea)
			{
				SCircle accu;
				accu._center = Point(x, y);
				accu._radius = (minradius + maxradius) >> 1;
				accu._value = curmask.at<int>(y, x);
				pvecCircle->push_back(accu);
			}
		}
	}

	// sorting
	sort(pvecCircle->begin(), pvecCircle->end(), CompareAccumValDown);

	Mat resultcolor(pSrc->size(), CV_8UC3);
	cvtColor(*pSrc, resultcolor, CV_GRAY2BGR);
	int circnt = (int)pvecCircle->size();
	for (int i = 0; i < circnt; i++)
	{
		if (34 <= pvecCircle->at(i)._radius && pvecCircle->at(i)._radius <= 37)
			circle(resultcolor, pvecCircle->at(i)._center, pvecCircle->at(i)._radius, Scalar(0, 255, 255), 2);
		else if (50.f <= pvecCircle->at(i)._radius && pvecCircle->at(i)._radius <= 53.f)
			circle(resultcolor, pvecCircle->at(i)._center, pvecCircle->at(i)._radius, Scalar(0, 0, 255), 2);
		else  if (39.f <= pvecCircle->at(i)._radius && pvecCircle->at(i)._radius <= 43.f)
			circle(resultcolor, pvecCircle->at(i)._center, pvecCircle->at(i)._radius, Scalar(255, 0, 255), 2);
	}
	imshow("result", resultcolor);
}



