
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include <map>
#include <algorithm>

struct data
{
	std::string data_type;
	double feature1;
	double feature2;
};

std::vector <data> trainedDatas;
std::vector <data> testDatas;

void getDatas(std::string filePath, std::vector<data> &datas)
{
	std::ifstream infile;
	infile.open(filePath, std::ios::in);
	if (!infile.is_open())
	{
		return;
	}

	std::string line;
	while (std::getline(infile, line))
	{
		std::istringstream linestream(line);
		std::string data_type;
		double feature1, feature2;

		std::getline(linestream, data_type, ',');

		std::string strTemp;
		std::getline(linestream, strTemp, ',');
		std::istringstream issF1(strTemp);		
		issF1 >> feature1;

		std::getline(linestream, strTemp, ',');
		std::istringstream issF2(strTemp);
		issF2 >> feature2;

		data aData;
		aData.data_type = data_type;
		aData.feature1 = feature1;
		aData.feature2 = feature2;

		datas.push_back(aData);
	}
}

void showDatas(std::vector<data> &datas)
{
	std::cout << "enter " << __func__ << std::endl;
	for (int i = 0; i < (int)datas.size(); i++)
	{
		std::cout << i << "\t Type: " << datas[i].data_type << 
            "\t F1: " << datas[i].feature1 << 
            "\t F2: " << datas[i].feature2 << std::endl;
	}

	std::cout << "endof " << __func__ << std::endl;
}

double calcDistance(double f1, double f2, data aData)
{
	return sqrtf((f1 - aData.feature1) * (f1 - aData.feature1) + (f2 - aData.feature2) * (f2 - aData.feature2));
}

struct ST_TypeDistance
{
	std::string data_type;
	double dist;
};

bool compDistance(ST_TypeDistance &a, ST_TypeDistance &b)
{
	return a.dist < b.dist;
}

std::string calcType(double f1, double f2, int k = 5)
{
	std::vector<ST_TypeDistance> distList;

	for (int i = 0; i < (int)trainedDatas.size(); i++)
	{
		ST_TypeDistance aDist;
		aDist.data_type = trainedDatas[i].data_type;
		aDist.dist = calcDistance(f1, f2, trainedDatas[i]);
		distList.push_back(aDist);
	}

	std::sort(distList.begin(), distList.end(), compDistance);

	std::map<std::string, int> typeMap;
	for (int i = 0; i < k; i++)
	{
		typeMap[distList[i].data_type]++;
		//std::cout << i << "\tType: " << distList[i].data_type << "\tDist: " << distList[i].dist << std::endl;
	}

	std::string ans;
	int ansCount = 0;
	for (auto iter : typeMap)
	{
		if (iter.second > ansCount)
		{
			ans = iter.first;
			ansCount = iter.second;
		}
	}

	return ans;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: please input 1 data path\n";
        return 1;
    }
    std::cout << "trained_data file: " << argv[1] << '\n';

	getDatas(argv[1], trainedDatas);
    getDatas(argv[2], testDatas);

    int rightCount = 0;
    int totalSize = testDatas.size();
    for(int i = 0; i < (int)testDatas.size(); i++)
    {
        std::cout << i << "\tf1: " << testDatas[i].feature1 
            << "\tf2: " << testDatas[i].feature2;
        std::string type = calcType(testDatas[i].feature1, testDatas[i].feature2);
        std::cout << "\t" << type;

        if (testDatas[i].data_type == type)
        {
            rightCount++;
            std::cout << "\tTrue";
        }
        else
        {
            std::cout << "\tFalse";
        }

        std::cout << std::endl;
    }

    std::cout << "true rate = " << (double)((double)rightCount / (double)totalSize);
	
    return 0;
}