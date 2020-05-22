
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include <map>
#include <algorithm>

#include "GradientDescent.hpp"

struct ST_data
{
	double                   target;
    int                      featuresCount;
	std::vector<double>      features;
};

std::vector <ST_data> trainedDatas;
std::vector <ST_data> testDatas;

void getDatas(std::string filePath, std::vector<ST_data> &datas)
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
		std::string strTarget;

		std::getline(linestream, strTarget, ',');
        double target;
        std::istringstream issTarget(strTarget);
        issTarget >> target;

        ST_data aData;
        aData.target = target;
        aData.featuresCount = 0;

        std::string strTemp;
        while(std::getline(linestream, strTemp, ','))
        {
            std::istringstream issFeature(strTemp);
            double feature;
		    issFeature >> feature;

            aData.featuresCount++;
            aData.features.push_back(feature);
        }

		datas.push_back(aData);
	}
}

void showDatas(std::vector<ST_data> &datas)
{
	std::cout << "enter " << __func__ << std::endl;
	for (int i = 0; i < (int)datas.size(); i++)
	{
		std::cout << i << "\t target: " << datas[i].target;
        for(int j = 0; j < datas[i].featuresCount; j++)
        {
            std::cout << "\t feature" << j << ": " << datas[i].features[j];
        }
        std::cout << std::endl;
	}

	std::cout << "endof " << __func__ << std::endl;
}

void testGradientDescent()
{
	GradientDescent aGD(0.00001, 200);
	std::vector<double> trained_targets;
	std::vector<std::vector<double>> trained_features;

	std::vector<double> test_targets;
	std::vector<std::vector<double>> test_features;

	for (int i = 0; i < (int)trainedDatas.size(); i++)
	{
		trained_features.push_back(trainedDatas[i].features);
		trained_targets.push_back(trainedDatas[i].target);
	}

	for (int i = 0; i < (int)testDatas.size(); i++)
	{
		test_features.push_back(testDatas[i].features);
		test_targets.push_back(testDatas[i].target);
	}

	aGD.fit(trained_features, trainedDatas[0].featuresCount, trained_targets);
	std::vector<double> preds = aGD.predict(test_features);

	std::fstream fs;
	fs.open("preds.csv", std::ios::out);
	if (fs.is_open())
	{
		for (int i = 0; i < (int)test_targets.size(); i++)
		{
			fs << preds[i] << "," << test_targets[i] << std::endl;
			std::cout << preds[i] << "," << test_targets[i] << std::endl;
		}
	}
	fs.close();
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

    // showDatas(trainedDatas);
    // showDatas(testDatas);

	testGradientDescent();

    return 0;
}