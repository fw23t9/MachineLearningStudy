#include <iostream>

#include "GradientDescent.hpp"

GradientDescent::GradientDescent(double alpha, int iterations)
{
    m_thetaCount = 0;
    m_theta0 = 0;

    m_alpha = alpha;
    m_iterations = iterations;
}

GradientDescent::~GradientDescent()
{

}

double GradientDescent::calcSignalTarget(std::vector<double> features)
{
    double ans = m_theta0;
    for(int i = 0; i < (int)m_thetas.size(); i++)
    {
        ans += m_thetas[i] * features[i];
    }

    return ans;
}

std::vector<double> GradientDescent::predict(std::vector<std::vector<double>> featuresList)
{
    int len = featuresList.size();
    std::vector<double> preds(len, 0);
    for(int i = 0; i < len; i++)
    {
        preds[i] = calcSignalTarget(featuresList[i]);
    }

    return preds;
}

std::vector<double> GradientDescent::calcDiffs(std::vector<double> preds, std::vector<double>targets)
{
    int nPredLen = preds.size();
    int nTargetLen = targets.size();
    int len = nPredLen < nTargetLen ? nPredLen : nTargetLen;
    
    std::vector<double> diffs(len, 0);
    for(int i = 0; i < len; i++)
    {
        diffs[i] = preds[i] - targets[i];
    }

    return diffs;
}

std::vector<double> GradientDescent::calcMutiDiffs(std::vector<double> diffs, std::vector<std::vector<double>> featuresList, int nFeatureIndex)
{
    int nDiffLen = diffs.size();
    int nFeaturesLen = featuresList.size();
    int len = nDiffLen < nFeaturesLen ? nDiffLen : nFeaturesLen;

    std::vector<double> mutiDiffs(len, 0);
    for(int i = 0; i < len; i++)
    {
        mutiDiffs[i] = diffs[i] * featuresList[i][nFeatureIndex];
    }

    return mutiDiffs;
}

double GradientDescent::calcVectorSum(std::vector<double> vec)
{
    double ans = 0;
    for(int i = 0; i < (int)vec.size(); i++)
    {
        ans += vec[i];
    }

    return ans;
}

std::vector<double> GradientDescent::calcVectorPow(std::vector<double> vec, int k)
{
	std::vector<double> ans(vec.size(), 0);
	for (int i = 0; i < (int)vec.size(); i++)
	{
		ans[i] = pow(vec[i], k);
	}

	return ans;
}

void GradientDescent::fit(std::vector<std::vector<double>> featuresList, int featuresCount, std::vector<double> targetList)
{
    m_thetaCount = featuresCount;

	int nFeaturesLen = featuresList.size();
	int nTargetLen = targetList.size();
	int nTrainDatasLen = nFeaturesLen < nTargetLen ? nFeaturesLen : nTargetLen;

    m_thetas.resize(m_thetaCount);
    for(int i = 0; i < (int)m_thetas.size(); i++)
    {
		double colSum = 0;
		for (int j = 0; j < nTrainDatasLen; j++)
		{
			colSum += featuresList[j][i];
		}
        m_thetas[i] = (calcVectorSum(targetList) / colSum) / m_thetaCount;
    }
    m_theta0 = targetList[0] - calcSignalTarget(featuresList[0]);
	//std::cout << "iteration i: " << -1 << " t0 " << m_theta0;
	//for (int j = 0; j < m_thetaCount; j++)
	//{
	//	std::cout << " t" << (j + 1) << " " << m_thetas[j];
	//}
	//std::cout << std::endl;

    for(int i = 0; i < m_iterations; i++)
    {
		double old_theta0 = m_theta0;
		std::vector<double> old_thetas(m_thetas);

		double tmp0 = 0.0;
		std::vector<double> tmpThetaList(m_thetaCount, 0);

		for (int j = 0; j < nTrainDatasLen; j++)
		{
			tmp0 += targetList[j] - calcSignalTarget(featuresList[j]);
			for (int k = 0; k < m_thetaCount; k++)
			{
				tmpThetaList[k] += (targetList[j] - calcSignalTarget(featuresList[j])) * featuresList[j][k];
			}
		}
		m_theta0 = m_theta0 + m_alpha * tmp0 / nTrainDatasLen;
		for (int j = 0; j < m_thetaCount; j++)
		{
			m_thetas[j] = m_thetas[j] + m_alpha * tmpThetaList[j] / nTrainDatasLen;
		}

		//std::cout << "iteration i: " << i << " t0 " << m_theta0;
		//for (int j = 0; j < m_thetaCount; j++)
		//{
		//	std::cout << " t" << (j + 1) << " " << m_thetas[j];
		//}
		//std::cout << std::endl;

		//std::vector<double> preds = predict(featuresList);
		//std::vector<double> diffs = calcDiffs(preds, targetList);
		
		double err = pow((m_theta0 - old_theta0), 2);
		for (int j = 0; j < m_thetaCount; j++)
		{
			err += pow((m_thetas[j] - old_thetas[j]), 2);
		}
		if (err < 0.000003)
		{
			break;
		}
    }
}