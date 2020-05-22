#ifndef GRADIENTDESCENT_HPP
#define GRADIENTDESCENT_HPP

#include <vector>

class GradientDescent
{
public:
    GradientDescent(double alpha, int iterations);
    ~GradientDescent();

public:
    void fit(std::vector<std::vector<double>> featuresList, int featuresCount, std::vector<double> targetList);
	std::vector<double> predict(std::vector<std::vector<double>> featuresList);

private:
    double calcSignalTarget(std::vector<double> features);
    std::vector<double> calcDiffs(std::vector<double> preds, std::vector<double>targets);
    std::vector<double> calcMutiDiffs(std::vector<double> diffs, std::vector<std::vector<double>> featuresList, int nFeatureIndex);
    double calcVectorSum(std::vector<double> vec);
	std::vector<double> calcVectorPow(std::vector<double> vec, int k);

private:
    double m_alpha;
    int m_iterations;
    int m_thetaCount;
    std::vector<double> m_thetas;
    double m_theta0;

};


#endif