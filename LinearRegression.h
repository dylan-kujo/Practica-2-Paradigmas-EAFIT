#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>

class LinearRegression {
private:
    std::vector<double> weights;
    double bias;
    
   
    std::vector<double> mean_X;
    std::vector<double> std_X;
    bool is_scaled;

public:
    LinearRegression();

    
    void data_scaling(std::vector<std::vector<double>>& X);
    void fit(std::vector<std::vector<double>>& X, const std::vector<double>& y, int epochs, double learning_rate);
    double predict_single(const std::vector<double>& x) const;
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    double score(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const;

    
    std::vector<double> getWeights() const;
    double getBias() const;
};

#endif