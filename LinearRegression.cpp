#include "LinearRegression.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

LinearRegression::LinearRegression() : bias(0.0), is_scaled(false) {}

void LinearRegression::data_scaling(std::vector<std::vector<double>>& X) {
    if (X.empty() || X[0].empty()) return;
    size_t num_samples = X.size();
    size_t num_features = X[0].size();

    mean_X.assign(num_features, 0.0);
    std_X.assign(num_features, 1.0);

    for (size_t j = 0; j < num_features; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < num_samples; ++i) {
            sum += X[i][j];
        }
        mean_X[j] = sum / num_samples;
    }

    for (size_t j = 0; j < num_features; ++j) {
        double sum_sq_diff = 0.0;
        for (size_t i = 0; i < num_samples; ++i) {
            sum_sq_diff += std::pow(X[i][j] - mean_X[j], 2);
        }
        std_X[j] = std::sqrt(sum_sq_diff / num_samples);
        if (std_X[j] == 0.0) std_X[j] = 1.0; 
    }

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            X[i][j] = (X[i][j] - mean_X[j]) / std_X[j];
        }
    }
    is_scaled = true;
}

void LinearRegression::fit(std::vector<std::vector<double>>& X, const std::vector<double>& y, int epochs, double learning_rate) {
    if (X.empty() || y.empty() || X.size() != y.size()) return;

    data_scaling(X);

    size_t num_samples = X.size();
    size_t num_features = X[0].size();

    weights.assign(num_features, 0.0);
    bias = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> dw(num_features, 0.0);
        double db = 0.0;

        for (size_t i = 0; i < num_samples; ++i) {
            double prediction = predict_single(X[i]);
            double error = prediction - y[i];

            for (size_t j = 0; j < num_features; ++j) {
                dw[j] += error * X[i][j];
            }
            db += error;
        }

        for (size_t j = 0; j < num_features; ++j) {
            weights[j] -= (learning_rate / num_samples) * dw[j];
        }
        bias -= (learning_rate / num_samples) * db;
    }
}

double LinearRegression::predict_single(const std::vector<double>& x) const {
    double prediction = bias;
    for (size_t i = 0; i < weights.size() && i < x.size(); ++i) {
        prediction += weights[i] * x[i];
    }
    return prediction;
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> predictions;
    for (const auto& row : X) {
        std::vector<double> row_scaled = row;
        if (is_scaled && !mean_X.empty()) {
            for (size_t j = 0; j < row.size() && j < mean_X.size(); ++j) {
                row_scaled[j] = (row[j] - mean_X[j]) / std_X[j];
            }
        }
        predictions.push_back(predict_single(row_scaled));
    }
    return predictions;
}

double LinearRegression::score(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const {
    auto preds = predict(X);
    double mse = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        mse += std::pow(preds[i] - y[i], 2);
    }
    return mse / y.size();
}

std::vector<double> LinearRegression::getWeights() const {
    return weights;
}

double LinearRegression::getBias() const {
    return bias;
}