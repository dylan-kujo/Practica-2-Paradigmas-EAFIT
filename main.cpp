#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "LinearRegression.h"

bool loadDataset(const std::string& filename, std::vector<std::vector<double>>& X, std::vector<double>& y, bool hasHeader = true) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo: " << filename << std::endl;
        return false;
    }

    std::string line;
    if (hasHeader && std::getline(file, line)) {
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        double target = 0.0;
        int colIdx = 0;

        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (...) {
            }
        }

        if (!row.empty()) {
            target = row.back();
            row.pop_back(); 
            X.push_back(row);
            y.push_back(target);
        }
    }
    file.close();
    return true;
}

int main() {
    std::cout << "=== PRUEBA DE REGRESION LINEAL - EAFIT ===" << std::endl;

    std::vector<std::vector<double>> X_ice;
    std::vector<double> y_ice;
    if (loadDataset("Ice_cream_selling_data.csv", X_ice, y_ice)) {
        std::cout << "\n[+] Dataset 'Ice_cream_selling_data.csv' cargado con exito. Filas: " << X_ice.size() << std::endl;
        
        LinearRegression mlr_ice;
        mlr_ice.fit(X_ice, y_ice, 1000, 0.01);

        std::cout << "Weights (Ice cream): ";
        for (double w : mlr_ice.getWeights()) std::cout << w << " ";
        std::cout << "\nBias (Ice cream): " << mlr_ice.getBias() << std::endl;
        std::cout << "Score (MSE): " << mlr_ice.score(X_ice, y_ice) << std::endl;
    }

    std::vector<std::vector<double>> X_students;
    std::vector<double> y_students;
    if (loadDataset("student_exam_scores.csv", X_students, y_students)) {
        std::cout << "\n[+] Dataset 'student_exam_scores.csv' cargado con exito. Filas: " << X_students.size() << std::endl;
        
        LinearRegression mlr_students;
        mlr_students.fit(X_students, y_students, 1000, 0.01);

        std::cout << "Weights (Student exam): ";
        for (double w : mlr_students.getWeights()) std::cout << w << " ";
        std::cout << "\nBias (Student exam): " << mlr_students.getBias() << std::endl;
        std::cout << "Score (MSE): " << mlr_students.score(X_students, y_students) << std::endl;
    }

    return 0;
}