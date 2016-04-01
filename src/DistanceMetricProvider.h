#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

/**
* @class DistanceMetricProvider
* @brief Offer different distance metrics that can be used in the KNN.
*/
class DistanceMetricProvider {
public:
  static bool EuclideanDistance(const std::vector<double> & a,
                                const std::vector<double> & b,
                                double *result);
  static bool EuclideanSquaredDistance(const std::vector<double> & a,
                                       const std::vector<double> & b,
                                       double *result);
  static bool CityBlockDistance(const std::vector<double> & a,
                                const std::vector<double> & b,
                                double *result);
  static bool ChebyshevDistance(const std::vector<double> & a,
                                const std::vector<double> & b,
                                double *result);
};
