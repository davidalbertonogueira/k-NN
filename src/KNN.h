#include "DistanceMetricProvider.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

/**
* @struct Instance
* @brief Instance of training sample for the KNN.
* Holds the features values and class membership of a training sample.
*/
struct Instance {
  std::vector<double> features;
  uint32_t  class_membership;
};

/**
* @class KNN
* @brief Implements a KNN algorithm.
* k-NN is a type of instance-based learning (lazy learning), and therefore, it
* does not offer regular Train/Test methods.
* It stores instances from the training set (training samples), and upon a
* classification request for a new instance, compares it against the train set.
*/
class KNN {
public:
  /**
  * @brief Stores the features values and class membership of a training sample.
  * @param [in] instance Training sample to store.
  */
  bool AddInstance(const Instance & instance);
  /**
  * @brief Classify a new instance.
  * @param [in] k                     Number of nearest neighbors from which
  *                                   to select the most common class.
  * @param [in] distance_function     Function that defines the
  *                                   distance metric to be used in the KNN.
  * @param [in] new_instance_features Feature vector of the instance
  *                                   to be classified.
  * @param [out] class_membership     Predicted class for the input instance.
  */
  bool Classify(uint32_t k,
                bool(*distance_function)(const std::vector<double> &,
                                         const std::vector<double> &,
                                         double *),
                const std::vector<double> & new_instance_features,
                uint32_t* class_membership);
  // GETTERS
  const std::vector<Instance> GetInstances() const { return instances; };
protected:
  std::vector<Instance> instances;
};
