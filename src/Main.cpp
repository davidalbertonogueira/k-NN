#include "KNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>

const size_t NUMBER_CLASSES = 3;//10;
const size_t NUMBER_INSTANCES = 10;//100;
const size_t NUMBER_FEATURES = 2;// 25;

int main() {
  std::random_device rd;     // only used once to initialise (seed) engine
  std::mt19937 rng(rd());    // random-number engine (Mersenne-Twister)
  std::uniform_int_distribution<int> class_id_rnd(0, NUMBER_CLASSES);
  std::uniform_real_distribution<double> feature_value_rnd(-10, 10);

  {
    KNN my_knn;

    //Create 10 training instances
    for (size_t i = 0; i < NUMBER_INSTANCES; i++) {
      Instance instance;
      instance.class_membership = class_id_rnd(rng);
      instance.features.resize(NUMBER_FEATURES);
      for (size_t j = 0; j < NUMBER_FEATURES; j++) {
        instance.features[j] = feature_value_rnd(rng);
      }
      my_knn.AddInstance(instance);
    }

    //Create one test instance
    Instance new_instance;
    new_instance.features.resize(NUMBER_FEATURES);
    for (size_t j = 0; j < NUMBER_FEATURES; j++) {
      new_instance.features[j] = feature_value_rnd(rng);
    }
    my_knn.Classify(5,
                    DistanceMetricProvider::EuclideanDistance,
                    new_instance.features,
                    &new_instance.class_membership);

    for (size_t i = 0; i < NUMBER_INSTANCES; i++) {
      uint32_t test;
      my_knn.Classify(1,
                      DistanceMetricProvider::EuclideanDistance,
                      my_knn.GetInstances()[i].features,
                      &test);
      if (test != my_knn.GetInstances()[i].class_membership) {
        std::cout << "Error" << std::endl;
        break;
      }
    }
  }
  {
    KNN my_knn;

    //Create 10 training instances
    std::vector<Instance> instances = {
      {{-10.0,-10.0 },0},
      {{ -9.0, -9.0 },1}, {{-8.0,-8.0},2}, {{-7.0,-7.0},3}, {{-6.0,-6.0},4},
      {{ -5.0, -5.0 },5},
      {{ -4.0, -4.0 },1}, {{-3.0,-3.0},2}, {{-2.0,-2.0},3}, {{-1.0,-1.0},4},
      {{  0.0,  0.0 },0},
      {{  1.0,  1.0 },1}, {{ 2.0, 2.0},2}, {{ 3.0, 3.0},3}, {{ 4.0, 4.0},4},
      {{  5.0,  5.0 },5},
      {{  6.0,  6.0 },1}, {{ 7.0, 7.0},2}, {{ 8.0, 8.0},3}, {{ 9.0, 9.0},4},
      {{ 10.0, 10.0 },0},
      {{-10.0, 10.0 },0},
      {{ -9.0,  9.0 },1}, {{-8.0, 8.0},2}, {{-7.0, 7.0},3}, {{-6.0, 6.0},4},
      {{ -5.0,  5.0 },5},
      {{ -4.0,  4.0 },1}, {{-3.0, 3.0},2}, {{-2.0, 2.0},3}, {{-1.0, 1.0},4},
      {{  0.0,  0.0 },0},
      {{  1.0, -1.0 },1}, {{ 2.0,-2.0},2}, {{ 3.0,-3.0},3}, {{ 4.0,-4.0},4},
      {{  5.0, -5.0 },5},
      {{  6.0, -6.0 },1}, {{ 7.0,-7.0},2}, {{ 8.0,-8.0},3}, {{ 9.0,-9.0},4},
      {{ 10.0, -10.0},0}
    };

    for (size_t i = 0; i < instances.size(); i++) {
      my_knn.AddInstance(instances[i]);
    }
    {
      //Create one test instance
      Instance new_instance = { { 0.0,0.0 }, 0 };
      uint32_t test;
      my_knn.Classify(5,
                      DistanceMetricProvider::EuclideanDistance,
                      new_instance.features,
                      &test);
      std::cout << "Class " << test << std::endl;
      if (test != new_instance.class_membership) {
        std::cout << "Error" << std::endl;
      }
    }
    {
      //Create one test instance
      Instance new_instance = { { -10.0,0.0 }, 5 };
      uint32_t test;
      my_knn.Classify(2,
                      DistanceMetricProvider::EuclideanDistance,
                      new_instance.features,
                      &test);
      std::cout << "Class " << test << std::endl;
      if (test != new_instance.class_membership) {
        std::cout << "Error" << std::endl;
      }
    }
  }
  return 0;
}
