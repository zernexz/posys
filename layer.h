#ifndef LAYER_H
#define LAYER_H

#include "vol.h"
#include "utils.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <random>
#include <cmath>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>

template < typename FP >
class Layer{
private:
public:
virtual Vol<FP>* forward(Vol<FP>* V,bool is_training)=0;
virtual void backward(int y)=0;
virtual vector< map<string, vector<FP>* > > getParamsAndGrads()=0;
virtual string get_layer_type()=0;
virtual Vol<FP>* get_in_act()=0;
virtual Vol<FP>* get_out_act()=0;
virtual vector<FP> get_all_w()=0;
virtual void set_all_w(vector<FP> aw)=0;

string layer_type;
Vol<FP>* in_act;
Vol<FP>* out_act;
};

#endif

