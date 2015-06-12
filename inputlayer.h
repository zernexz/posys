#ifndef INPUTLAYER_H
#define INPUTLAYER_H

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

#ifndef NOT_USE
#define NOT_USE 2<<20
#endif

using namespace std;

template < typename FP >
class InputLayer{
private:

public:
	int out_depth;
	int out_sx;
	int out_sy;

	int in_depth;
	int in_sx;
	int in_sy;

	int num_inputs;
	
	FP l1_decay_mul;
	FP l2_decay_mul;

	string layer_type;

	FP bias;
	Vol<FP>* biases;

	vector<Vol<FP>* > filters;

	Vol<FP>* in_act;
	Vol<FP>* out_act;
	
	//num_neurons   In:{d,x,y} Conf:{l1_decay,l2_decay}
	InputLayer(int out_depth,int out_sx,int out_sy):layer_type("input"),in_act(NULL),out_act(NULL),out_depth(out_depth),out_sx(out_sx),out_sy(out_sy){
	}
	~InputLayer(){
		cout << "clearrr" << endl;
		cout << "clearrr3" << endl;
		if(this->in_act != NULL){delete this->in_act;this->in_act=NULL;}
		cout << "clearrr5" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		cout << "clearrr4" << endl;
	}

	Vol<FP>* forward(Vol<FP>* V,bool is_training=false){
		if(this->in_act != NULL)
			delete this->in_act;
		this->in_act = V->clone();

		cout << "feed e" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		cout << "feed f" << endl;
		this->out_act = V;
		cout << "feed g" << endl;
		cout << "feed h" << endl;
		return V->clone();
	}
	void backward(){
	}
	
	vector< map<string,void* > > getParamsAndGrads(){
		vector< map<string,void* > > v;
		return v;
	}
	
};


/**
 * Mat::data Specification
 * 2x2  1 channel
 * [ R , R ;
 *   R , R ]
 * 
 * 2x2  2 channel
 * [ R , G , R , G ;
 *   R , G , R , G ]
 * 
 * 2x2  3 channel
 * [ R , G , B , R , G , B ;
 *   R , G , B . R , G , B ] 
 */
#endif