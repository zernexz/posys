#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "vol.h"
#include "layer.h"
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
class SoftmaxLayer : public Layer<FP>{
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

	
	vector<FP> es;
	
	//num_neurons   In:{d,x,y} Conf:{l1_decay,l2_decay}
	SoftmaxLayer(int in_depth,int in_sx,int in_sy){
		this->layer_type="softmax";
		this->in_act=NULL;
		this->out_act=NULL;
		this->in_depth=in_depth;
		this->in_sx=in_sx;
		this->in_sy=in_sy;
		this->num_inputs=in_sx*in_sy*in_depth;
		this->out_depth=num_inputs;
		this->out_sx=1;
		this->out_sy=1;
	}
	~SoftmaxLayer(){
		//cout << "clearrr" << endl;
		this->es.clear();
		//cout << "clearrr3" << endl;
		if(this->in_act != NULL){delete this->in_act;this->in_act=NULL;}
		//cout << "clearrr5" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		//cout << "clearrr4" << endl;
	}

	Vol<FP>* forward(Vol<FP>* V,bool is_training=false){
		if(this->in_act != NULL)
			delete this->in_act;
		this->in_act = V->clone();
		
		Vol<FP>* A = new Vol<FP>(1,1,this->out_depth,FP(0));

		
		FP amax = V->w[0];
		for(int i=1;i<this->out_depth;i++){
			if(V->w[i] > amax) amax = V->w[i];
		}

		Utils<FP> ut;
		vector<FP> es = ut.zeros(this->out_depth);
		FP esum(0.0);
		
		for(int i=0;i<this->out_depth;i++){
			FP e( exp(V->w[i] - amax) );
			esum += e;
			es[i] = e;
		}

		for(int i=0;i<this->out_depth;i++){
			es[i] /= esum;
			A->w[i] = es[i];
		}



		{this->es.clear();}
		//cout << "feed f" << endl;
		this->es = es;

		//cout << "feed e" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		//cout << "feed f" << endl;
		this->out_act = A;
		//cout << "feed g" << endl;
		//cout << "feed h" << endl;
		return A->clone();
	}
	void backward(int y){
		Vol<FP>* x=this->in_act;
		Utils<FP> ut;
		
		x->dw = ut.zeros(x->w.size());
		for(int i=0;i<this->out_depth;i++){
			FP indicator( (i==y)?1:0 );
			FP mul(  -(indicator - this->es[i])  );
			x->dw[i] = mul;
		}
		
		//return -log(this->es[y]);		
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
