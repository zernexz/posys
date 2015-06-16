#ifndef FULLYCONNLAYER_H
#define FULLYCONNLAYER_H

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
class FullyConnLayer : public Layer<FP>{
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

	
	//num_neurons   In:{d,x,y} Conf:{l1_decay,l2_decay}
	FullyConnLayer(int num_neurons,int in_depth,int in_sx,int in_sy,FP l1_decay_mul=FP(0),FP l2_decay_mul=FP(1),FP bias_pref=FP(0)):in_depth(in_depth),in_sx(in_sx),in_sy(in_sy),l1_decay_mul(l1_decay_mul),l2_decay_mul(l2_decay_mul),layer_type("fc"),bias(bias_pref),biases(NULL),out_depth(num_neurons){
	this->in_act=NULL;
	this->out_act=NULL;
	cout << "cv 0" << endl;
	this->num_inputs = this->in_sx * this->in_sy * this->in_depth;
	this->out_sx = 1;
	this->out_sy = 1;
	cout << "cv 1" << endl;
	this->bias = bias_pref;
	cout << "cv 2" << endl;
	for(int i=0;i<this->out_depth;i++){
		cout << "cv 2.1" << endl;
		this->filters.push_back(new Vol<FP>(1,1,this->num_inputs));cout << "cv 2.2" << endl;}
	cout << "cv 3" << endl;
	this->biases = new Vol<FP>(1, 1, this->out_depth , FP(bias) );
	cout << "cv 4" << endl;
	}
	~FullyConnLayer(){
		cout << "clearrr" << endl;
		if(this->biases != NULL){delete this->biases;this->biases=NULL;}
		cout << "clearrr1" << endl;

		for(int i=0;i<this->filters.size();i++)
			delete this->filters[i];
		cout << "clearrr2" << endl;
		this->filters.clear();
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

		cout << "feed b" << endl;
		Vol<FP>* A = new Vol<FP>(1,1,this->out_depth,FP(0.0));
		cout << "feed c" << endl;
		
		cout << "feed ddd" << endl;
		for(int i=0;i<this->out_depth;i++){
			FP a(0);
			Vol<FP>* wi = this->filters[i];
			for(int d=0;d<this->num_inputs;d++){
				//Vw * wi
				a += V->w[d] * wi->w[d];
			}
			a+=this->biases->w[i];
			A->w[i] = a;
		}

		cout << "feed e" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		cout << "feed f" << endl;
		this->out_act = A;
		cout << "feed g" << endl;
		cout << "feed h" << endl;
		return A->clone();
	}
	void backward(int tmpy=0){
		Vol<FP>* V = this->in_act;
		Utils<FP> ut;
		V->dw = ut.zeros(V->w.size());
		
		for(int i=0;i<this->out_depth;i++){
			Vol<FP>* tfi = this->filters[i];
			FP chain_grad = this->out_act->dw[i];
			for(int d=0;d<this->num_inputs;d++){
				V->dw[d] += tfi->w[d]*chain_grad;
				tfi->dw[d] += V->w[d]*chain_grad;
			}
			this->biases->dw[i] += chain_grad;
		}

	}
	
	vector< map<string,void* > > getParamsAndGrads(){
		vector< map<string,void* > > v;
		for(int i=0;i<this->out_depth;i++){
			map<string,void* > m;
			m["params"] = (void*) &this->filters[i]->w;
			m["grads"] = (void*) &this->filters[i]->dw;
			m["l2_decay_mul"] = (void*) &this->l2_decay_mul;
			m["l1_decay_mul"] = (void*) &this->l1_decay_mul;
			v.push_back(m);
		}
		map<string,void* > m;
		m["params"] = (void*) &this->biases->w;
		m["grads"] = (void*) &this->biases->dw;
		FP* l1=new FP(0.0);
		FP* l2=new FP(0.0);
		m["l2_decay_mul"] = (void*) &l2;
		m["l1_decay_mul"] = (void*) &l1;
		v.push_back(m);
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
