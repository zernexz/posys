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
	FullyConnLayer(int num_neurons,int in_depth,int in_sx,int in_sy,FP l1_decay_mul=FP(0),FP l2_decay_mul=FP(1),FP bias_pref=FP(0)):in_depth(in_depth),in_sx(in_sx),in_sy(in_sy),l1_decay_mul(l1_decay_mul),l2_decay_mul(l2_decay_mul),layer_type("fc"),bias(bias_pref),biases(NULL){
	this->in_act=NULL;
	this->out_act=NULL;
	cout << "cv 0" << endl;
	this->num_inputs = this->in_sx * this->in_sy * this->in_depth;
	this->out_sx = 1;
	this->out_sy = 1;
	this->out_depth = num_neurons;

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

vector<FP> get_all_w(){
	vector<FP> out;

	Vol<FP>* V;
	vector< Vol<FP>* > list;

	for(int q=0;q<this->filters.size();q++)
		list.push_back(this->filters[q]);
	list.push_back(this->biases);

	for(int z=0;z<list.size();z++){
		V=list[z];
		int size=V->w.size();
		//cout << size << endl;
		for(int q=0;q<size;q++){
			out.push_back(V->w[q]);
		}
	}

	return out;
}
void set_all_w(vector<FP> aw){
	Vol<FP>* V;
	vector< Vol<FP>* > list;
	vector<int> slist;
	int as=0;

	for(int q=0;q<this->filters.size();q++){
		list.push_back(this->filters[q]);
		slist.push_back(this->filters[q]->w.size());
		as+=this->filters[q]->w.size();		
	}
	slist.push_back(this->biases->w.size());
	list.push_back(this->biases);
	as+=this->biases->w.size();

	for(int i=0,q=0;i<slist.size();i++){
		V = list[i];
		for(int j=0;j<slist[i];j++,q++){
			V->w[j]=aw[q];
			
		}
	}

}
	Vol<FP>* forward(Vol<FP>* V,bool is_training=false){
		this->in_act = V;

		//cout << "feed b" << endl;
		Vol<FP>* A = new Vol<FP>(1,1,this->out_depth,FP(0.0));
		//cout << "feed c" << endl;
		

		//cout << " ::: ";
		//cout << "feed ddd" << endl;
		for(int i=0;i<this->out_depth;i++){
			FP a(0);
			//Vol<FP>* wi = this->filters[i];
			for(int d=0;d<this->num_inputs;d++){
				//Vw * wi
				a += this->in_act->w[d] * this->filters[i]->w[d];
			}
			a+=this->biases->w[i];
			A->w[i] = a;
			//cout << a << " ";
		}
		//cout << " ::: " << V->sx << " " << V->sy << " " << V->depth << endl;

		//cout << "feed e" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		//cout << "feed f" << endl;
		//cout << " ---- " << A->w.size() << endl;
		this->out_act = A;
		//cout << "feed g" << endl;
		//cout << "feed h" << endl;
		return this->out_act;
	}
	void backward(int tmpy=0){
		//Vol<FP>* V = this->in_act;
		Utils<FP> ut;
		this->in_act->dw = ut.zeros(this->in_act->w.size());
		
		for(int i=0;i<this->out_depth;i++){
			//Vol<FP>* tfi = this->filters[i];
			FP chain_grad = this->out_act->dw[i];
			for(int d=0;d<this->num_inputs;d++){
				this->in_act->dw[d] += this->filters[i]->w[d]*chain_grad;
				this->filters[i]->dw[d] += this->in_act->w[d]*chain_grad;
			}
			this->biases->dw[i] += chain_grad;
		}

	}
	
	vector< map<string, vector<FP>* > > getParamsAndGrads(){
		vector< map<string, vector<FP>* > > v;
		for(int i=0;i<this->out_depth;i++){
			map<string, vector<FP>* > m;
			m["params"] = &this->filters[i]->w;
			m["grads"] = &this->filters[i]->dw;
			v.push_back(m);
		}
		map<string,vector<FP>* > m;
		m["params"] = &this->biases->w;
		m["grads"] = &this->biases->dw;
		v.push_back(m);
		return v;
	}
string get_layer_type(){
	return this->layer_type;
}
Vol<FP>* get_in_act(){
	return this->in_act;
}
Vol<FP>* get_out_act(){
	return this->out_act;
	Vol<FP>* oa=this->out_act;
	
	int maxi=0;
	FP maxv=oa->w[0];
	int N=oa->w.size();
	for(int i=1;i<N;i++){
		if( maxv < oa->w[i] ){
			maxv = oa->w[i];
			maxi = i;
		}
	}
Vol<FP>* A = new Vol<FP>(oa->depth,oa->sy,3,FP(0.0));
	for(int i=0;i<N;i++){
		for(int d=0;d<3;d++){
			if(d==2 && i==maxi)
			A->w[i*3+d]=oa->w[i];
			if(i!=maxi)
			A->w[i*3+d]=oa->w[i];
		}
	}

	return A;
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
