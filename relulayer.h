#ifndef RELULAYER_H
#define RELULAYER_H

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
#include <ctime>

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
class ReluLayer : public Layer<FP>{
private:

public:
	int out_depth;
	int out_sx;
	int out_sy;

	int sx;
	int sy;
	int stride;
	int pad;

	int in_depth;
	int in_sx;
	int in_sy;

	int winx;
	int winy;

	string layer_type;

	vector<FP> switchx;
	vector<FP> switchy;

	
	//num_neurons   In:{d,x,y} Conf:{l1_decay,l2_decay}
	ReluLayer(int in_depth,int in_sx,int in_sy){
	
	this->in_depth=in_depth;
	this->in_sx=in_sx;
	this->in_sy=in_sy;
	this->layer_type="relu";
	this->in_act=NULL;
	this->out_act=NULL;

	//cout << "cv 0" << endl;
	//cout << "cv 1" << endl;
	this->out_depth = this->in_depth;
	this->out_sx = this->in_sx;
	this->out_sy = this->in_sy;

	//cout << "cv 2" << endl;
	/*Utils<FP> ut;
	this->switchx = ut.zeros(this->out_sx*this->out_sy*this->out_depth);
	this->switchy = ut.zeros(this->out_sx*this->out_sy*this->out_depth);*/
	//cout << "cv 4" << endl;
	}
	~ReluLayer(){

		//cout << "clearrr3" << endl;
		if(this->in_act != NULL){delete this->in_act;this->in_act=NULL;}
		//cout << "clearrr5" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		//cout << "clearrr4" << endl;
	}

vector<FP> get_all_w(){
	vector<FP> out;

	Vol<FP>* V;
	vector< Vol<FP>* > list;

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

	for(int i=0,q=0;i<slist.size();i++){
		V = list[i];
		for(int j=0;j<slist[i];j++,q++){
			V->w[j]=aw[q];
		}
	}

}
	Vol<FP>* forward(Vol<FP>* V,bool is_training=false){
		//cout << "@1" << endl;
		this->in_act = V;
		//cout << "@2" << endl;
		//cout << "feed b" << endl;
		Vol<FP>* V2 = V->clone();
		int N = V2->w.size();
		//cout << "@3" << endl;
		clock_t begin_time = clock();
		//#pragma omp parallel for
		for(int i=0;i<N;i++){
			if(V2->w[i] < 0) V2->w[i]=0;
		}
		//std::cout << "ReluLayer : " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

		
		//cout << "feed e" << endl;
		if(this->out_act != NULL){delete this->out_act;this->out_act=NULL;}
		//cout << "feed f" << endl;
		this->out_act = V2;
		//cout << "feed g" << endl;
		//cout << "feed h" << endl;
		return this->out_act;
	}
	void backward(int tmpy=0){
		Vol<FP>* V = this->in_act;		
		Vol<FP>* V2 = this->out_act;
		int N = V->w.size();
		Utils<FP> ut;
		V->dw = ut.zeros(V->w.size());
		
		for(int i=0;i<N;i++){
			if(V2->w[i] <= 0) V->dw[i] = 0;
			else V->dw[i] = V2->dw[i];
		}
	}
	
	vector< map<string, vector<FP>* > > getParamsAndGrads(){
		vector< map<string, vector<FP>* > > v;
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
