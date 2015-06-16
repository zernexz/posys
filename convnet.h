#ifndef CONVNET_H
#define CONVNET_H

#include "utils.h"
#include "vol.h"
#include "convlayer.h"
#include "poollayer.h"
#include "relulayer.h"
#include "softmaxlayer.h"
#include "convnet.h"

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
class ConvNet{
private:

public:
vector<Layer<FP>* > net;
ConvNet(vector<Layer<FP>* >  inet){
		net = inet;        
}
~ConvNet(){
	for(int i=0;i<net.size();i++){
			delete net[i];
			net[i]=NULL;
	}
	net.clear();
}

Vol<FP>* forward(Vol<FP>* Vin,bool is_training=false){
		for(int i=0;i<net.size();i++){
			Vin=net[i]->forward(Vin,is_training);
		}
		return Vin;
}

void backward(int y){
	int N = net.size();
	net[N-1]->backward(y);
	for(int i=N-2;i>=0;i--){
		net[i]->backward(y);
	}
}

int getPrediction(){
	Layer<FP>* S = net[net.size()-1];
	if( S->layer_type.compare("softmax") == 0 ){
		cout << "getPrediction function assumes softmax as last layer of the net!" << endl;
	}

	vector<FP> p=S->out_act->w;
	FP maxv = p[0];
	int maxi = 0;
	for(int i=1;i<p.size();i++){
		if(p[i] > maxv){
			maxv = p[i];
			maxi = i;
		}
	}
	return maxi;//return index of the class with highest class prob
}

FP learning_rate=FP(0.01);
FP l1_decay=FP(0.0);
FP l2_decay=FP(0.0);
int batch_size=1;
string method = "sgd";
//sgd/adagrad/adadelta/windowgrad/netsterov
FP momentum=FP(0.9);
FP ro=FP(0.95);
FP eps=FP(1e-6);
int k=0;
vector<FP> gsum;//last iteration gradients 
vector<FP> xsum;//used in adadelta


void train(Vol<FP>* x,int y){
	this->forward(x,true);
	this->backward(y);
	FP l2_decay_loss=FP(0.0);
	
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
