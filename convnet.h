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
		this->net = inet;        
}
~ConvNet(){
	for(int i=0;i<this->net.size();i++){
			delete this->net[i];
			this->net[i]=NULL;
	}
	this->net.clear();
}

void forward(Vol<FP>* Vin,bool is_training=false){
//Vol<FP>* Vin=V->clone();
Vol<FP>* Vout;

		for(int i=0;i<this->net.size();i++){
			//cout << "Cnet : Feed " << this->net[i]->get_layer_type() << " " << i << endl;
			Vin=this->net[i]->forward(Vin,is_training);
			//delete Vin;
			//Vin=Vout;
		}
		//delete Vin;
}

void backward(int y){
	int N = this->net.size();
	this->net[N-1]->backward(y);
	for(int i=N-2;i>=0;i--){
		this->net[i]->backward(y);
	}
}

vector< map<string,vector<FP>* > >  getParamsAndGrads(){
	vector< map<string,vector<FP>* > > resp;
	for(int i=0,k=0,m=0;i<this->net.size();i++){
		vector< map<string,vector<FP>* > > layer_resp = this->net[i]->getParamsAndGrads();
		for(int j=0;j<layer_resp.size();j++){
			//cout << k << " " << this->net[i]->get_layer_type() << m  << "   " << j << "/" << layer_resp.size() << endl;m++;
			resp.push_back(layer_resp[j]);
		}k++;
	}
	return resp;
}

int getPrediction(){
	Layer<FP>* S = net[net.size()-1];
	if( S->layer_type.compare("softmax") == 0 ){
		cout << "getPrediction function assumes softmax as last layer of the net!" << endl;
	}
	
	vector<FP> p=S->get_out_act()->w;
	//cout << "getPrediction : N " << p.size() << endl;
	FP maxv = p[0];
	int maxi = 0;
	for(int i=1;i<p.size();i++){
		if(p[i] > maxv){
			maxv = p[i];
			maxi = i;
		}
	}
	//cout << "Pd : " << maxv << endl;
	return maxi;//return index of the class with highest class prob
}

FP learning_rate=FP(0.01);
FP l1_decay=FP(0.0);
FP l2_decay=FP(0.0001);
int batch_size=10;
string method = "sdg";
//sgd/adagrad/adadelta/windowgrad/netsterov
FP momentum=FP(0.9);
FP ro=FP(0.95);
FP eps=FP(1e-6);
int k=0;
vector<vector<FP> > gsum;//last iteration gradients 
vector<vector<FP> > xsum;//used in adadelta


void train(Vol<FP>* x,int y){
	Utils<FP> ut;

	this->forward(x,true);
	//cout << "Traind :: feed" << endl;
	this->backward(y);
	//cout << "Traind :: back" << endl;
	FP l2_decay_loss=FP(0.0);
	FP l1_decay_loss=FP(0.0);
	this->k++;
	if(this->k % this->batch_size == 0){
		vector< map<string,vector<FP>* > >  pglist = this->getParamsAndGrads();
		if(this->gsum.size() == 0 && ( this->method.compare("sgd") != 0 || this->momentum > 0.0  )){
			for(int i=0;i<pglist.size();i++){
				vector<FP>& tmp = *pglist[i]["params"];
				//cout << "params " << tmp.size() << " " << i << "/" <<pglist.size() << endl;

				this->gsum.push_back(ut.zeros(tmp.size()));
				this->xsum.push_back(ut.zeros(tmp.size()));
				/*if( this->method.compare("adadelta") ){
					this->xsum.push_back(ut.zeros(tmp.size()));
				}
				else{
				}*/
			}
		}
	  
		

		for(int i=0;i<pglist.size();i++){
			map<string,vector<FP>* > pg = pglist[i];
			vector<FP>& p =  *pglist[i]["params"];
			vector<FP>& g =  *pglist[i]["grads"];

			FP l2_decay_mul = FP(1.0);
			FP l1_decay_mul = FP(1.0);

			FP l2_decay = this->l2_decay * l2_decay_mul;
			FP l1_decay = this->l1_decay * l1_decay_mul;

			int plen = p.size();
			for(int j=0;j<plen;j++){
				//cout << " #1 " << this->gsum.size() << " " << this->xsum.size() << " " << pglist.size() << " " << plen << endl;
				l2_decay_loss += l2_decay*p[j]*p[j]/2;
				l1_decay_loss += l1_decay*abs(p[j]);
				//cout << " #2 " << endl;
				FP l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
				FP l2grad = l2_decay * (p[j]);
				FP gij = FP(l2grad + l1grad + g[j]) / this->batch_size;
				//cout << " #3 " << gij << endl;
				vector<FP> gsumi = this->gsum[i];
				vector<FP> xsumi = this->xsum[i];
				//cout << " #4 " << endl;
				if( this->method.compare("adagrad") == 0 ){
					//adagrad update
					gsumi[j] = gsumi[j] + gij * gij;
					FP dx = - this->learning_rate / sqrt(gsumi[j] + this->eps) * gij;
					p[j] += dx;//cout << " dx : " << dx << endl;
				} else if( this->method.compare("windowgrad") == 0 ){
					gsumi[j] = this->ro * gsumi[j] + (1-this->ro)*gij*gij;
					FP dx = - this->learning_rate / sqrt(gsumi[j] + this->eps) * gij;
					p[j] += dx;//cout << " dx : " << dx << endl;
				} else if( this->method.compare("adadelta") == 0 ){
					gsumi[j] = this->ro * gsumi[j] + (1-this->ro) * gij * gij;
					FP dx = -sqrt( (xsumi[j] + this->eps)/(gsumi[j] + this->eps) ) * gij;
					xsumi[j] = this->ro * xsumi[j] + (1-this->ro) * dx * dx;
					p[j] += dx;//cout << " dx : " << dx << endl;
				} else if( this->method.compare("nesterov") == 0 ){
					FP dx = gsumi[j];
					gsumi[j] = gsumi[j] * this->momentum + this->learning_rate * gij;
					dx = this->momentum * dx - (1.0 + this->momentum) * gsumi[j];
					p[j] += dx;//cout << " dx : " << dx << endl;
				} else {
					if(this->momentum > 0.0){
						FP dx = this->momentum * gsumi[j] - this->learning_rate * gij;
						gsumi[j] = dx;
						p[j] += dx;//cout << " dx : " << dx << endl;
					}
					else{
						p[j] += -this->learning_rate * gij;
					}
				}
				g[j]=0.0;
			}
		}
	}
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
