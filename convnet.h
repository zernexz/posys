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
ConvNet(){
		vector<Layer<float>* > net;
		string sugar("input[sx:640,sy:480,depth:3]->conv[sx:100,sy:100]");
		
		{
			ConvLayer<float>* cvl=new ConvLayer<float>(3,5,5,3,640,480);
			cout << "1" << endl;
			//Vol<float>* v4 = cvl->forward(v3);
			net.push_back(cvl);
		}
        
        
        for(int i=0;i<net.size();i++){
			delete net[i];
			net[i]=NULL;
		}
        
}
~ConvNet(){
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
