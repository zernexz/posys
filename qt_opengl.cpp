// Yannick Verdie 2010
// --- Please read help() below: ---

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/compat.hpp>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "picojson/picojson.h"

#include "camera.h"
#include "memdump.h"

#include "utils.h"
#include "vol.h"

#include "inputlayer.h"
#include "convlayer.h"
#include "poollayer.h"
#include "relulayer.h"
#include "fullyconnlayer.h"
#include "softmaxlayer.h"
#include "convnet.h"


#define CAMERA_INTR 0
#define CAMERA_DIST 1

#define CAMERA -1


using namespace std;
using namespace cv;
using namespace CERN;

static void help()
{
    cout << "This demo demonstrates the use of the Qt enhanced version of the highgui GUI interface\n"
            "and dang if it doesn't throw in the use of of the POSIT 3D tracking algorithm too\n"
            "It works off of the video: cube4.avi\n"
            "Using OpenCV version " << CV_VERSION << "\n\n"

            " 1) This demo is mainly based on work from Javier Barandiaran Martirena\n"
            "    See this page http://code.opencv.org/projects/opencv/wiki/Posit.\n"
            " 2) This is a demo to illustrate how to use **OpenGL Callback**.\n"
            " 3) You need Qt binding to compile this sample with OpenGL support enabled.\n"
            " 4) The features' detection is very basic and could highly be improved\n"
            "    (basic thresholding tuned for the specific video) but 2).\n"
            " 5) Thanks to Google Summer of Code 2010 for supporting this work!\n" << endl;
}

#define FOCAL_LENGTH 600
#define CUBE_SIZE 0.5

static void renderCube(float size)
{
    glBegin(GL_QUADS);
    // Front Face
    glNormal3f( 0.0f, 0.0f, 1.0f);
    glVertex3f( 0.0f,  0.0f,  0.0f);
    glVertex3f( size,  0.0f,  0.0f);
    glVertex3f( size,  size,  0.0f);
    glVertex3f( 0.0f,  size,  0.0f);
    // Back Face
    glNormal3f( 0.0f, 0.0f,-1.0f);
    glVertex3f( 0.0f,  0.0f, size);
    glVertex3f( 0.0f,  size, size);
    glVertex3f( size,  size, size);
    glVertex3f( size,  0.0f, size);
    // Top Face
    glNormal3f( 0.0f, 1.0f, 0.0f);
    glVertex3f( 0.0f,  size,  0.0f);
    glVertex3f( size,  size,  0.0f);
    glVertex3f( size,  size, size);
    glVertex3f( 0.0f,  size, size);
    // Bottom Face
    glNormal3f( 0.0f,-1.0f, 0.0f);
    glVertex3f( 0.0f,  0.0f,  0.0f);
    glVertex3f( 0.0f,  0.0f, size);
    glVertex3f( size,  0.0f, size);
    glVertex3f( size,  0.0f,  0.0f);
    // Right face
    glNormal3f( 1.0f, 0.0f, 0.0f);
    glVertex3f( size,  0.0f, 0.0f);
    glVertex3f( size,  0.0f, size);
    glVertex3f( size,  size, size);
    glVertex3f( size,  size, 0.0f);
    // Left Face
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f( 0.0f,  0.0f, 0.0f);
    glVertex3f( 0.0f,  size, 0.0f);
    glVertex3f( 0.0f,  size, size);
    glVertex3f( 0.0f,  0.0f, size);
    glEnd();
}

static void on_opengl(void* param)
{
    //Draw the object with the estimated pose
    glLoadIdentity();
    glScalef( 1.0f, 1.0f, -1.0f);
    glMultMatrixf( (float*)param );
    glEnable( GL_LIGHTING );
    glEnable( GL_LIGHT0 );
    glEnable( GL_BLEND );
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    renderCube( CUBE_SIZE );
    glDisable(GL_BLEND);
    glDisable( GL_LIGHTING );
}

static void initPOSIT(std::vector<CvPoint3D32f> * modelPoints)
{
    // Create the model pointss
    modelPoints->push_back(cvPoint3D32f(0.0f, 0.0f, 0.0f)); // The first must be (0, 0, 0)
    modelPoints->push_back(cvPoint3D32f(0.0f, 0.0f, CUBE_SIZE));
    modelPoints->push_back(cvPoint3D32f(CUBE_SIZE, 0.0f, 0.0f));
    modelPoints->push_back(cvPoint3D32f(0.0f, CUBE_SIZE, 0.0f));
}

static void foundCorners(vector<CvPoint2D32f> * srcImagePoints, const Mat & source, Mat & grayImage)
{
    cvtColor(source, grayImage, COLOR_RGB2GRAY);
    GaussianBlur(grayImage, grayImage, Size(11, 11), 0, 0);
    normalize(grayImage, grayImage, 0, 255, NORM_MINMAX);
    threshold(grayImage, grayImage, 26, 255, THRESH_BINARY_INV); //25

    Mat MgrayImage = grayImage;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(MgrayImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    Point p;
    vector<CvPoint2D32f> srcImagePoints_temp(4, cvPoint2D32f(0, 0));

    if (contours.size() == srcImagePoints_temp.size())
    {
        for (size_t i = 0; i < contours.size(); i++ )
        {
            p.x = p.y = 0;

            for (size_t j = 0 ; j < contours[i].size(); j++)
                p += contours[i][j];

            srcImagePoints_temp.at(i) = cvPoint2D32f(float(p.x) / contours[i].size(), float(p.y) / contours[i].size());
        }

        // Need to keep the same order
        // > y = 0
        // > x = 1
        // < x = 2
        // < y = 3

        // get point 0;
        size_t index = 0;
        for (size_t i = 1 ; i<srcImagePoints_temp.size(); i++)
            if (srcImagePoints_temp.at(i).y > srcImagePoints_temp.at(index).y)
                index = i;
        srcImagePoints->at(0) = srcImagePoints_temp.at(index);

        // get point 1;
        index = 0;
        for (size_t i = 1 ; i<srcImagePoints_temp.size(); i++)
            if (srcImagePoints_temp.at(i).x > srcImagePoints_temp.at(index).x)
                index = i;
        srcImagePoints->at(1) = srcImagePoints_temp.at(index);

        // get point 2;
        index = 0;
        for (size_t i = 1 ; i<srcImagePoints_temp.size(); i++)
            if (srcImagePoints_temp.at(i).x < srcImagePoints_temp.at(index).x)
                index = i;
        srcImagePoints->at(2) = srcImagePoints_temp.at(index);

        // get point 3;
        index = 0;
        for (size_t i = 1 ; i<srcImagePoints_temp.size(); i++ )
            if (srcImagePoints_temp.at(i).y < srcImagePoints_temp.at(index).y)
                index = i;
        srcImagePoints->at(3) = srcImagePoints_temp.at(index);

        Mat Msource = source;
        stringstream ss;
        for (size_t i = 0; i<srcImagePoints_temp.size(); i++ )
        {
            ss << i;
            circle(Msource, srcImagePoints->at(i), 5, Scalar(0, 0, 255));
            putText(Msource, ss.str(), srcImagePoints->at(i), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
            ss.str("");

            // new coordinate system in the middle of the frame and reversed (camera coordinate system)
            srcImagePoints->at(i) = cvPoint2D32f(srcImagePoints_temp.at(i).x - source.cols / 2,
                                                 source.rows / 2 - srcImagePoints_temp.at(i).y);
        }
    }
}

static void createOpenGLMatrixFrom(float * posePOSIT, const CvMatr32f & rotationMatrix,
                                   const CvVect32f & translationVector)
{
    // coordinate system returned is relative to the first 3D input point
    for (int f = 0; f < 3; f++)
        for (int c = 0; c < 3; c++)
            posePOSIT[c * 4 + f] = rotationMatrix[f * 3 + c]; // transposed

    posePOSIT[3] = translationVector[0];
    posePOSIT[7] = translationVector[1];
    posePOSIT[11] = translationVector[2];
    posePOSIT[12] = 0.0f;
    posePOSIT[13] = 0.0f;
    posePOSIT[14] = 0.0f;
    posePOSIT[15] = 1.0f;
}


static void printMat(const Mat& M){	
	
	for(int i=0;i<M.rows;i++){
		const float* Mi = M.ptr<float>(i);
		for(int j=0;j<M.cols;j++){
			cout << "(" << i << "," << j << ") : " << Mi[j] << " ";
		}
		cout << endl;
	}
	
}


bool write_file_binary (std::string const & filename, 
  char const * data, size_t const bytes)
{
  std::ofstream b_stream(filename.c_str(), 
    std::fstream::out | std::fstream::binary);
  if (b_stream)
  {
    b_stream.write(data, bytes);
    return (b_stream.good());
  }
  return false;
}


char* read_file_binary (std::string const & filename, size_t const bytes){
	char* data=new char[bytes];
	
	std::ifstream b_stream(filename.c_str(), 
    std::fstream::out | std::fstream::binary);
  if (b_stream)
  {
    b_stream.read(data, bytes);
    return data;
  }
  return NULL;
}

static void writeMat(const char * path ,const Mat& M){
	
	long size=M.rows*M.cols;
	float * buffer = new float[size];
	
	for(int i=0;i<M.rows;i++){
		const float* Mi = M.ptr<float>(i);
		for(int j=0;j<M.cols;j++){
			buffer[i*M.cols + j]=Mi[j];
		}
		cout << endl;
	}
	
	write_file_binary(path, 
    reinterpret_cast<char const *>(buffer), 
    sizeof(float)*size);
    
	delete[] buffer;
}

static void readMat(const char * path ,Mat& M){
	
	long size=M.rows*M.cols;
	float * buffer = reinterpret_cast<float *>(read_file_binary(path,sizeof(float)*size));
	
	for(int i=0;i<M.rows;i++){
		float* Mi = M.ptr<float>(i);
		for(int j=0;j<M.cols;j++){
			Mi[j]=buffer[i*M.cols + j];
		}
		cout << endl;
	}
    
	delete[] buffer;
}

static char * getConfigPath(int type,int no){
	char winstr[512]="/home/ryouma/opencv-2.4.9/samples/cpp/posys/";
	char numstr[512]; // enough to hold all numbers up to 64-bits
    sprintf(numstr,"conf_%d_%d.bin",type,no);
    strcat( winstr ,numstr);
    return winstr;
}

static char * getTrainingSetPath(int type,int no){
	return "/home/ryouma/Desktop/cow.png";
	
	char winstr[512]="/home/ryouma/opencv-2.4.9/samples/cpp/posys/";
	char numstr[512]; // enough to hold all numbers up to 64-bits
    sprintf(numstr,"conf_%d_%d.bin",type,no);
    strcat( winstr ,numstr);
    return winstr;
}


static void calibrateCamera(int CamNum){
	int numBoards = 40;
    int numCornersHor = 4;
    int numCornersVer = 3;
    
    printf("Enter number of corners along width: ");
    scanf("%d", &numCornersHor);

    printf("Enter number of corners along height: ");
    scanf("%d", &numCornersVer);

    printf("Enter number of boards: ");
    scanf("%d", &numBoards);
    
    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor, numCornersVer);
    VideoCapture capture = VideoCapture(CamNum);
    
    Camera cam;
    
    vector<Point2f> corners;
    int successes=0;
    
    Mat gray_image;
    capture >> cam.image;
    
    vector<Point3f> obj;
    for(int j=0;j<numSquares;j++)
        obj.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f));
        
    while(successes<numBoards)
    {
		cvtColor(cam.image, gray_image, CV_BGR2GRAY);
		bool found = findChessboardCorners(cam.image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if(found)
        {
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray_image, board_sz, corners, found);
        }
        imshow("win1", cam.image);
        
        char winstr[21]="win2"; // enough to hold all numbers up to 64-bits
        
        //char numstr[21]; // enough to hold all numbers up to 64-bits
        //sprintf(numstr,"%d",successes);
		//strcat( winstr ,numstr);

        imshow( winstr , gray_image);

        capture >> cam.image;
        int key = waitKey(1);
        
        if(key==27)

            return ;

        if(key==' ' && found!=0)
        {
            cam.image_points.push_back(corners);
            cam.object_points.push_back(obj);

            cout << "Snap stored! pic" << successes+1 << "/" << numBoards << endl;

            successes++;

            if(successes>=numBoards)
                break;
        }
	}
	//capture.release();
	
    
    cam.intrinsic.ptr<float>(0)[0] = 1;
    cam.intrinsic.ptr<float>(1)[1] = 1;
    
    
    
    
    
    calibrateCamera(cam.object_points, cam.image_points, cam.image.size(), cam.intrinsic, cam.distCoeffs, cam.rvecs, cam.tvecs);
    
    {
		std::ofstream ofs(getConfigPath(CAMERA,CamNum));
        boost::archive::text_oarchive oa(ofs);
        oa << cam;
	}
	
    cout << cam.distCoeffs.channels() << endl;
    cout << cam.intrinsic.channels() << endl;
    
    //Build DistCoeffs & Intrinsic JSON
    cout << "Intrinsic" << endl;
	printMat(cam.intrinsic);
	writeMat(getConfigPath(CAMERA_INTR,CamNum),cam.intrinsic);
	
	cout << "DistCoeffs" << endl;
	printMat(cam.distCoeffs);
	writeMat(getConfigPath(CAMERA_DIST,CamNum),cam.distCoeffs);
	
	
	
	
	cout << "Intrinsic" << endl;
	readMat(getConfigPath(CAMERA_INTR,CamNum),cam.intrinsic);
	printMat(cam.intrinsic);
	
	cout << "DistCoeffs" << endl;
	readMat(getConfigPath(CAMERA_DIST,CamNum),cam.distCoeffs);
    printMat(cam.distCoeffs);
    
    
    Mat imageUndistorted;
    while(1)
    {
        capture >> cam.image;
        undistort(cam.image, imageUndistorted, cam.intrinsic,cam.distCoeffs);

        imshow("win1", cam.image);
        imshow("win2", imageUndistorted);
        waitKey(1);
    }
	 
	capture.release();
	//Write JSON Camera Config to File
	
	
    
}

static void unDist(int CamNum){
	VideoCapture capture = VideoCapture(CamNum);
	 
    
	Camera cam;
	
    {
        // create and open an archive for input
        std::ifstream ifs(getConfigPath(CAMERA,CamNum));
        boost::archive::text_iarchive ia(ifs);
        // read class state from archive
        ia >> cam;
        // archive and stream closed when destructors are called
    }
    
    //calibrateCamera(cam.object_points, cam.image_points, cam.image.size(), cam.intrinsic, cam.distCoeffs, cam.rvecs, cam.tvecs);
    Mat image;
    Mat imageUndistorted;
    while(1)
    {
        capture >> image;
        undistort(image, imageUndistorted, cam.intrinsic, cam.distCoeffs);

        imshow("win1", cam.image);
        imshow("win2", imageUndistorted);
        waitKey(1);
    }
	 
	capture.release();
}
typedef double FP;


vector<Mat> load_mnist(){
	
	
    vector<Mat> vm;
    for(int no;no<51;no++){
		char numstr[512]; // enough to hold all numbers up to 64-bits
		sprintf(numstr,"cifar10_batch_%d.png",no);
		Mat mnist = imread(numstr, CV_LOAD_IMAGE_COLOR);
		for(int i=0;i<mnist.rows;i++){
			Mat m(32,32, CV_8UC3);
			for(int j=0;j<32;j++){
				for(int k=0;k<32;k++){
					for(int d=0;d<3;d++){
						
						m.data[(j*32+k)*3+d]= (uint8_t) mnist.data[((i*1024)+(j*32+k))*3+d];
						
					}
				}
			}
			vm.push_back(m);
		}
	}
	
	return vm;
}

vector<int> load_mnist_label(){
	string line;
	int val;
	vector<int> vl;
	ifstream myfile ("label");
	  if (myfile.is_open())
	  {
		while ( getline (myfile,line,',') )
		{
			if(stringstream(line)>>val){
				vl.push_back(val);
			}
		}
		myfile.close();
	  }
	  return vl;
}
vector<Layer<FP>* > new_convnet(){
	//input[sx:3200,sy:3200,depth:3]>conv[sx:5,filters:16,stride:1,pad:2]>relu>pool[sx:4,sy:4]>fc[num_classes:10]
	string sugar("input[sx:32,sy:32,depth:3]>conv[sx:5,filters:16,stride:1,pad:2]>relu>pool[sx:5,sy:5]>fc[num_classes:10]");
	
	vector<Layer<FP>* > vl;
	
	string line;
	string type;
	string conf;
	string att;
	int val;
	stringstream ss;
	ss << sugar;
	
	int po_sx;
	int po_sy;
	int po_depth;
	while ( getline (ss,line,'>') )
		{
			if(stringstream(line)>>conf){
				cout << "conf:" << conf << endl;
				stringstream sc;
				string tmp;
				sc << conf;
				
				string ltype;
				vector<string> attrs;
				vector<int> vals;
							
				int co=0;
				while ( getline (sc,tmp,'[') )
				{
					if(stringstream(tmp)>>type){
						if(co==0){//Type of Layer
							//cout << "Layer:" << type << endl;
							ltype=type;
						}
						else{//Attributes
							//cout << "Attr " <<  type << endl;
							stringstream cc;
							cc << type;
							string tm,tmm;
							
							while ( getline (cc,tm,',') )
							{
								if(stringstream(tm)>>tmm){
									
									stringstream scc;
									scc << tmm;
									string tp;
									int cn=0;
									while ( getline (scc,tmm,':') )
									{
										if(cn==0&&stringstream(tmm)>>tp){
											
										}
										else if(cn==1&&stringstream(tmm)>>val){
											//cout << "Attr " << tp << ":" << val << endl;
											attrs.push_back(tp);
											vals.push_back(val);
										}
										cn++;
									}
								}
							}
							//Add new Layer here
							{
								cout << "* " << ltype << endl;
								int sx=0;
								int sy=0;
								int depth=0;
								int pad=0;
								int stride=0;
								int filters=0;
								int num_neurons=0;
/*
var layer_defs, net, trainer;
var t = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});\n\
* InputLayer(int out_depth,int out_sx,int out_sy)
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});\n\
* ConvLayer(int out_depth,int sx,int sy,int in_depth,int in_sx,int in_sy,int stride=1,int pad=0,FP l1_decay_mul=FP(0),FP l2_decay_mul=FP(1),FP bias_pref=FP(0))
* ReluLayer(int in_depth,int in_sx,int in_sy)
layer_defs.push({type:'pool', sx:2, stride:2});\n\
* PoolLayer(int sx,int sy,int in_depth,int in_sx,int in_sy,int stride=2,int pad=0)
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});\n\
layer_defs.push({type:'pool', sx:2, stride:2});\n\
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});\n\
layer_defs.push({type:'pool', sx:2, stride:2});\n\
layer_defs.push({type:'softmax', num_classes:10});\n\
* SoftmaxLayer(int in_depth,int in_sx,int in_sy)
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});\n\
";

 * */
 
								for(int i=0;i<attrs.size();i++){
									cout << attrs[i] << " " << vals[i] << endl;
									if(attrs[i].compare("sx") == 0){
										sx=vals[i];
									}
									if(attrs[i].compare("sy") == 0){
										sy=vals[i];
									}
									if(attrs[i].compare("depth") == 0){
										depth=vals[i];
									}
									if(attrs[i].compare("stride") == 0){
										stride=vals[i];
									}
									if(attrs[i].compare("pad") == 0){
										pad=vals[i];
									}
									if(attrs[i].compare("filters") == 0){
										filters=vals[i];
									}
									if(attrs[i].compare("num_neurons") == 0){
										num_neurons=vals[i];
									}
								}
							
								cout << " >> " << ltype << " " << sx << " " << sy << " " << depth << " " << stride << " " << pad << " " << filters << endl;
								if(ltype.compare("input") == 0){
									InputLayer<FP>* il=new InputLayer<FP>(sx,sy,depth);	
									vl.push_back(il);
									po_sx=sx;
									po_sy=sy;
									po_depth=depth;
								}
								else{
									if(ltype.compare("conv") == 0){
										ConvLayer<FP>* cl=new ConvLayer<FP>(filters,sx,sx,po_depth,po_sx,po_sy);
										vl.push_back(cl);
										po_sx=cl->out_sx;
										po_sy=cl->out_sy;
										po_depth=cl->out_depth;
									}
									if(ltype.compare("relu") == 0){
										ReluLayer<FP>* rl=new ReluLayer<FP>(po_depth,po_sx,po_sy);
										vl.push_back(rl);
									}
									if(ltype.compare("pool") == 0){
										cout << sx << " " << sy << " " << po_depth << " " << po_sx << " " << po_sy << endl;
										PoolLayer<FP>* pl=new PoolLayer<FP>(sx,sy,po_depth,po_sx,po_sy);
										//PoolLayer<FP>* pl=new PoolLayer<FP>(5,5,16,1000,1000);
										vl.push_back(pl);
									}
									if(ltype.compare("softmax") == 0){
										SoftmaxLayer<FP>* sml=new SoftmaxLayer<FP>(po_depth,po_sx,po_sy);
										vl.push_back(sml);
									}
									if(ltype.compare("fc") == 0){
										cout << " ** " << num_neurons << " " << po_depth << " " << po_sx << " " << po_sy << endl;
										FullyConnLayer<FP>* fc=new FullyConnLayer<FP>(num_neurons,po_depth,po_sx,po_sy);
										vl.push_back(fc);
										po_sx=fc->out_sx;
										po_sy=fc->out_sy;
										po_depth=fc->out_depth;
									}
								}
							}
						}
					}
					co++;
				}
			}
			
		}
	return vl;
}
int main(void)
{
	//calibrateCamera(0);
	//unDist(0);
	
	
	//Test Utils
	/*
	Utils<FP> ut;
	
	cout << "mrand" << ut.mrand() << endl;
	cout << "randf" << ut.randf(-1,1) << endl;
	cout << "randi" << ut.randi(-1,1) << endl;
	cout << "randn" << ut.randn(-1,1) << endl;
	cout << "guassRandom" << ut.gaussRandom() << endl;
	vector<FP> vf=ut.zeros(5);
	cout << vf.size() << endl;
	for(int i=0;i<vf.size();i++){
		cout << vf[i] << " ";
	}
	cout << endl;
	
	cout << "contains 0 : " << ut.arrContains(vf,0) << endl;
	cout << "contains 1 : " << ut.arrContains(vf,1) << endl;
	
	vector<FP> uvf=ut.arrUnique(vf);
	cout << uvf.size() << endl;
	for(int i=0;i<uvf.size();i++){
		cout << uvf[i] << " ";
	}
	cout << endl;
	
	
	for(int i=0;i<vf.size();i++){
		vf[i]=i;
		cout << vf[i] << " ";
	}
	cout << endl;
	
	map<string,FP> m=ut.maxmin(vf);
	cout << m["maxi"] << endl;
	cout << m["maxv"] << endl;
	cout << m["mini"] << endl;
	cout << m["minv"] << endl;
	cout << m["dv"] << endl;
	
	vector<int> vp=ut.randperm(10);
	for(int i=0;i<10;i++)
		cout << vp[i] << " ";
	cout << endl;
	*/
	
	//Test Vol
	Vol<float>* v1=new Vol<float>(5,4,3);
	for(int i=0;i<5*4*3;i++){
		cout << v1->w[i] << " ";
	}
	cout << endl;
	
	delete v1;
	
	Vol<float>* v2=new Vol<float>(5,4,3,-1.3f);
	for(int i=0;i<5*4*3;i++){
		cout << v2->w[i] << " ";
	}
	cout << endl;
	
	delete v2;
	
	Mat image,gray_image;
    image = imread(getTrainingSetPath(0,0), CV_LOAD_IMAGE_COLOR);
    
    
    
    
    //Vol mat_to_img
    
    
	vector<Mat> vm = load_mnist();
	vector<int> vl = load_mnist_label();
		
    
    
    
    
    while(true)
    {
		cvtColor(image, gray_image, CV_BGR2GRAY);
		
		Vol<float>* v5 = new Vol<float>(2,2,2);
		
		Vol<float>* v3 = Vol<float>::mat_to_vol(image);
		
		//Vol<float>* v4 = Vol<float>::augment(*v3,300,0,0,true);
		//cout << "0" << endl;
		ConvLayer<float>* cvl=new ConvLayer<float>(3,9,9,3,v3->sx,v3->sy);
		//cout << "1" << endl;
		Vol<float>* v4 = cvl->forward(v3);
		

		
		//cout << "2" << endl;
		Mat convimg = v4->npho_to_mat();
		//cout << "3" << endl;
		//cout << "4" << endl;
		
		
		for(int i=0;i<v5->sx*v5->sy*v5->depth;i++){
			v5->w[i]=i;
		}
		
		
		
		SoftmaxLayer<float>* sml=new SoftmaxLayer<float>(2,2,2);
		Vol<float>* smr=sml->forward(v5);
		
		//cout << smr->sx << " " << smr->sy << " " << smr->depth << endl;
		for(int i=0;i<smr->sx;i++){
			for(int j=0;j<smr->sy;j++){
				for(int k=0;k<smr->depth;k++){
					//cout << smr->get(i,j,k) << " ";
				}
			}
		}
		//cout << endl;
		
		
		
		PoolLayer<float>* pl=new PoolLayer<float>(5,5,16,9000,9000);
		//Vol<float>* v6=pl->forward(v4);
		//Mat poolimg=v6->npho_to_mat();
		
		//cout << v3->sx << " x " << v3->sy << endl;
		//cout << v4->sx << " x " << v4->sy << endl;
		//cout << v6->sx << " x " << v6->sy << endl;
		
		//ReluLayer<float>* rl=new ReluLayer<float>(v6->depth,v6->sx,v6->sy);
		//Vol<float>* v7=rl->forward(v6);
		//Mat reluimg = v7->po_to_mat();
		
        imshow("win1" , image);
        imshow("win2" , gray_image);
        imshow("ConvLayer" , convimg);
        //imshow("PoolLayer" , poolimg);
        
        
        //imshow("ReluLayer" , reluimg);
        
        //cout << "******* " << vm.size() << " " << vl.size() << endl;
        char numstr[512]; // enough to hold all numbers up to 64-bits
        for(int i=29995;i<29995+10;i++){
			
			sprintf(numstr,"MNIST %d - %d",i,vl[i]);
			imshow(numstr , vm[i]);
		}
		
		
		
        //cout << "5" << endl;
        
        
        
        
        //delete rl;
        //delete v7;
        
        
        delete pl;
        //delete v6;
        
		delete smr;
		delete sml;
        
       
        delete v3;
		delete v5;
		delete v4;
		delete cvl;
		//cout << "6" << endl;
		
		
		vector<Layer<FP>*> vl=new_convnet();
		for(int i=0;i<vl.size();i++){
			delete vl[i];
		}
		
        int key = waitKey(1);
        if(key==27)
            return 0;
    }
    
	
	/*
	int numBoards = 40;
    int numCornersHor = 4;
    int numCornersVer = 3;
    
    picojson::value v;
    v.value();
    
    printf("Enter number of corners along width: ");
    scanf("%d", &numCornersHor);

    printf("Enter number of corners along height: ");
    scanf("%d", &numCornersVer);

    printf("Enter number of boards: ");
    scanf("%d", &numBoards);
    
    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor, numCornersVer);
    VideoCapture capture = VideoCapture(0);
	vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > image_points;
    vector<Point2f> corners;
    int successes=0;
    
    Mat image;
    Mat gray_image;
    capture >> image;
    
    vector<Point3f> obj;
    for(int j=0;j<numSquares;j++)
        obj.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f));
        
    while(successes<numBoards)
    {
		cvtColor(image, gray_image, CV_BGR2GRAY);
		bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if(found)
        {
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray_image, board_sz, corners, found);
        }
        imshow("win1", image);
        
        char winstr[21]="win2"; // enough to hold all numbers up to 64-bits
        char numstr[21]; // enough to hold all numbers up to 64-bits
        sprintf(numstr,"%d",successes);
		strcat( winstr ,numstr);

        imshow( winstr , gray_image);

        capture >> image;
        int key = waitKey(1);
        
        if(key==27)

            return 0;

        if(key==' ' && found!=0)
        {
            image_points.push_back(corners);
            object_points.push_back(obj);

            cout << "Snap stored!" << successes;

            successes++;

            if(successes>=numBoards)
                break;
        }
	}
	
	Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
    
    
    Mat imageUndistorted;
    while(1)
    {
        capture >> image;
        undistort(image, imageUndistorted, intrinsic, distCoeffs);

        imshow("win1", image);
        imshow("win2", imageUndistorted);
        waitKey(1);
    }
	capture.release();

    return 0;
	*/
	
	/*
    help();

    string fileName = "cube4.avi";
    VideoCapture video(fileName);
    if (!video.isOpened())
    {
        cerr << "Video file " << fileName << " could not be opened" << endl;
        return EXIT_FAILURE;
    }

    Mat source, grayImage;
    video >> source;

    namedWindow("Original", WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
    namedWindow("POSIT", WINDOW_OPENGL | CV_WINDOW_FREERATIO);
    resizeWindow("POSIT", source.cols, source.rows);

    displayOverlay("POSIT", "We lost the 4 corners' detection quite often (the red circles disappear).\n"
                   "This demo is only to illustrate how to use OpenGL callback.\n"
                   " -- Press ESC to exit.", 10000);

    float OpenGLMatrix[] = { 0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0 };
    setOpenGlContext("POSIT");
    setOpenGlDrawCallback("POSIT", on_opengl, OpenGLMatrix);

    vector<CvPoint3D32f> modelPoints;
    initPOSIT(&modelPoints);

    // Create the POSIT object with the model points
    CvPOSITObject* positObject = cvCreatePOSITObject( &modelPoints[0], (int)modelPoints.size());

    CvMatr32f rotation_matrix = new float[9];
    CvVect32f translation_vector = new float[3];
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1e-4f);
    vector<CvPoint2D32f> srcImagePoints(4, cvPoint2D32f(0, 0));

    while (waitKey(33) != 27)
    {
        video >> source;
        if (source.empty())
            break;

        imshow("Original", source);

        foundCorners(&srcImagePoints, source, grayImage);
        cvPOSIT(positObject, &srcImagePoints[0], FOCAL_LENGTH, criteria, rotation_matrix, translation_vector);
        createOpenGLMatrixFrom(OpenGLMatrix, rotation_matrix, translation_vector);

        updateWindow("POSIT");

        if (video.get(CV_CAP_PROP_POS_AVI_RATIO) > 0.99)
            video.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
    }

    setOpenGlDrawCallback("POSIT", NULL, NULL);
    destroyAllWindows();
    cvReleasePOSITObject(&positObject);

    delete[]rotation_matrix;
    delete[]translation_vector;
*/
    return EXIT_SUCCESS;
}
