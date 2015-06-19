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
			//resize(m,m,Size(36,36), 0, 0, INTER_AREA);
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

int main(void)
{
	//calibrateCamera(2);
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
//ConvNet
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
		
    
    FP* pred=new FP[200];
	int i_pred=0;
    
    int k=0;
    
ConvNet<FP>* cnet=new ConvNet<FP>("");
/*

\
input[sx:32,sy:32,depth:3]>conv[sx:4,filters:40,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>conv[sx:4,filters:50,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>conv[sx:4,filters:60,stride:1,pad:2]>relu[]>pool[sx:2,sy:2]\
>fc[num_neurons:10]>softmax[]\

*/

Utils<FP> ut;
FP rp=FP(0);
int saw=0;

string convpath="conv_no2/cnet";
//cnet->save(convpath);
cnet->load(convpath);

    while(true)
    {
		//cvtColor(image, gray_image, CV_BGR2GRAY);
		
		//>pool[sx:5,sy:5]>fc[num_classes:10]
		
//>softmax[num_classes:10]
		
		Vol<FP>* v3 = Vol<FP>::mat_to_vol(vm[k]);
		//static int pp=0;
		//if(pp++==0)
		cnet->forward(v3);
		FP result = ( vl[k] == cnet->getPrediction() )?FP(1.0):FP(0.0);
		//if(result < 0.5)
		{
			cnet->train(v3,vl[k]);
			result = ( vl[k] == cnet->getPrediction() )?FP(1.0):FP(0.0);
			//cout << "	Result " << vl[k] << " : " << cnet->getPrediction() << endl;
		}
		saw++;
		
		//cout << "Result " << vl[k] << cnet->getPrediction() << endl;
	if(saw%100==0){
		
	for(int q=0;q<100;q++){
		int kk=  (int)(  ut.mrand() * (1000)  );
		kk+=(vm.size()-1000);
		Vol<FP>* v4 = Vol<FP>::mat_to_vol(vm[kk]);
		cnet->forward(v4);
		FP result = ( vl[kk] == cnet->getPrediction() )?FP(1.0):FP(0.0);
		pred[i_pred++]= result;
		delete v4;
	}
	
		for(int i=0;i<100;i++){
			rp+=pred[i];
		}
		rp/=100.0;
		
		i_pred = (i_pred >= 100)?0:i_pred;
	
		
	}
	
	if(saw%1000==0){
		cnet->save(convpath);
		cnet->load(convpath);
	}
		cout << k << "/" << vm.size() << " saw : " << saw << "  Correct Percent : " << rp << endl;
		//Vol<FP>* v4 = cnet->forward(v3);
		//Mat convnet = v4->npho_to_mat();
		for(int i=0;i<cnet->net.size()-1;i++){
			
			//cout << cnet->net[i]->get_layer_type() << " " << cnet->net[i]->get_out_act()->sx  << " " << cnet->net[i]->get_out_act()->sy << " "  << cnet->net[i]->get_out_act()->depth << endl;
			Mat inp = cnet->net[i]->get_out_act()->npho_to_mat();
			if(i==cnet->net.size()-2)
			inp = cnet->net[i]->get_out_act()->po_to_mat();
			//cout << " == " << inp.cols << " " << inp.rows << endl;
			resize(inp,inp,Size(512,512), 0, 0, INTER_AREA);
			char numstr[512]; // enough to hold all numbers up to 64-bits
			sprintf(numstr,"%d %s",i,cnet->net[i]->get_layer_type().c_str());
			imshow(numstr , inp );
		}
		
		//if(result > 0.5)
			k=  (int)(  ut.mrand() * (vm.size()-1000)  );
			
		if(k>vm.size()-1)
			k=0;
			
		delete v3;
		//delete v4;

        int key = waitKey(1);
        if(key==27)
            return 0;
    }
    
	delete cnet;
	
	
	
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

    float OpenGLMatrix[] = { 1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1 };
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

    while (true)
    {
        video >> source;
        if (source.empty())
            break;

        imshow("Original", source);

        //foundCorners(&srcImagePoints, source, grayImage);
        //cvPOSIT(positObject, &srcImagePoints[0], FOCAL_LENGTH, criteria, rotation_matrix, translation_vector);
        //createOpenGLMatrixFrom(OpenGLMatrix, rotation_matrix, translation_vector);

        updateWindow("POSIT");
        int keycode=waitKey(33);
		if(keycode>-1){
			cout << keycode << endl;
		}
        if (video.get(CV_CAP_PROP_POS_AVI_RATIO) > 0.99)
            video.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
    }

    setOpenGlDrawCallback("POSIT", NULL, NULL);
    destroyAllWindows();
    cvReleasePOSITObject(&positObject);

    delete[]rotation_matrix;
    delete[]translation_vector;

    return EXIT_SUCCESS;*/
}
