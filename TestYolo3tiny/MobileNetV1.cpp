#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/tracking.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include <cmath>


using namespace cv;
using namespace std;

const size_t width = 300;
const size_t height = 300;

Mat frame;
int m_x1, m_x2, m_y1, m_y2;


// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 352;        // Width of network's input image
int inpHeight = 352;       // Height of network's input image


vector<String> getOutputsNames(const cv::dnn::Net& net);
void postprocess(Mat& frame, const vector<Mat>& outs);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

std::vector<std::string> Labels;

cv::VideoCapture cap("libcamerasrc ! video/x-raw,  width=(int)1280, height=(int)1280, framerate=(fraction)30/1 "
                     "! videoconvert ! videoscale ! video/x-raw, width=(int)352, height=(int)352 ! appsink", cv::CAP_GSTREAMER);



static bool getFileContent(std::string fileName)
{

	// Open the File
	std::ifstream in(fileName.c_str());
	// Check if object is valid
	if(!in.is_open()) return false;

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size()>0) Labels.push_back(str);
	}
	// Close The File
	in.close();
	return true;
}


int main(int argc,char ** argv)
{
    float f;
    float FPS[16];
    int i, Fcnt=0;

    sleep(3);
    std::cout << "Using pipeline: libcamerasrc ! video/x-raw,  width=(int)1280, height=(int)1280, framerate=(fraction)30/1 "
                     "! videoconvert ! videoscale ! video/x-raw, width=(int)352, height=(int)352 ! appsink\n\n";

	// Get the names
	bool result = getFileContent("COCO_labels.txt");
	if(!result){
        cout << "loading labels failed";
        exit(-1);
	}

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;
    /*
     // Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classesFile.push_back(line);*/

    // Give the configuration and weight files for the model
    string modelConfiguration = "/home/pi/Desktop/TestYolo3tiny/yolov3-tiny.cfg";
    string modelWeights = "/home/pi/Desktop/TestYolo3tiny/yolov3-tiny.weights";

    // Load the network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);


	while(1){
//        frame=imread("Traffic.jpg");  //need to refresh frame before dnn class detection
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }

    chrono::steady_clock::time_point Tbegin, Tend;

    Mat blob = cv::dnn::blobFromImage(frame,  1/255.0, cv::Size(352, 352), Scalar(0,0,0), true, false);
    for(i=0;i<16;i++) FPS[i]=0.0;

    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
     Tbegin = chrono::steady_clock::now();
    net.forward(outs, getOutputsNames(net));


    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);
                    Tend = chrono::steady_clock::now();
                //calculate frame rate
                f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
                if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
                for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
             //   cout << f/16<< "\n";

                putText(frame, format("FPS %0.2f", f/16),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));
    // Write the frame with the detection boxes
  //  Mat detectedFrame;

        imshow("Detector and Tracking", frame);
        char esc = waitKey(5);
        if(esc == 27)
            break;
    }

    cout << "Closing the camera" << endl;
    destroyAllWindows();
    cout << "Bye!" << endl;

  return 0;
}


// Get the names of the output layers
vector<String> getOutputsNames(const cv::dnn::Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}



// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!Labels.empty())
    {
        CV_Assert(classId < (int)Labels.size());
        label = Labels[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}
