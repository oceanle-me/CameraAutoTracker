#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include <cmath>

#define REAL_FPS 13

#include <iostream>
#include <pigpio.h>

#include "tracking.h"

using namespace cv;
using namespace std;

const size_t width = 300;
const size_t height = 300;

std::vector<std::string> Labels;
std::unique_ptr<tflite::Interpreter> interpreter;

/***************************************************/
std::string gstreamer_pipeline(int capture_width, int capture_height, int framerate, int display_width, int display_height) {
    return
            " libcamerasrc ! video/x-raw, "
            " width=(int)" + std::to_string(capture_width) + ","
            " height=(int)" + std::to_string(capture_height) + ","
            " framerate=(fraction)" + std::to_string(framerate) +"/1 !"
            " videoconvert ! videoscale !"
            " video/x-raw,"
            " width=(int)" + std::to_string(display_width) + ","
            " height=(int)" + std::to_string(display_height) + " ! appsink";
}


/*****************************************************/

void control_motor(int x,int y){ //input: x= (x1+x2)/2; y = (y1+y2)/2 -------300xx300
    //signal scaler in -+ 200:1000
    //cout <<x << "  ";
    if (x<150){
       unsigned int scaler = ((150-(float)x)/150)*185 + 70 ;
        cout <<scaler << " \n ";
        gpioPWM(12,0);
        gpioPWM(13,scaler);
    } else {
       unsigned int scaler = (((float)x-150)/150)*185 + 70 ;
        cout <<scaler << " \n ";
        gpioPWM(12,scaler);
        gpioPWM(13,0);
    }

}

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

void detect_from_video(Mat &src)
{
    Mat image;
    int cam_width =src.cols;
    int cam_height=src.rows;

    // copy image to input as input tensor
    cv::resize(src, image, Size(300,300));
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      //quad core

//        cout << "tensors size: " << interpreter->tensors_size() << "\n";
//        cout << "nodes size: " << interpreter->nodes_size() << "\n";
//        cout << "inputs: " << interpreter->inputs().size() << "\n";
//        cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
//        cout << "outputs: " << interpreter->outputs().size() << "\n";

    interpreter->Invoke();      // run your model

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    const float confidence_threshold = 0.7;

    float pre_x1 =150;
    float pre_x2 =150;
    float pre_number_diff_error = -1;

    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > confidence_threshold){

            int  det_index = (int)detection_classes[i]+1;
            if (det_index==1){ //detecting person only
                float y1=detection_locations[4*i  ]*cam_height;
                float x1=detection_locations[4*i+1]*cam_width;
                float y2=detection_locations[4*i+2]*cam_height;
                float x2=detection_locations[4*i+3]*cam_width;

                //Rect:   x coordinate of the top-left corner, y coordinate of the top-left corner
                Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
                rectangle(src,rec, Scalar(0, 0, 255), 1, 8, 0);
                putText(src, format("%s", Labels[det_index].c_str()), Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 0, 255), 1, 8, 0);
                circle(src, Point((int)(x1 + ((x2 - x1)/2)) ,(int)(y1 + ((y2 - y1)/2))), 4, Scalar(0, 0, 255), 4,8,0);

                float diff_error = abs((x1+x2) - (pre_x1+pre_x2));
                if(pre_number_diff_error==-1){
                    pre_number_diff_error=diff_error;
                    pre_x1=x1;
                    pre_x2=x2;
                }
                if (diff_error < pre_number_diff_error){
                    pre_number_diff_error=diff_error;
                    pre_x1=x1;
                    pre_x2=x2;
                }

                }
        }
    }

   // control_motor((pre_x1 + pre_x2)/2,0);

}



int main(int argc,char ** argv)
{


    gpioInitialise();

    gpioSetMode(12, PI_ALT0);
    gpioSetPullUpDown(12, PI_PUD_DOWN);
    gpioSetPWMfrequency(12,10000);
    gpioPWM(12,0);

    gpioSetMode(13, PI_ALT0);
    gpioSetPullUpDown(13, PI_PUD_DOWN);
    gpioSetPWMfrequency(13,10000);


 //   gpioSetMode(21,PI_OUTPUT);


    //pipeline parameters
    int capture_width = 1280; //1280 ;
    int capture_height = 1280; //1080 ;
    int framerate = 30 ;
    int display_width = 300; //1280 ;
    int display_height = 300; //720 ;

   /* int capture_width = 1280; //1280 ;
    int capture_height = 720; //720 ;
    int framerate = 20 ;
    int display_width = 1280; //1280 ;
    int display_height = 720; //720 ;*/

    //libcamerasrc ! video/x-raw,  width=(int)1280, height=(int)1280, framerate=(fraction)25/1 ! videoconvert ! videoscale ! video/x-raw, width=(int)300, height=(int)300 ! appsink
 //   gst-launch-1.0 libcamerasrc ! video/x-raw, width=1280, height=1280, framerate=30/1 ! videoconvert ! videoscale ! video/x-raw, width=300, height=300 ! clockoverlay time-format="%D %H:%M:%S" ! autovideosink


        //reset frame average
    std::string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate,
                                              display_width, display_height);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n\n\n";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    float f;
    float FPS[16];
    int i, Fcnt=0;
    Mat frame;
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("detect.tflite");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();

	// Get the names
	bool result = getFileContent("COCO_labels.txt");
	if(!result)
	{
        cout << "loading labels failed";
        exit(-1);
	}

  // VideoCapture cap("James.mp4");
     //   VideoCapture cap(0);
   /* if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }*/

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;
	while(1){
//        frame=imread("Traffic.jpg");  //need to refresh frame before dnn class detection
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }

        Tbegin = chrono::steady_clock::now();

    //    cv::resize(frame, frame, Size(300,300));
        detect_from_video(frame);

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(frame, format("FPS %0.2f", f/16),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));

       // cout << "FPS" << f/16 << endl;
        imshow("RPi 4 - 1,9 GHz - 2 Mb RAM", frame);

        char esc = waitKey(5);
        if(esc == 27) break;
    }


    gpioWrite(12,0);
    gpioWrite(13,0);
    cout << "Closing the camera" << endl;
    destroyAllWindows();
    cout << "Bye!" << endl;

  /***************************/
    gpioTerminate();

  return 0;
}
