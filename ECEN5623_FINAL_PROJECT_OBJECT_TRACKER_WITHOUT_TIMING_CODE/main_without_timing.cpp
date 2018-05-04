/**************************************************************************************************************************************************************************************************
 *
 *    date edited: 1st May 2018
 *   
 *    File name: main_without_timing.cpp   
 * 
 *    The code for Kalman filter algorithm for object detection and tracking has been referred from: Myzhar's simple opencv kalman tracker
 *
 *    CODE DESCRIPTION:
 *
 *    - Main creates three threads namely capthread, trackerthread,motorthread
 *    - All the threads are created with the same priority and SCHED_FIFO as the scheduling policy
 *	  - cap thread captures raw image, converts RGB to HSV and applies different filters to smoothen image. Apply threshold values to detect object of particular colour and display threshold image
 *    - tracker thread detects the contours from the threshold image to extract object from background. Updates the Kalman filter matrix using the calculate values of center & area for detected image 
 *    - motor threads displays the output image with the trajectory lines and also moves the motor based on the position of the object.This is done supplying pwm signals to the motor. 
 *    - Binary semaphores are used to signal the threads 
 *    - Capthread waits for capthread_sempahore, trackerthread waits for tracker_thread_semaphore and motor thread waits for motor_thread_semaphore
 *    - Main threads posts the cap_thread_sempahore
 *    - Once cap thread finishes its execution, it signals the tracker_thread_semaphore and thus tracker thread starts executing
 *    - Once tracker thread finishes its execution, it signals the motor_thread_semaphore and thus motor thread starts executing 
 *    - motor thread finishes its execution and signals cap_thread_semaphore. The same pattern continues
 *
 *    NOTE: run the code using sudo command
 *
 **************************************************************************************************************************************************************************************************/

// Header files
#include <raspicam/raspicam_cv.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <semaphore.h>
#include <sched.h>
#include <errno.h>
#include <signal.h>

// Header files for opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <vector>
#include <wiringPi.h>

using namespace std;
using namespace cv;

// Value of Hue to be tracked in HSV color space for light green

// compile time flag for blue color
#ifdef COLOR_BLUE
#define MIN_BGR_1 (100)
#define MIN_BGR_2 (100)
#define MIN_BGR_3 (80)

#define MAX_BGR_1 (150)
#define MAX_BGR_2 (255)
#define MAX_BGR_3 (255)
#endif

// complie time flag for red color
#ifdef COLOR_RED
#define MIN_BGR_1 (160)
#define MIN_BGR_2 (100)
#define MIN_BGR_3 (80)

#define MAX_BGR_1 (180)
#define MAX_BGR_2 (255)
#define MAX_BGR_3 (255)
#endif

// macros for kalman filter
#define STATE_SIZE (6)
#define MEASURE_SIZE (4)
#define CONTOUR_SIZE (0)

//compile time switch for trajectory
#ifdef TRAJECTORY_ON
#define TRAJECTORY (1)
#endif


#define ARRAY_COUNT (100)

// gpio pins for PWM0 and PWM1
#define GPIO_PWM_PAN (18)
#define GPIO_PWM_TILT (13)

//values required for Servo
#define SERVO_RANGE (2000)
#define SERVO_CLK (192)

//Neutral,minimum and maximum positon values for PAN
#define PAN_NEUTRAL (95)
#define PAN_MAXIMUM (140)
#define PAN_MINIMUM (50)

//Neutral,minimum and maximum positon values for TILT
#define TILT_NEUTRAL (210)
#define TILT_MAXIMUM (240)
#define TILT_MINIMUM (195)

//MACRO to determine position of object on x axis
#define CENTER_X_LEFT1 (80)
#define CENTER_X_LEFT2 (160)
#define CENTER_X_LEFT3 (240)
#define CENTER_X (320)
#define CENTER_X_RIGHT1 (400)
#define CENTER_X_RIGHT2 (480)
#define CENTER_X_RIGHT3 (560)

//MACRO to determine position of object on y axis
#define CENTER_Y_UP (120)
#define CENTER_Y_DOWN (360)
#define NUMBER_OF_FRAMES (100)

//arrays required to store center x and center y values
int center_array_x[ARRAY_COUNT] = {0};
int center_array_y[ARRAY_COUNT] = {0};

//pthread_t
pthread_t capthread;
pthread_t trackerthread;
pthread_t motorthread;

//pthread attribute
pthread_attr_t capthread_attr;
pthread_attr_t trackerthread_attr;
pthread_attr_t motorthread_attr;

//structure for sched paramters
struct sched_param capthread_param;
struct sched_param trackerthread_param;
struct sched_param motorthread_param;

//declare semaphores
sem_t capthread_semaphore;
sem_t trackerthread_semaphore;
sem_t motorthread_semaphore;

//Global declarations for Kalman filter
raspicam::RaspiCam_Cv cap;
Mat object_range;
Mat result;
Mat frame;
unsigned int type = CV_32F;
KalmanFilter kalmanfilter((int)STATE_SIZE, (int)MEASURE_SIZE, (int)CONTOUR_SIZE, type);
Mat state((int)STATE_SIZE, 1, type);     
Mat measure((int)MEASURE_SIZE, 1, type); 
Point center;
Point prev_center;
	
//Global variables
bool found = false;
bool exitflag = false;
double ticks = 0;
int notFoundCount = 0;
int myframecount = 0;

//interrupt handler for CTRL-C
void siginthandler(int signum)
{
  //set the exit flag for graceful exit
	exitflag=true;	
	sem_post(&capthread_semaphore);
}

//function for capture thread
void *capthread_func(void *args)
{
  //run until exit flag is set
  while (!exitflag)
  {
 
    // wait for semaphore
    sem_wait(&capthread_semaphore);
	
    double precTick = ticks;
    ticks = (double) getTickCount();
    double dT = (ticks - precTick) / getTickFrequency();

    // Frame acquisition
		cap.grab();
    cap.retrieve(frame);
    frame.copyTo( result );

    if (found)
    {
    
      // Kalman Filter Matrix A
      kalmanfilter.transitionMatrix.at<float>(2) = dT;
      kalmanfilter.transitionMatrix.at<float>(9) = dT;
      
      state = kalmanfilter.predict();

      Rect predRect;
      predRect.width = state.at<float>(4);
      predRect.height = state.at<float>(5);
      predRect.x = state.at<float>(0) - predRect.width / 2;
      predRect.y = state.at<float>(1) - predRect.height / 2;

      Point center;
      center.x = state.at<float>(0);
      center.y = state.at<float>(1);
      circle(result, center, 2, CV_RGB(255,0,0), -1);

      rectangle(result, predRect, CV_RGB(255,0,0), 2);  
    }

    // apply gaussian blur function to smoothen noise 
    Mat blur;
    GaussianBlur(frame, blur, Size(5, 5), 3.0, 3.0);

    //convert RGB to HSV
    Mat frmHsv;
    cvtColor(blur, frmHsv, CV_BGR2HSV);
    
    //Apply threshold values to detect color
    // Note: change parameters for different colors
    object_range = Mat::zeros(frame.size(), CV_8UC1);
    inRange(frmHsv, Scalar(MIN_BGR_1, MIN_BGR_2, MIN_BGR_3),Scalar(MAX_BGR_1, MAX_BGR_2, MAX_BGR_3), object_range);
 
    // Apply erode and dilate filters to improve the image
    erode(object_range, object_range, Mat(), Point(-1, -1), 2);
    dilate(object_range, object_range, Mat(), Point(-1, -1), 2);
    
#ifdef THRESHOLD_ON
    // Display the threshold image using imshow
    imshow("Threshold", object_range);
#endif

    // post semaphore to release tracker thread
    sem_post(&trackerthread_semaphore);
	}

}

//function for tracker thread
void *trackerthread_func(void *args)
{

  //run until exit flag is set
  while (!exitflag)
  {
	
    //wait for semaphore
    sem_wait(&trackerthread_semaphore);
  
    vector<vector<Point> > balls;
		vector<Rect> ballsBox;
		vector<vector<Point> > contours;
 
    //Detect Contours using findContours function
		findContours(object_range, contours, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    //Apply filtering to getthe rectangle
    for (size_t i = 0; i < contours.size(); i++)
    {
      Rect bBox;
      bBox = boundingRect(contours[i]);

      float ratio = (float) bBox.width / (float) bBox.height;
      if (ratio > 1.0f)
      ratio = 1.0f / ratio;

      // Search for a box which almost resembles a square
      if (ratio > 0.75 && bBox.area() >= 400)
      {
        balls.push_back(contours[i]);
        ballsBox.push_back(bBox);
      }
        
    }
   
    
    // result after filtering    
    for (size_t i = 0; i < balls.size(); i++)
    {
   
      drawContours(result, balls, i, CV_RGB(20,150,20), 1);
      rectangle(result, ballsBox[i], CV_RGB(0,255,0), 2);
      
#ifdef TRAJECTORY
      if(myframecount > 0)
   		{
	       prev_center.x = center.x;
	       prev_center.y = center.y;
	    }
#endif

      center.x = ballsBox[i].x + ballsBox[i].width / 2;
      center.y = ballsBox[i].y + ballsBox[i].height / 2;
     

#ifdef TRAJECTORY
      center_array_x[myframecount] = center.x;
			center_array_y[myframecount] = center.y;	    
#endif

      myframecount++;
			
      circle(result, center, 2, CV_RGB(20,150,20), -1);

      //debug print statements
//		cout << "x : " << center.x << " y : " << center.y << endl;

      //displaying the center coordinates on the image
      stringstream sstr;
      sstr << "(" << center.x << "," << center.y << ")";
			putText(result, sstr.str(),
			Point(center.x + 3, center.y - 3),FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
        
    }

    // Updating the Kalman Filter
    if (balls.size() == 0)
    {
      //increment not found ball count 
      notFoundCount++;
		
      //reset the frame count if count of number of balls not found reaches 15
		 	if(notFoundCount  == 15)
		  {
		   	myframecount = 0;
		  }
		
      if( notFoundCount >= 100 )
			{
        found = false;
      }

    }   
    else
    {
      //if ball is found, reset not found count to zero
      notFoundCount = 0;
      measure.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
      measure.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
      measure.at<float>(2) = (float)ballsBox[0].width;
      measure.at<float>(3) = (float)ballsBox[0].height;


      // Detection of ball for the first time
      if (!found) 
      {
        // Initialize the kalman filter
        kalmanfilter.errorCovPre.at<float>(0) = 1; 
        kalmanfilter.errorCovPre.at<float>(7) = 1; 
        kalmanfilter.errorCovPre.at<float>(14) = 1;
        kalmanfilter.errorCovPre.at<float>(21) = 1;
        kalmanfilter.errorCovPre.at<float>(28) = 1;       
        kalmanfilter.errorCovPre.at<float>(35) = 1;

        state.at<float>(0) = measure.at<float>(0);
        state.at<float>(1) = measure.at<float>(1);

        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        state.at<float>(4) = measure.at<float>(2);
        state.at<float>(5) = measure.at<float>(3);

        kalmanfilter.statePost = state;
                
        found = true;      
      }      
      else
      {
        // make correction to the kalman filter
        kalmanfilter.correct(measure); 
      }
        
    }

    //post semaphore to release motor thread
		sem_post(&motorthread_semaphore);
	}

}


// function for motor thread
void *motorthread_func(void *args)
{
 
  //variables required to store the current motor position
  static uint32_t pan_current_value = PAN_NEUTRAL;
  static uint32_t tilt_current_value = TILT_NEUTRAL;

  //run until exit flag is set
  while (!exitflag)
  {
		//wait for semaphore  
    sem_wait(&motorthread_semaphore);
	
    //function to draw the trajectory lines. openMP used for parallel execution of for loop
#ifdef TRAJECTORY
        if(myframecount > 2)
	    {
#pragma omp parallel shared(center_array_x,center_array_y,result)
          {
            int i =0;
#pragma omp for private(i)
		for(i = 1; i<myframecount; i++)
			  {
				  line(result, Point(center_array_x[i],center_array_y[i]), Point(center_array_x[i-1] ,center_array_y[i-1]), Scalar(0,255,0),5);   	
			  }
		  }
		}
#endif
	  
    // reset frame count
    if(myframecount == ARRAY_COUNT)
	  {
		  myframecount = 0;
	  }
	
    // Display the final result
    imshow("Tracking", result);
		waitKey(1);
	  
		
    //check for frame count greater than zero
    if(myframecount > 0)
		{
      
      //determine in which frame region the object is for the x axis
      if((center.x > CENTER_X_RIGHT1 && center.x < CENTER_X_RIGHT2) || (center.x > CENTER_X_RIGHT2 && center.x < CENTER_X_RIGHT3) || (center.x > CENTER_X_RIGHT3))
      { 
          
        //check if motor has reached maximum rotation 
        if((pan_current_value + 5) < PAN_MAXIMUM)
        {
          //move the pan motor accordingly to try and get the frame in the center of frame 
          pan_current_value = pan_current_value+5;
          pwmWrite(GPIO_PWM_PAN,pan_current_value);
        }
   
      }
      else if((center.x < CENTER_X_LEFT3 && center.x > CENTER_X_LEFT2) || (center.x < CENTER_X_LEFT2 && center.x > CENTER_X_LEFT1) || (center.x < CENTER_X_LEFT1))
      {
          
        //check if motor has reached maximum rotation 
        if((pan_current_value - 5) > PAN_MINIMUM)
        { 
          //move the pan motor accordingly to try and get the frame in the center of frame 
          pan_current_value = pan_current_value-5;
          pwmWrite(GPIO_PWM_PAN,pan_current_value);
        } 
      }
       
      //determine in which frame region the object is for the y axis
      if(center.y < CENTER_Y_UP)  
      {
        //check if motor has reached maximum rotation 
        if((tilt_current_value + 2) < TILT_MAXIMUM)
        {   
          //move the tilt motor accordingly to try and get the frame in the center of frame 
          tilt_current_value = tilt_current_value + 2;
          pwmWrite(GPIO_PWM_TILT,tilt_current_value);  
        }
       
      }
      else if(center.y > CENTER_Y_DOWN)
      {
        
        //check if motor has reached maximum rotation 
        if((tilt_current_value - 2) > TILT_MINIMUM)
        { 
          //move the tilt motor accordingly to try and get the frame in the center of frame 
          tilt_current_value = tilt_current_value - 2;
          pwmWrite(GPIO_PWM_TILT,tilt_current_value);
        }      
        
      }
    }
	
    //post the semaphore to release capture thread
    sem_post(&capthread_semaphore);   
    
  }
}
	


int main()
{
	
  /*initialize signal handler*/
	signal(SIGINT,siginthandler);    
	
	/*initialize semaphores*/
	sem_init(&capthread_semaphore, 0, 0);
	sem_init(&trackerthread_semaphore, 0, 0);  
	sem_init(&motorthread_semaphore, 0, 0);
  
	//setting pthread attributes for capthread
 	capthread_param.sched_priority = 98;
  pthread_attr_init(&capthread_attr);
  pthread_attr_setinheritsched(&capthread_attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setschedpolicy(&capthread_attr , SCHED_FIFO);
  pthread_attr_setschedparam(&capthread_attr, &capthread_param);	
	
	//setting pthread attributes for tracker thread
 	trackerthread_param.sched_priority = 98;
  pthread_attr_init(&trackerthread_attr);
  pthread_attr_setinheritsched(&trackerthread_attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setschedpolicy(&trackerthread_attr , SCHED_FIFO);
  pthread_attr_setschedparam(&trackerthread_attr, &trackerthread_param);	

	//setting pthread attributes for motor thread
  motorthread_param.sched_priority = 98;
  pthread_attr_init(&motorthread_attr);
  pthread_attr_setinheritsched(&motorthread_attr, PTHREAD_EXPLICIT_SCHED);
  pthread_attr_setschedpolicy(&motorthread_attr , SCHED_FIFO);
  pthread_attr_setschedparam(&motorthread_attr, &motorthread_param);	


	//setup gpio with wiring pi for raspberry pi gpio
	wiringPiSetupGpio();
	
	//set pin mode to output for pwm0 and pwm1
	pinMode (GPIO_PWM_PAN, PWM_OUTPUT);
 	pinMode (GPIO_PWM_TILT, PWM_OUTPUT);

  //setup clock and range for servo motor at 50 Hz frequency
	pwmSetMode (PWM_MODE_MS);
  pwmSetRange (SERVO_RANGE);
  pwmSetClock (SERVO_CLK);	

  //initialize the motors to be at neutral position
  pwmWrite(GPIO_PWM_PAN , PAN_NEUTRAL);
  pwmWrite(GPIO_PWM_TILT , TILT_NEUTRAL);

	uint32_t return_val=0;
	
	//create capture thread
  return_val = pthread_create(&capthread,&capthread_attr, capthread_func, NULL);
	if(return_val == 0)
	{
	  cout<<"Cap thread created"<<endl;
	}
	else
	{
	  perror("\nERROR: \n");
	  exit(1);
	}
	
	//create tracker thread
	return_val = pthread_create(&trackerthread,&trackerthread_attr, trackerthread_func, NULL);
	if(return_val == 0)
	{
	  cout<<"Tracker thread created"<<endl;
	}
	else
	{
	  perror("\nERROR: \n");
	  exit(1);
	}


	//create motor thread
  return_val = pthread_create(&motorthread,&motorthread_attr, motorthread_func, NULL);
	if(return_val == 0)
	{
	  cout<<"Motor Thread created"<<endl;
	}
	else
	{
	  perror("\nERROR: \n");
	  exit(1);
	}

  //set the number of thread required for openmp to the number of cores available
  omp_set_num_threads(4);

  setIdentity(kalmanfilter.transitionMatrix);

  //Measurement of kalan filter Matrix
  kalmanfilter.measurementMatrix = Mat::zeros((int)MEASURE_SIZE,(int) STATE_SIZE, type);
  kalmanfilter.measurementMatrix.at<float>(0) = 1.0f;
  kalmanfilter.measurementMatrix.at<float>(7) = 1.0f;
  kalmanfilter.measurementMatrix.at<float>(16) = 1.0f;
  kalmanfilter.measurementMatrix.at<float>(23) = 1.0f;

  // Process the noise covariance
  kalmanfilter.processNoiseCov.at<float>(0) = 1e-2;
  kalmanfilter.processNoiseCov.at<float>(14) = 5.0f;
  kalmanfilter.processNoiseCov.at<float>(28) = 1e-2;
  kalmanfilter.processNoiseCov.at<float>(35) = 1e-2;

  // Measures Noise Covariance Matrix R
  setIdentity(kalmanfilter.measurementNoiseCov, Scalar(1e-1));
    
  // Index required for camera
  int idx = 0;

  // Camera Capture
  // VideoCapture cap;      //if using Jetson TK1
	

  //set all camera capture parameters 
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  cap.set(CV_CAP_PROP_BRIGHTNESS,50);
  cap.set(CV_CAP_PROP_CONTRAST,50);
  cap.set(CV_CAP_PROP_SATURATION,50);
  cap.set(CV_CAP_PROP_GAIN,50);
  cap.set(CV_CAP_PROP_FORMAT,CV_8UC3);
  cap.set(CV_CAP_PROP_EXPOSURE,-1);
  cap.set(CV_CAP_PROP_WHITE_BALANCE_RED_V,-1);
  cap.set(CV_CAP_PROP_WHITE_BALANCE_BLUE_U,-1);


  // OPEN Camera
  if (!cap.open())
  {
    cout << "Webcam not connected.\n" << "Please verify\n";
    return EXIT_FAILURE;
  }    
    
  cout << "\nHit Ctrl-C to exit...\n";
	
  //post capture thread to start execution
  sem_post(&capthread_semaphore);

	
	//wait for thread to finish its execution
	pthread_join(capthread, NULL);
	pthread_join(trackerthread, NULL);
	pthread_join(motorthread, NULL);
	
  //release the camera
  cap.release();

  cout<<"\n----------------------------END OF MAIN----------------------------\n"<<endl;
	return 0;
}
