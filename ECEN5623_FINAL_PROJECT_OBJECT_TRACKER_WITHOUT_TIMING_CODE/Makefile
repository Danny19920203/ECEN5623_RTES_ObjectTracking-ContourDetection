COLOR=BLUE
TRAJECTORY = OFF
THRESHOLD = OFF

ifeq ($(COLOR),RED)
	FLAGS:=-DCOLOR_RED
endif

ifeq ($(COLOR),BLUE)
	FLAGS:=-DCOLOR_BLUE
endif

ifeq ($(TRAJECTORY),ON)
	TRAJ:=-DTRAJECTORY_ON
endif

ifeq ($(TRAJECTORY),OFF)
	TRAJ:=-DTRAJECTORY_OFF
endif

ifeq ($(THRESHOLD),OFF)
	THRESH:=-DTHRESHOLD_OFF
endif

ifeq ($(THRESHOLD),ON)
	THRESH:=-DTHRESHOLD_ON
endif


all: main_without_timing.cpp 
	g++ main_without_timing.cpp -o object_tracker -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio -lopencv_videostab -lraspicam -lraspicam_cv -lrt -lpthread -l wiringPi -fopenmp $(FLAGS) $(TRAJ) $(THRESH)

clean:
	rm -rf object_tracker
