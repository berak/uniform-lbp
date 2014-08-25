cmake_minimum_required (VERSION 2.6) 

# Requires OpenCV v2.4.9 or later, because of the changed headers
#FIND_PACKAGE( OpenCV REQUIRED )
#IF (${OpenCV_VERSION} VERSION_LESS 2.4.9)
#    MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}. This requires atleast OpenCV v2.4.9")
#ENDIF()

SET(OPENCV_CPP_REQUIRED_DEPS opencv_core opencv_imgproc opencv_flann
    opencv_imgcodecs opencv_videoio opencv_highgui opencv_ml opencv_video
    opencv_objdetect opencv_photo opencv_features2d opencv_calib3d
    opencv_stitching opencv_videostab opencv_shape)

ocv_check_dependencies(${OPENCV_CPP_REQUIRED_DEPS})


PROJECT(Face)
SET(SRC
    main.cpp
    extractDB.cpp
    tan_triggs.cpp
    Extractor.cpp
    Classifier.cpp
)
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC} )
#TARGET_LINK_LIBRARIES( ${PROJECT_NAME}  ${OpenCV_LIBS} )
ocv_target_link_libraries(${PROJECT_NAME} ${OPENCV_LINKER_LIBS} ${OPENCV_CPP_REQUIRED_DEPS})
