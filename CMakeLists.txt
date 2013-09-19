cmake_minimum_required (VERSION 2.6) 

# Requires OpenCV v2.4.9 or later, because of the changed headers
FIND_PACKAGE( OpenCV REQUIRED )
IF (${OpenCV_VERSION} VERSION_LESS 2.4.9)
    MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}. FaceRecShootout requires atleast OpenCV v2.4.9")
ENDIF()

PROJECT(FaceRecTest)
SET(SRC
    read.cpp
    lbp2.cpp
    combinedLBP.cpp
    Linear.cpp
    minmax.cpp
    SpatialHistogramReco.cpp
    wld.cpp
    VarLBP.cpp
)
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME}  ${OpenCV_LIBS} )


PROJECT(FaceRecOnline)
SET(SRC
    online.cpp
    lbp2.cpp
    combinedLBP.cpp
    Linear.cpp
    minmax.cpp
    SpatialHistogramReco.cpp
    wld.cpp
    VarLBP.cpp
)
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME}  ${OpenCV_LIBS} )


PROJECT(FaceRecShootout)
SET(SRC
    main.cpp
    lbp2.cpp
    ltp.cpp
    clbp_dist.cpp
    combinedLBP.cpp
    Linear.cpp
    minmax.cpp
    SpatialHistogramReco.cpp
    wld.cpp
    VarLBP.cpp
)
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME}  ${OpenCV_LIBS} )


