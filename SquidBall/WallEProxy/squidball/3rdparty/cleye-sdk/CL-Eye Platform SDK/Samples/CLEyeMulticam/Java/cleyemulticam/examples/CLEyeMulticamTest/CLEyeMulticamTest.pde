///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// This file is part of CL-EyeMulticam SDK
//
// Java Processing CLEyeProcessingTest Sketch
//
// It allows the use of multiple CL-Eye cameras in your own Java Processing applications
//
// For updates and file downloads go to: http://codelaboratories.com/about/product/eye
//
// Copyright 2008-2010 (c) Code Laboratories, Inc. All rights reserved.
//
// Dependancies: CLEyeMulticam.dll - CL-Eye Platform Multicam API
//               CLEyeMulticam.jar - For processing.org move this into your libraries folder
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Imports
import cl.eye.*;

// Camera Variables
int numCams;
CLCamera myCameras[] = new CLCamera[2];
PImage myImages[] = new PImage[2];
int cameraWidth = 640;
int cameraHeight = 480;
int cameraRate = 30;

// Animation Variables (not required)
boolean animate = true; 
float zoomVal, zoomDelta;
float rotateVal, rotateDelta;

void setup(){
  // Library loading via native interface (JNI)
  // If you see "UnsatisfiedLinkError" then target the library path otherwise leave it commented out.
  // CLCamera.loadLibrary("C://PATH//TO//CL-EYE SDK FOLDER//Bin/CLEyeMulticam.dll");

  // Verifies the native library loaded
  if(!setupCameras()) exit();
  // Setups animated variables 
  if(animate) setupAnimation();
}

void draw(){ 
  // Loops through available cameras and updates
  for(int i = 0; i < numCams; i++)
  {
    // ---------------------   (image destination, wait timeout)
    myCameras[i].getCameraFrame(myImages[i].pixels, (i==0) ? 1000 : 0);
    myImages[i].updatePixels();
    image(myImages[i], cameraWidth*i, 0);   
  }
  // Updates the animation
  if(animate)updateAnimation();
}

boolean setupCameras(){
  println("Getting number of cameras");
  // Checks available cameras
  numCams = CLCamera.cameraCount();
  println("Found " + numCams + " cameras");
  if(numCams == 0)  return false;
  // create cameras and start capture
  for(int i = 0; i < numCams; i++)
  {
    // Prints Unique Identifier per camera
    println("Camera " + (i+1) + " UUID " + CLCamera.cameraUUID(i));
    // New camera instance per camera
    myCameras[i] = new CLCamera(this);
    // ----------------------(i, CLEYE_GRAYSCALE/COLOR, CLEYE_QVGA/VGA, Framerate)
    myCameras[i].createCamera(i, CLCamera.CLEYE_COLOR_PROCESSED, CLCamera.CLEYE_VGA, cameraRate);
    // Starts camera captures
    myCameras[i].startCamera();
    myImages[i] = createImage(cameraWidth, cameraHeight, RGB); 
  }
  // resize the output window
  size(cameraWidth*numCams, cameraHeight);
  println("Complete Initializing Cameras");  
  return true;
}

void setupAnimation(){
   // General Animation Variables
   zoomVal = 0;
   zoomDelta = TWO_PI/75.0;
   rotateVal = 0;
   rotateDelta = TWO_PI/125.0;
}

void updateAnimation(){
   myCameras[0].setCameraParam(CLCamera.CLEYE_HKEYSTONE, (int)(150 * sin(rotateVal))); 
   myCameras[0].setCameraParam(CLCamera.CLEYE_VKEYSTONE, (int)(200 * cos(rotateVal)));
   //myCameras[0].setCameraParam(Multicam.CLEYE_LENSCORRECTION1, (int)(75 * sin(rotateVal)));
   if(numCams>1)
   {
     myCameras[1].setCameraParam(CLCamera.CLEYE_ZOOM, (int)(200 * sin(zoomVal)));
   }
   rotateVal += rotateDelta;
   zoomVal += zoomDelta;
}


