/**
 * Players uncover images by throwing balls around in a large
 * motion capture volume.
 *
 * Dragging the mouse changes the camera angle. Option-dragging
 *  (Alt-dragging) changes the zoom. Shift-drag controls dolly.
 *
 * Press s to skip ahead in the simulated data. 
 * Press spacebar to clear the current strokes. 
 * Press c to show/hide the cursor.
 *
 */

import processing.opengl.*;
import org.json.*;
import ddf.minim.*;

String master = null;

// Top level globals you might want to change
boolean simulated = false;


int windowWidth = 1280;
int windowHeight = 1024;
int skipAmount = 300;

// Variables for controlling the camera view
float delta = .05;
float minAngle = -PI/2;
float maxAngle = PI/2;

//gw
//float startZoom = 0.35750014;
float startZoom = 1.0;
//gw
float zoom = startZoom;
float minZoom = .032;
float maxZoom = 2;
PVector dolly, prevDolly;
PVector startDolly = new PVector(89, -21);
float rotateFactor = .005;
float dollyFactor = 1;
float xzRatio = .67;
float zoomFactor = .0005;
int prevLevel = 0;

float angle, inc, prevZoom;
float deltaMouseX, deltaMouseY, startX, startY;

float startDeltaX = -1;
float startDeltaY = 250;//350;
float minDeltaY = 320;
float maxDeltaY = 60;

boolean paused;
boolean guessMode = true;
boolean showAnswer = false;
ViconData vd;
StrokeSet ss;

PFont font;
int quitCount = 0;
int quitMax = 100;
int effectCount = 0;
int maxEffect = 100;

PImage colImg, floorImg;

float columnWidth = 400;
PVector[] columns;
PImage[] images;
int numberImages;
int currentImage = 0;
String[] answers;
AudioSample win;
Minim minim;

void loadImages() {

  String[] imageStrings = loadStrings("images.txt");
  
  numberImages = imageStrings.length;

  images = new PImage[numberImages];
  answers = new String[numberImages];
  
  for (int i = 0; i < numberImages; i++) {
    String[] fields = split(imageStrings[i],",");
    images[i] = loadImage("guess/" + fields[0]);
    answers[i] = fields[1];
  }
}



// Standard methods: setup, draw, mousePressed, mouseDragged, keyPressed, keyReleased
void setup() {


  loadImages();

  noCursor();

  frame.setBackground(new java.awt.Color(0, 0, 0));
  frameRate(30);
  size(windowWidth, windowHeight, P3D); 
  frameRate(30);

   //gw hint(DISABLE_OPENGL_2X_SMOOTH);
  //colorMode(HSB);
   minim = new Minim(this);
   win = minim.loadSample("applause.wav", 2048);

  vd = new ViconData(simulated, (master != null));
  font = loadFont("Arial-BoldMT-128.vlw");
  textFont(font, 128);
  textAlign(CENTER);

  ss = new StrokeSet(this);
  ss.xzRatio = xzRatio;

  dolly = startDolly;
  inc = delta;
  angle = 0;

  //Initially tilt the camera so we're looking with z up
  deltaMouseY = startDeltaY;
  deltaMouseX = startDeltaX;

  ss.rx = deltaMouseX*rotateFactor;    
  ss.ry = deltaMouseY*rotateFactor;
}


void loadData() {
  String message = vd.getData();
  System.out.println(message);
  ss.addStrokesFromJSON(message);
}


void fade() {
  int r,g,b;
  loadPixels();
  for (int i=0;i<pixels.length;i++) {
    r=pixels[i]>>16&255;
    g=pixels[i]>>8&255;
    b=pixels[i]&255;
    pixels[i]=color(r-1,g-1,b-1);
  }
  updatePixels();
}

void draw() {

  if (!paused) {
    loadData();
  }

  if (quitCount != 0) {
    //hint(ENABLE_DEPTH_TEST);
  }

  pushMatrix();
  background(0,0,0); 
  noFill();
  noStroke();

  translate(width/2, height/2, 0);
  translate(-dolly.x*dollyFactor, -dolly.y*dollyFactor, 0);
  scale(zoom,zoom,-zoom);

  float rr = sin(angle)/20;
  // Rotate based on the mouse delta plus the floating angle
  rotateY(rr);
  rotateX(deltaMouseY*rotateFactor);
  rotateZ(deltaMouseX*rotateFactor);

  // Draw main bounding box
  strokeWeight(3);
  stroke(255,0,255,200);
  //box(vd.deltaXYZ.x, vd.deltaXYZ.y, vd.deltaXYZ.z);

  noStroke();

  if (!guessMode) {

    stroke(255);
    strokeWeight(1);

    float zz = -vd.deltaXYZ.z/2;
    float yy = 0;    
    stroke(255,255,255);

    beginShape(LINES);
    int grid = 10;
    for (int i = 0; i <= grid; i++) {
      float gx1 = (float)i / (float)grid;
  
       vertex(vd.minXYZ.x*xzRatio + xzRatio*(gx1 * vd.deltaXYZ.x), yy, vd.minXYZ.z);
       vertex(vd.minXYZ.x*xzRatio + xzRatio*(gx1 * vd.deltaXYZ.x), yy, vd.maxXYZ.z);
       vertex(vd.minXYZ.x*xzRatio, yy, vd.minXYZ.z + (gx1 * vd.deltaXYZ.z));
       vertex(vd.maxXYZ.x*xzRatio, yy, vd.minXYZ.z + (gx1 * vd.deltaXYZ.z));
   
    }
    endShape(LINES);    
    
  }
  noStroke();
  
  // The box primitive is drawn centered though the true bounds may not be centered in
  //  the real world. We need to translate by this position so strokes come out aligned.
  pushMatrix();
  translate(-vd.meanXYZ.x, -vd.meanXYZ.y, -vd.meanXYZ.z);


  if (guessMode) {
    float w1 = images[currentImage].width;
    float h1 = images[currentImage].height;

    float xx = 0;
    float yy = 0;
    float ww = windowWidth;
    float hh = windowHeight;

    float imgAspect = w1 / h1;
    float windowAspect = (float) windowWidth / (float) windowHeight;

    //Too wide, move it down
    if (imgAspect >= windowAspect) {
      hh = windowWidth / imgAspect;
      yy = (windowHeight - hh) / 2;
    } 
    else {
      ww = windowHeight * imgAspect;
      xx = (windowWidth - ww) / 2;
    }

    if (showAnswer) {      
      blend(images[currentImage], 0, 0, (int)w1, (int)h1, (int)xx, (int)yy, (int)ww, (int)hh, ADD);
    } 
    else {
      ss.draw();
      blend(images[currentImage], 0, 0, (int)w1, (int)h1, (int)xx, (int)yy, (int)ww, (int)hh, DARKEST);
    }
  } 
  else {
    ss.draw();
  }

  popMatrix();
  popMatrix();
  
  
  if (showAnswer && guessMode) {      
      
      pushMatrix();
      translate(windowWidth/2, windowHeight/2 + 50);
      float zoomScale = 2  *   (float)effectCount / (float)maxEffect;
      float fadeScale = 1;
      if (zoomScale > 1) fadeScale = 2 - zoomScale;

      scale(zoomScale, zoomScale, zoomScale);
      
      
      fill(255,255,255,255*fadeScale);
      text(answers[currentImage],3,3);
      fill(0,0,0,255*fadeScale);
      text(answers[currentImage],-3,-3);
      fill(255,0,0,255*fadeScale);
      text(answers[currentImage],0,0);
      

      effectCount++;
      popMatrix();
      if (effectCount > maxEffect) {
        showAnswer = false;
        currentImage++;
        if (currentImage >= numberImages) currentImage = 0;
        effectCount = 0;
      }
    } 

  if (quitCount != 0) {
    noStroke();
    hint(DISABLE_DEPTH_TEST);
    fill(0,0,0,255*((float)quitCount/(float)quitMax));
    ss.fade = 1-(float)quitCount/(float)quitMax;
    
    //gw rect(0,0,screenWidth,screenHeight);
    rect(0,0,displayWidth,displayHeight);
    
    quitCount++;
    if (quitCount >= quitMax) exit();
  }
  //  background(0,0,255,128);
}

boolean firstMouse = true;

void mousePressed() {
  startX = mouseX;
  startY = mouseY;
  prevDolly = dolly;
  prevZoom = zoom;
}


void mouseDragged() {
  //Prevents a glitch that arises of the first mouse movement is a drag
  if (firstMouse) {
    mousePressed();
    firstMouse = false;
    return;
  }

  // Option dragging vertically changes zoom
  if (keyCode == 18) {
    zoom += (startY - mouseY)*zoomFactor;
  }

  // Shift dragging changes rot
  else if (keyCode == 16) {
    deltaMouseX += -(mouseX - startX);
    deltaMouseY += (startY - mouseY);

    ss.rx = deltaMouseX*rotateFactor;    
    ss.ry = deltaMouseY*rotateFactor;
  } 
  else {

    dolly.x += (startX - mouseX)*dollyFactor;
    dolly.y += (startY - mouseY)*dollyFactor;
  }


  startX = mouseX;
  startY = mouseY;

  //println(deltaMouseX + "/" + deltaMouseY + "/" + zoom);
}


boolean cursorShowing = true;

void keyPressed() {

  if (key == ' ') {
    ss.clearStrokes();
  }

  //print(">" + keyCode + "<");

  if (key == 'p' && simulated) {
    paused = !paused;
  }

  if (keyCode == 40) {
    ss.weight-=20; 
    if (ss.weight < 10) ss.weight = 10;
  }

  if (keyCode == 38) {
    ss.weight+=20;
  }

  if (key == '[') {
    ss.xzRatio-=.05; 
    xzRatio = ss.xzRatio;
  }

  if (key == ']') {
    ss.xzRatio+=.05; 
    xzRatio = ss.xzRatio;
  }

  if (keyCode == 37) {
     if (ss.tail > 1) ss.tail--;
  }

  if (keyCode == 39) {
   if (ss.tail < ss.maxStrokes-1) ss.tail++;
  }

  if (key == 'c') {
    cursorShowing = !cursorShowing;
    if (cursorShowing) cursor();
    else noCursor();
  }


  if (key == 'n') {
    showAnswer = !showAnswer;
    if (!showAnswer) {
      currentImage++;
      ss.clearStrokes();
    } else if (currentImage > 0) {
       win.trigger(); 
       effectCount = 0;
    } else {
      effectCount = maxEffect - 5;
    }

    if (currentImage >= numberImages) currentImage = 0;
  }

  if (key == 'j' && simulated) {
    for (int i = 0; i < skipAmount; i++) {
      String newData = vd.getData();
      ss.addStrokesFromJSON(newData);
    }
  }

  if (key == 's' && simulated) {
    vd.skipFrames(skipAmount);
    ss.clearStrokes();
  }


  if (key == 'g') {
    guessMode = !guessMode;
  }

  if (key == 't') {
    ss.tailMode = !ss.tailMode;
    if (ss.tailMode) ss.tail = 5;
    else ss.tail = 1;
  }

  if (key == 'r') {
    deltaMouseX = startDeltaX;
    deltaMouseY = startDeltaY;
    zoom = startZoom;
    dolly.x = 0;
    dolly.y = 0;
    ss.rx = deltaMouseX*rotateFactor;    
    ss.ry = deltaMouseY*rotateFactor;
  }

  if (key == 'q') {
    quitCount = 1;
  }

  if (key == 'v') {
    println("Zoom: " + zoom);
    println("Dolly: (" + dolly.x + "," + dolly.y + ")");
    println("Rotate: (" + deltaMouseX + "," + deltaMouseY + ")");
  }
}


// Need to clear the keyCode explicitly or you get stuck in zoom mode
void keyReleased() {
  keyCode = 0;
}



void stop() {
  ss.stop();
}

