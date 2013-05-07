/**
 * Players draw by throwing balls around in a large
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


ViconData vd;
Board board;

boolean simulated = false;

int windowWidth = 1920;
int windowHeight = 1080;

int lastMouseX = -1;
int lastMouseY = -1;

float rescale = .5;

int skipAmount = 300;

boolean paused;
boolean cursorShowing = true;


float[][] locations;



void setup() {

  windowWidth = (int)(windowWidth * rescale);
  windowHeight = (int)(windowHeight * rescale);
  
  if (!cursorShowing) noCursor();

  background(0);
  size(windowWidth, windowHeight);
  colorMode(HSB);
  smooth();
  frameRate(30);

  vd = new ViconData(simulated);
  board = new Board();
}


void draw() {
  
  getData();

  background(0,0,0);
  
  board.checkBricks(locations);
  board.draw();
  drawBalls();
}


void drawBalls() {
  
  stroke(255, 0, 255, 255);
  fill(255, 0, 255, 200);
  
  for (int i = 0; i < locations.length; i++) {
      
      float x1 = (locations[i][0] + 1) / 2;
      float x = x1 * windowWidth;
      // gw - reverse
      float y = locations[i][1] * windowHeight;
      //float y = windowHeight - locations[i][1] * windowHeight;
      // gw
      ellipse(x,y,50 * rescale, 50 * rescale);

  }
}


void getData() {
    
   String newData = vd.getData();
   
   try {
     
      JSONObject vp = new JSONObject(newData);
      JSONArray objs = vp.getJSONArray("objs");
      int numberLocations = objs.length();
      
      if (lastMouseX != -1 && lastMouseY != -1) 
      {
        locations = new float[numberLocations+1][3];
        locations[numberLocations][0] = ((float)lastMouseX / windowWidth) * 2 - 1;
        locations[numberLocations][1] = (float)lastMouseY / windowHeight;
        locations[numberLocations][2] = 1;
      } 
      else 
      {
        locations = new float[numberLocations][3];
      }
      
      if (paused) return;
      for (int i=0; i<objs.length(); i++) 
      {       
        JSONArray xyz = objs.getJSONArray(i);
        locations[i][0] =(float)xyz.getDouble(0);
        locations[i][1] =(float)xyz.getDouble(1);
        locations[i][2] =(float)xyz.getDouble(2); 
        
        //gw - mirror y
        locations[i][1] = 1.0 - locations[i][1];
        //gw
        
      }

    }
    catch (JSONException e) { 
      println (e.toString()); 
      println( "Original Message: \""+newData+"\"");
    } 
}



void mousePressed() {}

void mouseDragged() {
  lastMouseX = mouseX;
  lastMouseY = mouseY;  
}

void mouseReleased() {
 lastMouseX = -1;
 lastMouseY = -1; 
}

void keyPressed() {

  if (key == 'p' && simulated) {
    paused = !paused;
  }

  if (key == 'c') {
    cursorShowing = !cursorShowing;
    if (cursorShowing) cursor();
    else noCursor();
  }

  if (key == ' ') {
    board.next();
  }

}


// Need to clear the keyCode explicitly or you get stuck in zoom mode
void keyReleased() {
  keyCode = 0;
}

