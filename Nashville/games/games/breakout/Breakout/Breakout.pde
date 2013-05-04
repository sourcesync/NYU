/**
 * Players break the bricks by throwing balls
 *
 *   GAME CONTROLS
 *
 *   [spacebar]  Start game or skip to next level
 *   f/F         Increase/decrease ball floor offset
 *   h/H         Increase/decrease ball height scaling
 *   w/W         Increase/decrease ball width scaling
 *   0           Reset floor, scaling
 *   +/-         Increase/decrease ball diameter
 *   s           Enter/exit super mode.
 *   l           Enter/exit large mode.
 *   drag mouse  Adds an extra ball for cheating
 */

import processing.opengl.*;
import org.json.*;
import ddf.minim.*;
import fullscreen.*;

Minim minim;
ViconData vd;
Board board;

boolean simulated = true;

int windowWidth = 1920;
int windowHeight = 1080;
int ballRadius = 80;
float rescale = 0.85; //gw.5;


int lastMouseX = -1;
int lastMouseY = -1;
int skipAmount = 300;
boolean paused;
boolean cursorShowing = false;
float[][] locations;

//gw
boolean large_mode = false;
float height_factor = 1.0;
float width_factor = 1.0;
float floor_offset = 0.0;

int frame = 0;

void setup() {

  FullScreen fs = new FullScreen(this); // Create the fullscreen object
  fs.enter(); // enter fullscreen mode
  windowWidth = displayWidth;
  windowHeight = displayHeight;
  width = windowWidth;
  height = windowHeight;
  System.out.println( windowWidth + " " + windowHeight );
  //exit();

  minim = new Minim(this);

  ballRadius *= rescale;
  
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
  
  board.checkBricks(locations, ballRadius );
  board.draw();
  drawBalls();
  
  frame++;
  //save("frame_" + frame + ".png")
  
}


void drawBalls() {
  
  stroke(255, 0, 255, 255);
  fill(255, 0, 255, 200);
  
  if (board.superMode) {
    fill(255,255,155,255); 
  }
  
  for (int i = 0; i < locations.length; i++) {
      
      float x1 = (locations[i][0] + 1) / 2;
      float x = x1 * windowWidth;
      float y = locations[i][1] * windowHeight;
    
      ellipse(x,y,ballRadius * 2, ballRadius * 2);

  }
}

   JSONArray getmax( JSONArray objs )
   {
    try
    {
    if ( objs.length() > 0 )
    {
      int max_idx = 0;
      float max_val = 0.0;
      for (int i=0;i<objs.length();i++)
      {
        JSONArray _xyz = objs.getJSONArray(i);
        float val = (float)_xyz.getDouble(2);
        if ( i==0 ) max_val = val;
        else if ( val >= max_val ) 
          { max_val = val; max_idx = i; }
      }
      
      JSONArray maxobj = objs.getJSONArray(max_idx);
      float _x = (float)maxobj.getDouble(0);
      float _y = (float)maxobj.getDouble(1);
      float _z = (float)maxobj.getDouble(2);
      //java.lang.String str = "[ " + _x + "," + _y + "," + _z + " ]";
      java.util.Collection col = new ArrayList<Double>();
      col.add(_x);col.add(_y);col.add(_z);
      JSONArray newobjs = new JSONArray();    
      newobjs.put( 0, col );
      return newobjs;
    }
    else
    {
      return null;
    }
    }
    catch (JSONException e) 
    { 
      println (e.toString());
      return null;
    }
   }
 


void getData() {
    
   String newData = vd.getData();
   //System.out.println( "data " + newData );
   
   try {
     
      JSONObject vp = new JSONObject(newData);
      JSONArray objs = vp.getJSONArray("objs");
      
      if (large_mode)
      {
        objs = getmax(objs);
        if (objs == null ) return;
      }
      
      int numberLocations = objs.length();
      
      if (lastMouseX != -1 && lastMouseY != -1) {
        locations = new float[numberLocations+1][3];
        locations[numberLocations][0] = ((float)lastMouseX / windowWidth) * 2 - 1;
        locations[numberLocations][1] = (float)lastMouseY / windowHeight;
        locations[numberLocations][2] = 1;
      } else {
        locations = new float[numberLocations][3];
      }
      
      
      
      if (paused) return;
      for (int i=0; i<objs.length(); i++) 
      {       
        JSONArray xyz = objs.getJSONArray(i);
        locations[i][0] =(float)xyz.getDouble(0);
        locations[i][1] =(float)xyz.getDouble(1);
        locations[i][2] =(float)xyz.getDouble(2); 
        
        //gw - mirror y and possibly scale...
        locations[i][1] = (1.0 - locations[i][1]*height_factor) + floor_offset;
        //gw
        
        locations[i][0] = (locations[i][0]*width_factor);

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


  if (key == 'c') {
    cursorShowing = !cursorShowing;
    if (cursorShowing) cursor();
    else noCursor();
  }

  if (key == ' ') {
    board.next();
  }
  
  if (key == '+') {
    ballRadius += 5;
  }

  if (key == '-' && ballRadius > 10) {
    ballRadius -= 5;
  }

  if (key == 's') {
    board.superMode = ! board.superMode;
  }

//gw
  if (key=='l') {
    large_mode = !large_mode;
  }
  if (key=='h')
  {
    height_factor = height_factor * 0.98;
  }
  if (key=='H')
  {
    height_factor = height_factor * 1.02;
  }
  //gw
  
  
  if (key=='w')
  {
    width_factor = width_factor * 0.98;
  }
  if (key=='W')
  {
    width_factor = width_factor * 1.02;
  }
    
  if (key=='f')
  {
    floor_offset  += (.01 * height_factor);
  }
  if (key=='F')
  {
    floor_offset -= (.01 * height_factor);
  }
  
  if (key=='0')
  {
    width_factor = 1;
    height_factor = 1;
    floor_offset = 0; 
  }
  
  if ( key=='v')
  {
    println("Width Factor:" + width_factor);
    println("Height Factor:" + height_factor);
    println("Floor Offset:" + floor_offset);
  }
  
}


// Need to clear the keyCode explicitly or you get stuck in zoom mode
void keyReleased() {
  keyCode = 0;
}

