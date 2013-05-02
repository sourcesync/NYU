/**
 * Class for organizing and drawing a set of strokes
 */

import org.json.*;


class StrokeSet {

  // Number of possible trails
  int maxObjects = 100;
  // Number of points for stroke
  int maxStrokes = 200;
  // maxObjects * maxStrokes should be less than 10,000,000

  int maxCircles = 100;
  int tail = 50;
  int weight = 100; //gw 500;
  float maxDist = 1000;
  PImage spot;

  HashMap objectLookup;
  int numObjects;
  int mode;
  int hue = 0;
  int[] numStrokes = new int[maxObjects];
  int[] startStroke = new int[maxObjects];
  PVector[][]strokeXYZ = new PVector[maxObjects][maxStrokes];
  color[][]strokeColor = new color[maxObjects][maxStrokes];

  PVector[]circles = new PVector[maxCircles];
  int[] circleRadii = new int[maxCircles];
  int[] circleHues = new int[maxCircles];
  PVector avgXY;
  int avgCount;
  int numCircles = 0;
  int radius = 500;
  int numScores = 4;
  float[] zScores = new float[numScores];
  float[] panScores = new float[numScores];

  
  int ceiling = 2500;
  int window = 60;
  int hitCount = 0;
  int shrink = 50;
  int oldHue;
  float fade = 1;
  float xzRatio = 1.0;
  boolean hit = false;
  float maxZZ;
  float rx = 0;
  float ry = 0;
  boolean tailMode = true;

  StrokeSet(PApplet parent) {
    objectLookup = new HashMap();
    objectLookup.put("gw_big_black",-2);
    objectLookup.put("gw_big_clear",-1);
    objectLookup.put("gw_big_yellow",42);
    
    objectLookup.put("MB_Black",230);
    objectLookup.put("MB_Trans",-1);
    objectLookup.put("MB_White",-1);
    objectLookup.put("MB_Yellow",42);

    objectLookup.put("MB_Blue",130);
    objectLookup.put("MB_Red",0);
    objectLookup.put("MB_Green",80);

  }




 void draw2() {






    stroke(255);



    stroke(weight);

    for (int j = 0; j < numObjects; j++) {

      PVector[] s = strokeXYZ[j];
      
      if (mode == 1) {
        beginShape();
      } 
      else {
        beginShape(POINTS);
      }  
      
      int prevII = -1;
      

     
      
      for (int i = 0; i < numStrokes[j]; i++) {
        int alpha = 128 + (int)(128 * (float)i / (float)numStrokes[j]);


        int ii = (i + startStroke[j]) % maxStrokes;




        if (mode == 1 && prevII != -1) {
          
          
          float dx = pow((s[ii].x -s[prevII].x),2);
          float dy = pow((s[ii].y -s[prevII].y),2);
          float dz = pow((s[ii].z -s[prevII].z),2);
          float distSq = dx+dy+dz;
          if (distSq > maxDist*maxDist) {
           endShape();
           beginShape(); 
          }
          
        }

         strokeWeight(weight);
         stroke(255);
         
         vertex(s[ii].x*xzRatio,s[ii].y,s[ii].z);
         

         
         prevII = ii; 
      }

      endShape();
    }
    
    
    
    
  }






  void draw() {


   // draw2();
    

    

    for (int j = 0; j < numObjects; j++) {

      PVector[] s = strokeXYZ[j];
      
      int prevII = -1;
     
     

       int startI = numStrokes[j] - tail;
       if (startI < 0) startI = 0;
     
      for (int i = startI; i < numStrokes[j]; i++) {
        int alpha = 128 + (int)(128 * (float)i / (float)numStrokes[j]);


        int ii = (i + startStroke[j]) % maxStrokes;


         pushMatrix();
         rotateX(HALF_PI);
         translate(s[ii].x*xzRatio,s[ii].z,0);
     
         //blend(spot, 0, 0, 128, 128, 0, 0, 128, 128, ADD);
         //image(spot,0,0);  
         
         fill(255);
         //stroke(255);
         //strokeWeight(10);
         //stroke(0,255,255,255);


         rotateY(-rx);
         rotateX(HALF_PI-ry);
         fill(255); 
         strokeWeight(0);
         if (i == numStrokes[j] - 1) {
           translate(0,0,10);
                    strokeWeight(5);
           stroke(255,0,0);
         }
         ellipse(0,0,weight,weight);
         popMatrix();
         
         prevII = ii; 
      }


    }
    
    
    
    
  }



  void addStrokesFromJSON(String message) {

    if (! message.substring(0,1).equals("{")) {
      return;
    }
    try {
      JSONObject vp = new JSONObject(message);
      JSONArray objs = vp.getJSONArray("objs");
      mode = vp.getInt("mode");
      addLoopingStrokes(objs);
    }
    catch (JSONException e) { 
      println (e.toString()); 
      println( "Original Message: \""+message+"\"");
    }
  }


  
    // This function is a bit ugly. Strokes are kept in simple arrays
  // and when maxStrokes has been reached, they are written to the beginning
  // of the array. The startStroke array keeps track of where each stroke
  // starts so they can been drawn in that order.
  void addLoopingStrokes(JSONArray objs) {
    

    
    int zCount = 0;
    float zAvg = 0;
    float panAvg = 0;
   
    
    if (hit) {
      hitCount++;

      if (hitCount > window) {
        hitCount = 0;
        hit = false;
      }
    }

    try {
      for (int i=0; i<objs.length(); i++) {
        if (i >= maxObjects) return;


        JSONArray xyz;
        int fixedHue=-1;

        if (mode == 1) {
          JSONObject single = objs.getJSONObject(i);
          xyz = single.getJSONArray("t");
          String objName = single.getString("name");
          Boolean oc = single.getBoolean("oc");
          if (oc) continue;
          Object objectHue = objectLookup.get(objName);
          if (objectHue == null) continue;
          fixedHue = ((Number)objectHue).intValue();
        } 
        else {
          xyz = objs.getJSONArray(i);
          System.out.println(xyz);
        }

    //gw
        float xx = (float)xyz.getDouble(0) * 400.0;
        //float yy = 0;//(float)xyz.getDouble(1);
        //float zz = (float)xyz.getDouble(2);
        float yy = 0;
        float zz = (float)xyz.getDouble(1) * 400.0;
     //gw
        
        //if (zz < 0) continue;

        int newPosition = numStrokes[i];
        
        if (!tailMode) {
           newPosition = 0;
           numStrokes[i] = 1; 
        } else {
        if (numStrokes[i] >= maxStrokes) {
          newPosition = startStroke[i];
          startStroke[i] = startStroke[i] + 1;
          if (startStroke[i] >= maxStrokes) startStroke[i] = 0;
        }
        else {
          numStrokes[i]++;
        }
        }

      
        


        zCount++;
        zAvg+= zz;
        panAvg+= xx;

        if (zz > ceiling && hit) {
          maxZZ = max(zz, maxZZ);
        }

        if (zz > ceiling && !hit) {
          oldHue = hue;
          maxZZ = ceiling;
          hue+=43;
          hue=hue%255;
          if (numCircles >= maxCircles) numCircles = 0;
          circles[numCircles] = new PVector(xx,yy,zz);
          circleHues[numCircles] = hue;
          circleRadii[numCircles] = radius;
          numCircles++;
          hit = true;
        }


        strokeXYZ[i][newPosition] = new PVector(xx, yy, zz);

        if (hit && zz >= maxZZ) {
          circles[numCircles-1].z = maxZZ;
          strokeColor[i][newPosition] = color(oldHue + i*3,255,255);
        } 
        else {
          strokeColor[i][newPosition] = color(hue + i*3,255,255);
        }

        if (mode == 1)
          if (fixedHue == -1) {
            strokeColor[i][newPosition] = color(0,0,255);
          } else if (fixedHue == -2) {
            strokeColor[i][newPosition] = color(0,0,128);
          }
          else {    
            strokeColor[i][newPosition] = color(fixedHue,255,255);
          }
        numObjects = max(numObjects,i+1);
       numObjects = i+1;
      }
    }  
    catch (JSONException e) { 
      println (e.toString());
    }


    
      if (zCount != 0) {
        zAvg /= zCount;
        panAvg /= zCount;
      }
      
      panAvg /= 4000;
      zAvg = (zAvg - 500)/ceiling;
      
      
      panAvg = constrain(panAvg, -1, 1);
      zAvg = constrain(zAvg, 0, 1);

  }
  
  
  
  
  
  
  
  
  


  // This is a simpler version that loops around to start back at
  // the beginning of each stroke array but doesn't care about getting
  // the starting position correct. There may be flickering artifacts.
  void addSimpleStrokes(JSONArray objs) {

    try {
      for (int i=0; i<objs.length(); i++) {
        if (i >= maxObjects) return;
        int newPosition = numStrokes[i];

        if (numStrokes[i] >= maxStrokes) {
          newPosition = 0;
        } 
        else {
          numStrokes[i]++;
        }

        JSONArray xyz = objs.getJSONArray(i);

        float xx = (float)xyz.getDouble(0);
        float yy = (float)xyz.getDouble(1);
        float zz = (float)xyz.getDouble(2);
        strokeXYZ[i][newPosition] = new PVector(xx, yy, zz);
        numObjects = i+1;
      }
    }
    catch (JSONException e) { 
      println (e.toString());
    }
  }

  void clearStrokes() {

    for (int i = 0; i < maxObjects; i++) {
      numStrokes[i] = 0;
      startStroke[i] = 0;
    }
    numObjects = 0;
  }


  void stop() {

  }
}

