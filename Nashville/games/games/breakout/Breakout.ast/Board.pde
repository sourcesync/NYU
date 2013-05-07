/**
 * Class for organizing and drawing bricks
 */

import java.io.*;


class Board {


  int rowHeight = 20;
  int cols = 20;
  int rows = 10;
  int pad = 2;
  int level = 0;
  PImage bg;
 
  ArrayList bricks;
  ArrayList logos;

  Board() {
    
     getLogoList();
     loadBackground(0);
     initBricks();
  }

  
  void initBricks() {
    
    bricks = new ArrayList();
    int colWidth = width / cols;

     for (int r = 0; r < rows; r++) {
      int blockHue = r * (255 / rows);
      for (int c = 0; c < cols; c++) {
        int brickId = r * cols + c;
        int x = c * colWidth;
        int y = r * rowHeight;
        Brick brick = new Brick(x, y, colWidth - pad, rowHeight - pad, blockHue);
        brick.loadBackground(bg);
        bricks.add(brick);
      }
    }  
  }
  
  
  void checkBricks(float[][] locations) {
    
    for (int b = 0; b < bricks.size(); b++) {
      Brick brick = (Brick)bricks.get(b);
      brick.clear();
      
      for (int i = 0; i < locations.length; i++) {
      
        float x1 = (locations[i][0] + 1) / 2;
        int x = (int)(x1 * width);
        int y = (int)(locations[i][1] * height);
        brick.check(x, y); 
        
      }
      
      if (brick.state == 0) {
       bricks.remove(brick); 
      }
      
    }
  }



  void draw() {

    for (int b = 0; b < bricks.size(); b++) {
      Brick brick = (Brick)bricks.get(b);
      brick.draw(); 
    }
  }
  
  void next() {
    level++;
    if (level >= logos.size()) level = 0;
    loadBackground(level);
    initBricks();
  }
  
  void loadBackground(int n) {
        
    bg = new PImage(width, rowHeight * rows);
    PImage white = new PImage(1,1);
    white.set(0, 0, color(128, 0, 255, 255));   
    bg.copy(white, 0, 0, 1, 1, 0, 0, bg.width, bg.height);
    
    PImage logo = loadImage("logos/" + logos.get(n));
    float logoAspect = (float)logo.width / (float)logo.height;
    float bgAspect = (float)bg.width / (float)bg.height;
    
    int dx = 0;
    int dy = 0;
    int dw = bg.width;
    int dh = bg.height;
    
    if (bgAspect > logoAspect) {
      dw = (int)(dh * logoAspect);
      dx = (bg.width - dw) / 2;
    } else {
      dh = (int)(dw / logoAspect);
      dy = (bg.height - dh) / 2;
    }
    
    bg.copy(logo, 0, 0, logo.width, logo.height, dx, dy, dw, dh);
  }
  
  void getLogoList() {
    logos = new ArrayList();
    File file = new File(sketchPath + "/logos");
    String names[] = file.list();
    for (int i = 0; i < names.length; i++) {
      logos.add(names[i]);
    }
   } 
  
  
}

