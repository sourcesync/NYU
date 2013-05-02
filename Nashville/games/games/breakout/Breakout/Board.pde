/**
 * Class for organizing and drawing bricks
 */

import java.io.*;


class Board {


  int rowHeight = height / 20;
  int cols = 10;
  int rows = 5;
  Brick[][] grid;
  int pad = 2;
  PImage bg1;
  PImage bg2;
  AudioSnippet hit;
  Boolean superMode = false;
 
  ArrayList bricks;
  ArrayList logos;
  int logo = 0;

  Board() {
     hit = minim.loadSnippet("hit.wav");
     getLogoList();
     bg1 = getBackground(0, width, rowHeight * rows);
     initBricks();
  }

  
  void initBricks() {
    frame = 0;
    grid = new Brick[cols][rows];
    bricks = new ArrayList();
    int colWidth = width / cols;

     for (int j = 0; j < rows; j++) {
      int blockHue = j * (255 / rows);
      for (int i = 0; i < cols; i++) {
        int brickId = j * cols + i;
        Brick brick = new Brick(i, j, colWidth, rowHeight, blockHue);
        brick.loadBackground(bg1);
        bricks.add(brick);
        grid[i][j] = brick;
      }
    }  
  
  }
  
  
  void updateBoundary() {
     for (int i = 0; i < cols; i++) {
      for (int j = 0; j < rows; j++) {
          Brick b = grid[i][rows-j-1];
           if (b != null) {
             b.boundary = true;
             break; 
           }
      }
    }
  }
  
  int frame = 0;
  
  void checkBricks(float[][] locations, float ballRadius) {
    
    if (frame % 100 == 0) updateBoundary();
    frame++;
    
    
    for (int b = 0; b < bricks.size(); b++) {
      Brick brick = (Brick)bricks.get(b);
      
      if (!this.superMode && !brick.boundary) continue;
      //brick.clear();
      
      for (int i = 0; i < locations.length; i++) {
      
        float x1 = (locations[i][0] + 1) / 2;
        int x = (int)(x1 * width);
        int y = (int)(locations[i][1] * height);
        if (brick.check(x, y, ballRadius)) {
          if (!hit.isPlaying()) hit.play(0);  
        } 
        
      }
      
      if (brick.state == 0) {
       bricks.remove(brick);
       grid[brick.i][brick.j] = null;
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
    logo++;
    if (logo >= logos.size()) logo = 0;
    bg1 = getBackground(logo, width, rowHeight * rows);
    initBricks();
  }
  
  PImage getBackground(int n, int w, int h) {
        
    PImage bg = new PImage(w, h);
    PImage white = new PImage(1,1);
    white.set(0, 0, color(128, 0, 255, 255));   
    bg.copy(white, 0, 0, 1, 1, 0, 0, w, h);
    
    PImage logo = loadImage("logos/" + logos.get(n));
    float logoAspect = (float)logo.width / (float)logo.height;
    float bgAspect = (float)w / (float)h;
    
    int dx = 0;
    int dy = 0;
    int dw = w;
    int dh = h;
    
    if (bgAspect > logoAspect) {
      dw = (int)(dh * logoAspect);
      dx = (w - dw) / 2;
    } else {
      dh = (int)(dw / logoAspect);
      dy = (h - dh) / 2;
    }
    
    bg.copy(logo, 0, 0, logo.width, logo.height, dx, dy, dw, dh);
    return bg;
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

