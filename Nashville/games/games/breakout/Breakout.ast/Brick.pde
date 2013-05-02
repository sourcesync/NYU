/**
 * Brick Object
 */



class Brick {

  int x;
  int y;
  int w;
  int h;
  int hue;
  int state;
  PImage bg;
   
  Brick(int x, int y, int w, int h, int hue) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.hue = hue;
    this.state = 1;
  }


  void clear() {
     this.state = 1;
  }

  void loadBackground(PImage p) {
    bg = p.get(x, y, w, h); 
  }
  

  void check(int x, int y) {
     
    float deltaX = abs((this.x + (this.w / 2)) - x);
    float deltaY = abs((this.y + (this.h / 2)) - y);
        
    if (deltaX < 40 && deltaY < 20) {
      this.state = 0; 
    }
    
  }

  void draw() {

    if (this.bg != null) {
     image(this.bg, x, y, w, h); 
    }
    
    stroke(255, 0, 255, 128);
    if (this.state == 1) {
      fill(this.hue, 255, 255, 100);
    } else {
      fill(this.hue, 0, 255, 100);
    }
    rect(x, y, w, h); 
  }
}

