/**
 * Brick Object
 */


class Brick {
  int state;
  // States
  // 0 = destroyed
  // 1 = intact
  
  int i, j, x, y, w, h;
  int hue;
  int pad = 2;
  Boolean boundary = false;
  PImage bg;
   
  Brick(int i, int j, int w, int h, int hue) {
    this.i = i;
    this.j = j;
    this.x = i * w;
    this.y = j * h;
    this.w = w - pad;
    this.h = h - pad;
    this.hue = hue;
    this.state = 1;
  }

  void clear() {
     this.state = 1;
  }

  void loadBackground(PImage p) {
    bg = p.get(x, y, w, h); 
  }
  

  Boolean check(int x, int y, float r) {
        
    float deltaX = abs((this.x + (this.w / 2)) - x);
    float deltaY = abs((this.y + (this.h / 2)) - y);
   
    if (deltaX > (this.w/2 + r)) { return false; }
    if (deltaY > (this.h/2 + r)) { return false; }
    
    //cornerDistance_sq = (deltaX - this.w/2)^2 +
    //                    (deltaY - this.h/2)^2;
    //if (cornerDistance_sq > (circle.r^2)) return false;

    this.state = 0;
    return true;
  }

  void draw() {

    if (this.bg != null) {
     image(this.bg, x, y, w, h); 
    }
    
    stroke(255, 0, 255, 128);
    
    if (this.boundary) {
      stroke(255, 255, 255, 255); 
    }
    
    if (this.state == 1) {
      fill(this.hue, 255, 255, 100);
    } else {
      fill(this.hue, 0, 255, 100);
    }
    rect(x, y, w, h); 
  }
}

