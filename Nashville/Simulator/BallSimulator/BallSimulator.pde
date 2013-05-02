// The Nature of Code
// <http://www.shiffman.net/teaching/nature>
// Spring 2011
// PBox2D example

// Basic example of falling rectangles

import pbox2d.*;
import org.jbox2d.collision.shapes.*;
import org.jbox2d.common.*;
import org.jbox2d.dynamics.*;
import java.io.*;

// A reference to our box2d world
PBox2D box2d;

// A list we'll use to track fixed objects
ArrayList<Boundary> boundaries;
// A list for all of our rectangles
ArrayList<Box> boxes;

// An ArrayList of particles that will fall on the surface
ArrayList<Particle> particles;

BufferedWriter out;
 
void setup() 
{
  size(640,360);
  smooth();

  // Initialize box2d physics and create the world
  box2d = new PBox2D(this);
  box2d.createWorld();
  // We are setting a custom gravity
  box2d.setGravity(0, -20);

  // Create ArrayLists	
  boxes = new ArrayList<Box>();
  boundaries = new ArrayList<Boundary>();

  // Create the empty list
  particles = new ArrayList<Particle>();
  
  // Add a bunch of fixed boundaries
  //boundaries.add(new Boundary(width/4,height-5,width/2-100,10));
  //boundaries.add(new Boundary(3*width/4,height-5,width/2-100,10));
  boundaries.add(new Boundary(0,height-5,width*2,10));
  boundaries.add(new Boundary(width-5,height/2,10,height));
  boundaries.add(new Boundary(5,height/2,10,height));
  
  System.out.println( width + " " + height );
  
  if (true)
  {
    
    float sz = random(4, 8);
    particles.add( new Particle(width/2,20,sz) ); //  new Particle(random(width), 20, sz));
    
    particles.add( new Particle( ( width/2 - width/4),20,sz) ); //  new Particle(random(width), 20, sz));
    
    
    particles.add( new Particle( (width/2 + width/4), 20,sz) ); //  new Particle(random(width), 20, sz));
  }
  
  try  
  {
      FileWriter fstream = new FileWriter("/tmp/out.txt", false); //true tells to append data.
      out = new BufferedWriter(fstream);
      //out.write("\nsue");
      //out.close();
  }
  catch (Exception e)
  {
      System.err.println("Error: " + e.getMessage());
  }
  
  
}

void draw() 
{
  background(255);

  // We must always step through time!
  box2d.step();

  // Compute coordinates...
  Particle pp = (Particle)particles.get(0);
  Vec2 pos = pp.body.getWorldCenter();
  //System.out.println( pos.x + " "+ pos.y );
  float xx = pos.x/30.0;
  float yy = (16.0 + pos.y)/32.0;
  float zz = 0.05;
  //System.out.println( xx + " "+ yy );
  
  String jsonline = "{\"v\":1,\"fno\":0,\"objno\":0,\"mode\":0,\"objs\":[";
  for ( int j=0; j< particles.size(); j++)
  {
    Particle ppp = (Particle)particles.get(j);
    Vec2 posl = ppp.body.getWorldCenter();
  //System.out.println( pos.x + " "+ pos.y );
  float xxx = posl.x/30.0;
  float yyy = (16.0 + posl.y)/32.0;
  float zzz = 0.05;
    jsonline += "[" + xxx + "," + yyy +"," + zzz +" ]";
    if ( j<( particles.size()-1))
      jsonline +=",";
  }
  jsonline += "]}";
    
  System.out.println( jsonline );
  
  
  try
    {
      out.write( jsonline + "\n");
    }
    catch (Exception e)
    {
    }
    
  // When the mouse is clicked, add a new Box object
  if (false)
  {
  //if (random(1) < 0.1) {
    Box b = new Box(random(width),10);
    boxes.add(b);
  }
  
  
  if (mousePressed) {
    for (Box b: boxes) {
     Vec2 wind = new Vec2(20,0);
     b.applyForce(wind);
    }
    
    for (Particle p: particles) {
     Vec2 wind = new Vec2(20,0);
     p.applyForce(wind);
    }
  }
  
  if (keyPressed)
  {
    System.out.println("quit");
    try
    {
      out.close();
    }
    catch (Exception e)
    {
    }
    exit();
  }
    
  

  // Display all the boundaries
  for (Boundary wall: boundaries) {
    wall.display();
  }

  // Display all the boxes
  for (Box b: boxes) {
    b.display();
  }
  
  
  // Look at all particles
  for (int i = particles.size()-1; i >= 0; i--) {
    Particle p = particles.get(i);
    p.display();
    // Particles that leave the screen, we delete them
    // (note they have to be deleted from both the box2d world and our list
    if (p.done()) {
      particles.remove(i);
    }
  }

  // Boxes that leave the screen, we delete them
  // (note they have to be deleted from both the box2d world and our list
  for (int i = boxes.size()-1; i >= 0; i--) {
    Box b = boxes.get(i);
    if (b.done()) {
      boxes.remove(i);
    }
  }
  
  fill(0);
  text("Click mouse to apply a wind force.",20,20);
}


