

import pbox2d.*;
import org.jbox2d.collision.shapes.*;
import org.jbox2d.common.*;
import org.jbox2d.dynamics.*;
import java.io.*;
import org.json.*;

BufferedReader br;
color col;
String fname = "/Users/gwilliams/Projects/NYU/sourcecontrol/NYU/Nashville/Processing/Simulator/Data/three_balls_z_0.05.txt";
void setup() 
{
  size(640,360);
  smooth();

  try  
  {
    
	FileInputStream reader = new FileInputStream(fname); 
	DataInputStream in = new DataInputStream(reader);
	br = new BufferedReader(new InputStreamReader(in));
  }
  catch (Exception e)
  {
      System.err.println("Error: " + e.getMessage());
  }
}


void draw() 
{
	background(255);
	try  
  	{
        	String strLine = br.readLine();
        	if (strLine != null)   
        	{
                	System.out.println (strLine);

			try 
			{
				JSONObject data = new JSONObject( strLine  );
				JSONArray objs = data.getJSONArray("objs");
                                for (int i=0;i<objs.length();i++)
                                {
                                  JSONArray vec = objs.getJSONArray(i);
                                  double xx = 640/2.0 + ( (640/2.0)*vec.getDouble(0) );
                                  double yy = 360.0 - 360.0*vec.getDouble(1);
                                  double zz = vec.getDouble(2);
                                  //System.out.println(xx + " " + yy + " " + zz);
                                  
                                  pushMatrix();
                                  translate( (int)xx, (int)yy );
                                  //rotate(a);
                                  col = color(255, 0, 0);
                                  fill(col);
                                  stroke(0);
                                  strokeWeight(1);
                                  int r = 20;
                                  ellipse(0, 0, r*2, r*2);
                                  // Let's add a line so we can see the rotation
                                  line(0, 0, r, 0);
                                  popMatrix();
                                  
                                }
				//int total = nytData.getInt("total");
				//println ("There were " + total + " occurences of the term O.J. Simpson in 1994 & 1995");
			}
			catch (JSONException e) 
			{
				println ("There was an error parsing the JSONObject.");
			};
        	}
  	}
  	catch (Exception e)
  	{
      		System.err.println("Error: " + e.getMessage());
  	}
}
