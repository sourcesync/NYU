////////////////////////////////////////////////////////////////////////////////////////////
// CL-Eye Platform SDK Example
//
// Actionscript 3.0 - Adobe Flash
//
// This library allows you to use multiple PS3Eye cameras in your own applications.
//
// For updates, more information and downloads visit: eye.codelaboratories.com
//
// Copyright 2009 (c) Code Laboratories, Inc.  All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////////////////

package
{
    import flash.display.Bitmap;
    import flash.display.BitmapData;
    import flash.display.Graphics;
    import flash.display.GradientType;
    import flash.display.Shape;
    import flash.display.Sprite;
    import flash.events.Event;
    import flash.geom.Matrix;
    import flash.media.Camera;
    import flash.media.Video;
	import flash.events.*;
	import flash.media.Camera;
	import flash.media.Video;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFieldAutoSize;
	
    //import com.flashdynamix.utils.SWFProfiler;

    [SWF(width=500, height=500, frameRate=60, backgroundColor=0x000000)]

    /**
     *
     */
    public class Main extends Sprite
    {
        public static const CAMERA_WIDTH:Number = 320;
        public static const CAMERA_HEIGHT:Number = 240;
		public static const FRAME_RATE:Number = 60;
        public static const REFLECT_SCALE:Number = .5;
        public static const REFLECT_MARGIN:Number = 0;
        
		private var n_text:TextField;
        private var video:Video;
        private var container:Sprite;
        private var film:BitmapData;
        private var camera:Camera;
        /**
         *
         */
        public function Main()
        {
            addEventListener(Event.ADDED_TO_STAGE, initialize);
        }

        /**
         *
         */
        private function initialize(evt:Event):void
        {
            removeEventListener(Event.ADDED_TO_STAGE, initialize);

           // SWFProfiler.init(this);

            container = new Sprite();

            var original:Bitmap,
                reflect:Bitmap,
                msk:Shape,
                mtx:Matrix,
                g:Graphics;

            //  get default web camera.
            camera = Camera.getCamera();

            //  stop process if couldn't found any camera.
            if (!camera) return;

            //  setup camera.
            camera.setMode(CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_RATE);

            //  create and setup video.
            video = new Video(CAMERA_WIDTH, CAMERA_HEIGHT);
            //  reverse and fix position.
            video.scaleX *= -1;
            video.x += CAMERA_WIDTH;
            //  bind camera.
            video.attachCamera(camera);

            //  create bitmapdata as a snapshot film.
            film = new BitmapData(CAMERA_WIDTH, CAMERA_HEIGHT, true, 0);

            //  create original bitmap.
            original = new Bitmap(film);
            container.addChild(original);

            //  create reflection bitmap.
            reflect = new Bitmap(film);
            //  reverse and fix position.
            reflect.scaleY *= -1;
            reflect.y = REFLECT_MARGIN + (CAMERA_HEIGHT << 1);
            //  cache for mask and fade.
            reflect.cacheAsBitmap = true;
            container.addChild(reflect);

            //  create mask shape.
            msk = new Shape();
            //  fix position.
            msk.y = CAMERA_HEIGHT + REFLECT_MARGIN;
            //  cache for mask and fade.
            msk.cacheAsBitmap = true;

            //  create matrix.
            mtx = new Matrix();
            mtx.createGradientBox(CAMERA_WIDTH, CAMERA_HEIGHT, Math.PI/2);

            //  draw gradient box.
            g = msk.graphics;
            g.beginGradientFill(
                GradientType.LINEAR,
                [0x00, 0x00],
                [REFLECT_SCALE, 0],
                [0, 255 * REFLECT_SCALE],
                mtx
            );
            g.drawRect(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
            g.endFill();

            container.addChild(msk);
            reflect.mask = msk;

            //  fix position.
            container.x = (stage.stageWidth - original.width) >> 1;
            container.y = (stage.stageHeight - original.height) >> 1;
            addChild(container);

			addKeyboardShortcuts();
			initLabel();
			
            addEventListener(Event.ENTER_FRAME, step);
        }
        /**
		 * Creates a label for framerate and stuff...
		 */
		public function initLabel():void
		{				
			
			n_text = new TextField();					
			var format:TextFormat = new TextFormat();
			format.font="Arial";
			format.color = 0xFFFFFF;
			format.size = 10;		
			n_text.width=500;
			n_text.height=500;
			n_text.x=10;
			n_text.y=10;
			//var count:int = 0;	
				
			//n_text.autoSize = TextFieldAutoSize.CENTER;
			n_text.selectable = false;
			n_text.defaultTextFormat = format;
			//n_text.embedFonts = true;
			n_text.text = "Test";
			//n_text.blendMode = 'invert';
			//n_text.alpha = 0;	
			container.addChild( n_text );
			
		}
		/**
		 * Add keyboard linkage event down
		 */
		protected function addKeyboardShortcuts():void 
		{
			stage.addEventListener( KeyboardEvent.KEY_DOWN, keyDownHandler );
		}
        /**
		 * Add keyboard linkage event space key
		 */
		protected function keyDownHandler(event:KeyboardEvent):void 
		{
			switch(event.keyCode) 
			{
				case 32:
					takeScreenShot();
					break;
			}
			
			//For debug right now
			trace( "keyDownHandler: " + event.keyCode );
			
		}
		/**
		 * Performs screenshot action. 
		 */
		private function takeScreenShot():void
		{
			//cameraBitmap.bitmapData.draw(video);
			if(n_text.visible){
		     n_text.visible=false;
			}else{
		    n_text.visible=true;
			}
		}
		/**
		 * Updates the current camera effect
		 */
		private var showText:Boolean = true;
        private function step(evt:Event):void
        {
            film.lock();
            film.fillRect(film.rect, 0);
            film.draw(video, video.transform.matrix);
            film.unlock();
			
			//effects.update();		
  			
			//if (!camera) return;
		
				
			n_text.x=10;
			n_text.y=10;	
			n_text.text="";
			//n_text.appendText("Index: "+camera.index+"\n");
			n_text.appendText("Name: "+camera.name+"\n");
			n_text.appendText("FPS: "+camera.currentFPS+"\n");
			n_text.appendText("Resolution: "+camera.width+"x"+camera.height+"\n");
			n_text.appendText("Quality: "+camera.quality+"\n");
			//n_text.appendText("Activity: "+camera.activityLevel+"\n");
			//n_text.appendText("Bandwidth: "+camera.bandwidth+"\n");
			//n_text.appendText("Loopback: "+camera.loopback+"\n");
			//n_text.appendText("Motion Level: "+camera.motionLevel+"\n");
			//n_text.appendText("Motion Timeout: "+camera.motionTimeout+"\n");
			//n_text.appendText("Muted: "+camera.muted+"\n");	
			//n_text.appendText("\n\n Spacebar to take screenshot");	
			/**/
		
        }
    }
}

