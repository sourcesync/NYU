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
	import flash.display.Sprite;
	import flash.display.StageAlign;
	import flash.display.StageScaleMode;
	import flash.events.*;
	import flash.media.Camera;
	import flash.media.Video;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFieldAutoSize;	
	
	public class Main extends Sprite 
	{

		private var camera:Camera;
		private var video:Video;
		private var cameraBitmap:Bitmap;
		private var n_text:TextField;
		
		/**
		 * Main Constructor
		 */
		public function Main()
		{
			stage.scaleMode = StageScaleMode.NO_SCALE;
			stage.align = StageAlign.TOP_LEFT;	
			
			camera = Camera.getCamera( );
			camera.setMode( 320, 240, 60 );
			video = new Video( camera.width, camera.height );
			video.attachCamera( camera );
			addChild( video );
			
			cameraBitmap = new Bitmap( new BitmapData( camera.width, camera.height, false, 0x000000 ) );
			cameraBitmap.y = video.height + 2;
			addChild( cameraBitmap );
			
			addKeyboardShortcuts();
			initLabel();
			addEventListener(Event.ENTER_FRAME, update);
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
			n_text.text = "";
			//n_text.blendMode = 'invert';
			//n_text.alpha = 0;	
			this.addChild( n_text );
			
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
			cameraBitmap.bitmapData.draw(video);
		}
		/**
		 * Updates the current camera effect
		 */
		private function update(e:Event):void
		{
			//effects.update();			
			n_text.x=10;
			n_text.y=10;	
			n_text.text="";
			n_text.appendText("Index: "+camera.index+"\n");
			n_text.appendText("Name: "+camera.name+"\n");
			n_text.appendText("FPS: "+camera.currentFPS+"\n");
			n_text.appendText("Resolution: "+camera.width+"x"+camera.height+"\n");
			n_text.appendText("Quality: "+camera.quality+"\n");
			n_text.appendText("Activity: "+camera.activityLevel+"\n");
			n_text.appendText("Bandwidth: "+camera.bandwidth+"\n");
			n_text.appendText("Loopback: "+camera.loopback+"\n");
			n_text.appendText("Motion Level: "+camera.motionLevel+"\n");
			n_text.appendText("Motion Timeout: "+camera.motionTimeout+"\n");
			n_text.appendText("Muted: "+camera.muted+"\n");	
			n_text.appendText("\n\n Spacebar to take screenshot");	
		}
	}
}