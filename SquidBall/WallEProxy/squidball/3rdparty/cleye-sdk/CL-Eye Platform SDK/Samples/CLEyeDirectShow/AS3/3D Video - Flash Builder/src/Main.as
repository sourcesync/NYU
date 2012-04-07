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
	import flash.display.*;
	import flash.events.*;
	import flash.media.*;
	
	[SWF(width = "500", height = "500", frameRate = "60", backgroundColor = "#FFFFFF")]

	public class Main extends Sprite
	{
		private const VIDEO_WIDTH  :int    = 320;
		private const VIDEO_HEIGHT :int    = 240;
		private const DECLARATION  :Number = .025;
		
		
		private var wrap  :Sprite;
		private var video :Video;
		private var cameraBitmap:Bitmap;
		
		public function Main():void
		{
			stage.scaleMode = StageScaleMode.NO_SCALE;
			stage.align = StageAlign.TOP_LEFT;
	
			wrap = new Sprite();
			addChild(wrap);

			var camera:Camera = Camera.getCamera();
			camera.setMode( 320, 240, 60 );
			
			video = new Video(VIDEO_WIDTH, VIDEO_HEIGHT);
			video.x = - VIDEO_WIDTH  / 2;
			video.y = - VIDEO_HEIGHT / 2;
			video.attachCamera(camera);
			wrap.addChild(video);
			this.addEventListener(Event.ENTER_FRAME, enterFrameHandler);
		}
				
		private function enterFrameHandler(event:Event):void
		{
			wrap.rotationX += (mouseY/stage.stageHeight * 720 - wrap.rotationX) * DECLARATION;
			wrap.rotationY += (mouseX/stage.stageWidth  * 720 - wrap.rotationY) * DECLARATION;
			wrap.x = stage.stageWidth  / 2;
			wrap.y = stage.stageHeight / 2;
		}		
	}
}