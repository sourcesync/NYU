#region BSD License
/*
 BSD License
Copyright (c) 2002, The CsGL Development Team
http://csgl.sourceforge.net/authors.html
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of The CsGL Development Team nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
 */
#endregion BSD License

using CsGL.OpenGL;
using CsGL.Util;
using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace CsGL.Basecode {
	/// <summary>
	/// Provides static (Shared in Visual Basic) methods and properties to manage a CsGL application.  
	/// It provides methods and properties to start and stop a looped CsGL application and to manage 
	/// its runtime state.  This class cannot be inherited.
	/// </summary>
	/// <remarks>
	/// <para>
	/// The <b>App</b> class has methods to start and stop CsGL applications and to provide useful 
	/// CsGL application functionality.
	/// </para>
	/// <para>
	/// Call <see cref="Run" /> to start a CsGL application using the appropriate inherited 
	/// <see cref="Model" />.  Call <see cref="Terminate" /> to forcibly end a running CsGL 
	/// application.  Calling <see cref="ToggleFullscreen" /> toggles between your supplied 
	/// Windows Form and the fullscreen form.  Call <see cref="ResetFramerate" /> to reset 
	/// the application's current framerate calculations.
	/// </para>
	/// <para>
	/// This class has numerous properties to get or set information about the current state of the 
	/// application.
	/// </para>
	/// <para>
	/// You cannot create an instance of this class.
	/// </para>
	/// </remarks>
	public sealed class App {
		// --- Fields ---
		#region Private Fields
		// Application Objects
		private readonly static App instance = new App();								// Only One Instance Of This Class (Singleton)
		private static Form form;														// Application's Current Form
		private static View view;														// Application's Current View
		private static Model model;														// Application's Model
		private static HighResolutionTimer timer = new HighResolutionTimer();			// Application's High Resolution Timer

		// Application State
		private static bool isDisposed = false;											// Has Dispose Been Called?
		private static bool isFullscreen = false;										// Is Application In Fullscreen Or Windowed Mode?
		private static bool isRunning = true;											// Is Application Currently Running?
		private static bool isActive = true;											// Is The Application Active?

		private static bool runWhileMinimized = true;									// Continue Running The Application's Main Loop Even When Application Is Minimized?
		
		private static bool showCursorFullscreen = false;								// Show Mouse Cursor While Application Is Fullscreen?
		private static bool showCursorWindowed = true;									// Show Mouse Cursor While Application Is Windowed?
		private static bool showStatusBar = true;										// Show The Basecode Status Bar While In Windowed Mode?

		// OpenGL State
		private static int width = 640;													// The Current Width Of The Application.
		private static int height = 480;												// The Current Height Of The Application.
		private static byte colorDepth = 16;											// The Current Color Depth, In Bits Per Pixel.

		private static bool isRgba = true;												// RGBA Or Color Index Mode?

		private static byte accumDepth = 0;												// OpenGL's Accumulation Buffer Depth, In Bits Per Pixel.
		private static byte stencilDepth = 0;											// OpenGL's Stencil Buffer Depth, In Bits Per Pixel.
		private static byte zDepth = 16;												// OpenGL's Z-Buffer Depth, In Bits Per Pixel.
		
		private static double nearClippingPlane = 0.1f;									// GLU's Distance From The Viewer To The Near Clipping Plane (Always Positive).
		private static double farClippingPlane = 100.0f;								// GLU's Distance From The Viewer To The Far Clipping Plane (Always Positive).
		private static double fovY = 45.0f;												// GLU's Field Of View Angle, In Degrees, In The Y Direction.

		// Framerate Related State
		private static bool framerateReady = false;										// Do We Have A Calculated Framerate?
		private static ulong timerFrequency = timer.Frequency;							// The Frequency Of The Timer
		private static ulong currentFrameTime;											// The Current Frame Time
		private static ulong lastCalculationTime;										// The Last Time We Calculated Framerate
		private static ulong framesDrawn;												// Frames Drawn Counter For FPS Calculations
		private static double currentFramerate;											// Current FPS
		private static double highestFramerate;											// Highest FPS
		private static double lowestFramerate = 999999999;								// Lowest FPS
		#endregion Private Fields

		#region Public Properties
		/// <summary>
		/// Gets or sets OpenGL's accumulation buffer depth, in bits per pixel.
		/// </summary>
		public static byte AccumDepth {
			get {
				return accumDepth;
			}
			set {
				accumDepth = value;
			}
		}

		/// <summary>
		/// Gets or sets the current color depth, in bits per pixel.
		/// </summary>
		public static byte ColorDepth {
			get {
				return colorDepth;
			}
			set {
				colorDepth = value;
			}
		}

		/// <summary>
		/// Gets the current framerate, in frames per second.
		/// </summary>
		public static double CurrentFramerate {
			get {
				return currentFramerate;
			}
		}

		/// <summary>
		/// Gets the frametime, in ticks, when the current frame began drawing.
		/// </summary>
		public static ulong CurrentFrameTime {
			get {
				return currentFrameTime;
			}
		}

		/// <summary>
		/// Gets or sets GLU's distance from the viewer to the far clipping plane (always positive).
		/// </summary>
		public static double FarClippingPlane {
			get {
				return farClippingPlane;
			}
			set {
				farClippingPlane = value;
			}
		}

		/// <summary>
		/// Gets or sets the application's form.  Should likely only be used in <see cref="CsGL.Basecode.Model.WindowsForm" />.
		/// </summary>
		public static Form Form {
			get {
				return form;
			}
			set {
				form = value;
			}
		}

		/// <summary>
		/// Gets or sets GLU's field of view angle, in degrees, in the Y direction.
		/// </summary>
		public static double FovY {
			get {
				return fovY;
			}
			set {
				fovY = value;
			}
		}

		/// <summary>
		/// Gets whether or not a framerate has been calculated.
		/// </summary>
		public static bool FramerateReady {
			get {
				return framerateReady;
			}
		}

		/// <summary>
		/// Gets or sets the height of the application's window at creation.
		/// </summary>
		public static int Height {
			get {
				return height;
			}
			set {
				height = value;
			}
		}

		/// <summary>
		/// Gets the highest framerate, in frames per second, recorded for the application's run.
		/// </summary>
		public static double HighestFramerate {
			get {
				return highestFramerate;
			}
		}

		/// <summary>
		/// Gets or sets the application's active state.  You could use this to make your application 
		/// idle while it's inactive by setting this property in your Windows Form's Activate and 
		/// Deactivated events.
		/// </summary>
		public static bool IsActive {
			get {
				return IsActive;
			}
			set {
				IsActive = value;
			}
		}

		/// <summary>
		/// Gets or sets whether the application is currently in fullscreen or windowed mode.  You
		/// should likely only use this to get, except if you define it in <see cref="Model.Setup" />.
		/// After the start of your application, use <see cref="App.ToggleFullscreen" />.
		/// </summary>
		public static bool IsFullscreen {
			get {
				return isFullscreen;
			}
			set {
				isFullscreen = value;
			}
		}

		/// <summary>
		/// Gets or sets whether the application uses RGBA or color index mode.
		/// </summary>
		public static bool IsRgba {
			get {
				return isRgba;
			}
			set {
				isRgba = value;
			}
		}

		/// <summary>
		/// Gets whether or not the application is currently running.
		/// </summary>
		public static bool IsRunning {
			get {
				return isRunning;
			}
		}

		/// <summary>
		/// Gets the lowest framerate, in frames per second, recorded for the application's run.
		/// </summary>
		public static double LowestFramerate {
			get {
				return lowestFramerate;
			}
		}

		/// <summary>
		/// Gets the application's <see cref="Model" />.
		/// </summary>
		public static Model Model {
			get {
				return model;
			}
		}

		/// <summary>
		/// Gets or sets GLU's distance from the viewer to the near clipping plane (always positive).
		/// </summary>
		public static double NearClippingPlane {
			get {
				return nearClippingPlane;
			}
			set {
				nearClippingPlane = value;
			}
		}

		/// <summary>
		/// Gets or sets whether the application should continue running the main loop even when 
		/// the application is minimized.  If it shouldn't, it'll wait until the application is put 
		/// back into normal mode, saving CPU cycles.
		/// </summary>
		public static bool RunWhileMinimized {
			get {
				return runWhileMinimized;
			}
			set {
				runWhileMinimized = value;
			}
		}

		/// <summary>
		/// Gets or sets whether the mouse cursor should be displayed in fullscreen mode.
		/// </summary>
		public static bool ShowCursorFullscreen {
			get {
				return showCursorFullscreen;
			}
			set {
				showCursorFullscreen = value;
			}
		}

		/// <summary>
		/// Gets or sets whether the mouse cursor should be displayed in windowed mode.
		/// </summary>
		public static bool ShowCursorWindowed {
			get {
				return showCursorWindowed;
			}
			set {
				showCursorWindowed = value;
			}
		}

		/// <summary>
		/// Gets or sets whether the StatusBar of the basecode's default windowed form 
		/// should be displayed.  If you're providing a custom form, you could use this 
		/// however you like.
		/// </summary>
		public static bool ShowStatusBar {
			get {
				return showStatusBar;
			}
			set {
				showStatusBar = value;
			}
		}

		/// <summary>
		/// Gets or sets OpenGL's stencil buffer depth, in bits per pixel.
		/// </summary>
		public static byte StencilDepth {
			get {
				return stencilDepth;
			}
			set {
				stencilDepth = value;
			}
		}

		/// <summary>
		/// Get's the application's <see cref="HighResolutionTimer" />.
		/// </summary>
		public static HighResolutionTimer Timer {
			get {
				return timer;
			}
		}

		/// <summary>
		/// Gets the application's current <see cref="View" />.
		/// </summary>
		public static View View {
			get {
				return view;
			}
		}

		/// <summary>
		/// Gets or sets the width of the application's window at creation.
		/// </summary>
		public static int Width {
			get {
				return width;
			}
			set {
				width = value;
			}
		}

		/// <summary>
		/// Gets or sets OpenGL's Z-buffer depth, in bits per pixel.
		/// </summary>
		public static byte ZDepth {
			get {
				return zDepth;
			}
		}
		#endregion Public Properties

		// --- Creation And Destruction Methods ---
		#region Constructor
		/// <summary>
		/// Contructor is private to prevent instantiation.  This is a Singleton class, only one 
		/// instance is allowed.
		/// </summary>
		private App() {
		}
		#endregion Constructor

		#region Dispose()
		/// <summary>
		/// Disposes of this class, since this is a Singleton class, we'll take care of it ourselves
		/// and not allow the user to call it themselves.
		/// </summary>
		private static void Dispose() {
			Dispose(true);																// We've Manually Called For A Dispose
			GC.SuppressFinalize(instance);												// Prevent Being Added To The Finalization Queue
		}
		#endregion Dispose()

		#region Dispose(bool disposing)
		/// <summary>
		/// Cleans up either unmanaged resources or managed and unmanaged resources.
		/// </summary>
		/// <remarks>
		/// <para>
		/// If disposing equals true, the method has been called directly or indirectly by a user's 
		/// code.  Managed and unmanaged resources can be disposed.
		/// </para>
		/// <para>
		/// If disposing equals false, the method has been called by the runtime from inside the 
		/// finalizer and you should not reference other objects.  Only unmanaged resources can 
		/// be disposed.
		/// </para>
		/// </remarks>
		/// <param name="disposing">Was Dispose called manually?</param>
		private static void Dispose(bool disposing) {
			if(!isDisposed) {															// Check To See If Dispose Has Already Been Called
				if(disposing) {															// If disposing Equals true, Dispose All Managed And Unmanaged Resources
					if(timer != null) {
						timer.Dispose();
					}
					if(form != null) {
						form.Close();
					}
					if(view != null) {
						view.Dispose();
					}
					if(model != null) {
						model.Dispose();
					}
				}

				// Release Any Unmanaged Resources Here, If disposing Was false, Only The Following Code Is Executed
				isRunning = false;
				timer = null;
				form = null;
				view = null;
				model = null;
			}
			isDisposed = true;															// Mark As disposed
		}
		#endregion Dispose(bool disposing)

		#region Finalizer
		/// <summary>
		/// This destructor will run only if the Dispose method does not get called.  It gives 
		/// the class the opportunity to finalize.  Simply calls Dispose(false).
		/// </summary>
		~App() {
			Dispose(false);																// We've Automatically Called For A Dispose
		}
		#endregion Finalizer

		// --- Private Methods ---
		#region CreateForm()
		/// <summary>
		/// Creates either a fullscreen form or the user-supplied windowed form, based on the
		/// <see cref="IsFullscreen" /> property.
		/// </summary>
		private static void CreateForm() {
			try {
				if(model != null) {
					view = new View(model);												// Create A New OpenGL View
					view.Context.Grab();												// Grab The OpenGL Context
					model.Initialize();													// Run Model's Initialize()
					OpenGLException.Assert();											// Check For Any OpenGL Errors

					if(isFullscreen) {													// If We're In Fullscreen Mode
						form = new ScreenForm(width, height);							// Create A New Fullscreen Form
						App.View.Dock = DockStyle.Fill;									// Fill It With view
						form.Controls.AddRange(new Control[] { view });					// Add view
						form.Show();													// Show The Fullscreen Form
					}
					else {																// Otherwise
						model.WindowsForm();											// Create The User's Defined Windows Form
					}
				}
				if(isFullscreen) {														// If We're In Fullscreen Mode
					if(showCursorFullscreen) {											// If We're Supposed To Show Cursor
						Cursor.Show();													// Show It
					}
					else {																// Otherwise
						Cursor.Hide();													// Hide It
					}
				}
				else {																	// Else We're In Windowed Mode
					if(showCursorWindowed) {											// If We're Supposed To Show Cursor
						Cursor.Show();													// Show It
					}
					else {																// Otherwise
						Cursor.Hide();													// Hide It
					}
				}
			}
			catch(Exception e) {
				// Handle Any Exceptions While Creating The Application's Form, Exit App
				string errorMsg = "A Basecode Error Occurred While Creating The Application's Form:\n\nStack Trace:\n\t" + e.StackTrace + "\n";
				MessageBox.Show(errorMsg, "Error", MessageBoxButtons.OK, MessageBoxIcon.Stop);
				App.Terminate();
			}
		}
		#endregion CreateForm()

		#region DestroyForm()
		/// <summary>
		/// Destroys the current Windows Form and releases its resources.
		/// </summary>
		private static void DestroyForm() {
			if(form != null) {
				form.Hide();															// Hide The Form
				form.Close();															// Close And Dispose Of The Form
				form = null;
			}
			if(view != null) {
				view.Dispose();
				view = null;
			}
			GC.Collect();
		}
		#endregion DestroyForm()

		#region MainLoop()
		/// <summary>
		/// Runs the main application loop until the application terminates.
		/// </summary>
		private static void MainLoop() {
			try {
				while(isRunning && !isDisposed && !form.IsDisposed) {					// Loop Until The Application Is Terminated
					if(!runWhileMinimized) {											// If We Shouldn't Run While Minimized...
						if(form.WindowState == FormWindowState.Minimized) {				// If We're Minimized, Save CPU Cycles
							WaitMessage();												// Wait Until We Get A Message Returning Us From Minimization
						}
					}

					if(!isActive) {														// If We Shouldn't Run While Not Active
						WaitMessage();													// Save CPU Cycles, By Just Waiting For A Message
					}

					Form f = view.TopLevelControl as Form;
					//if(f!=null && f.WindowState!=FormWindowState.Minimized)
						view.Redraw();														// Draw The Scene On Our View(s)

					// Calculate Framerate
					framesDrawn++;														// We've Drawn A Frame, Increase The Counter
					currentFrameTime = timer.Count;										// Get The Current Tick Count

					if((currentFrameTime - lastCalculationTime) > timerFrequency) {		// Is It Time To Update Our Calculations?
						// Calculate New Framerate
						currentFramerate = (framesDrawn * timerFrequency) / (currentFrameTime - lastCalculationTime);

						if(currentFramerate < lowestFramerate) {						// Is The New Framerate A New Low?
							lowestFramerate = currentFramerate;							// Set It To The New Low
						}

						if(currentFramerate > highestFramerate) {						// Is The New Framerate A New High?
							highestFramerate = currentFramerate;						// Set It To The New High
						}

						lastCalculationTime = currentFrameTime;							// Update Our Last Frame Time To Now
						framesDrawn = 0;												// Reset Our Frame Count
						framerateReady = true;											// Framerate Has Been Calculated
					}

					Application.DoEvents();												// Let The Form Have A Chance To Process Events
					model.ProcessInput();												// Let The Model Handle User Input
				}
			}
			catch(Exception e) {
				// Handle Any Exceptions While Running The Main Loop, Exit App
				string errorMsg = "A Basecode Error Occurred While Running Main Loop:\n"+e+"\n\nStack Trace:\n\t" + e.StackTrace + "\n";
				MessageBox.Show(errorMsg, "Error", MessageBoxButtons.OK, MessageBoxIcon.Stop);
				App.Terminate();
			}
		}
		#endregion MainLoop()

		// --- Public Methods ---
		#region ResetFramerate()
		/// <summary>
		/// Resets the framerate counter.
		/// </summary>
		public static void ResetFramerate() {
			// Reset Everything Related To Framerate Calculations
			lastCalculationTime = timer.Count;
			framesDrawn = 0;
			currentFramerate = 0.0f;
			highestFramerate = 0.0f;
			lowestFramerate = 999999999;
			framerateReady = false;
		}
		#endregion ResetFramerate()

		#region Run(Model inheritedModel)
		/// <summary>
		/// Runs the supplied <see cref="Model" /> as a Windows Forms application.
		/// </summary>
		/// <param name="inheritedModel">
		/// The inherited <see cref="Model" /> to run as an application.
		/// </param>
		public static void Run(Model inheritedModel) {
			model = inheritedModel;														// Set The Local Model To The Passed In Model
			model.Setup();																// Run Any One-Time-Only Setup Defined By The Model
			CreateForm();																// Create An Initial Form
			GC.Collect();																// Force A Collection
			MainLoop();																	// Runs The Main Application Loop
			Dispose();																	// Cleans Up When The Application Ends
			Application.Exit();															// Exit
		}
		#endregion Run(Model inheritedModel)

		#region Terminate()
		/// <summary>
		/// Terminates and cleans up the application.
		/// </summary>
		public static void Terminate() {
			isRunning = false;															// Stop The Application On The Next Loop
		}
		#endregion Terminate()

		#region ToggleFullscreen()
		/// <summary>
		/// Toggles between fullscreen and windowed mode.
		/// </summary>
		public static void ToggleFullscreen() {
			DestroyForm();																// Get Rid Of The Current Form
			isFullscreen = !isFullscreen;												// Toggle Our Fullscreen State
			CreateForm();																// Create A New Form
			ResetFramerate();															// Reset Our Framerate
			if(isFullscreen) {															// If We're In Fullscreen Mode
				if(showCursorFullscreen) {												// If We're Supposed To Show Cursor
					Cursor.Show();														// Show It
				}
				else {																	// Otherwise
					Cursor.Hide();														// Hide It
				}
			}
			else {																		// Else We're In Windowed Mode
				if(showCursorWindowed) {												// If We're Supposed To Show Cursor
					Cursor.Show();														// Show It
				}
				else {																	// Otherwise
					Cursor.Hide();														// Hide It
				}
			}
			MainLoop();																	// Return To The Main Loop
		}
		#endregion ToggleFullscreen()

		// --- Private Externs ---
		#region WaitMessage()
		/// <summary>
		/// Stalls the current thread until there's a message.
		/// </summary>
		[DllImport("user32")]
		private static extern void WaitMessage();
		#endregion WaitMessage()
	}
}
