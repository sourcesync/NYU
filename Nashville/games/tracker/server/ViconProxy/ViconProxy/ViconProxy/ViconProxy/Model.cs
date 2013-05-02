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
using System;
using System.Data;
using System.Drawing;
using System.Windows.Forms;

namespace CsGL.Basecode {
	/// <summary>
	/// Provides a template for a CsGL application using the CsGL Basecode.  It provides methods 
	/// for setting up the application, initializing, drawing, and user input.  Many of the provided 
	/// methods can and should be overridden in your inherited model.
	/// </summary>
	/// <remarks>
	/// <para>
	/// The <b>Model</b> class is the template for all applications using the CsGL Basecode.  It 
	/// provides basic functionality on its own, to allow for rapid development, however, by 
	/// overriding the appropriate methods you can modify and extend your application's functionality 
	/// over what is provided by default.
	/// </para>
	/// </remarks>
	public class Model : GL, IDisposable {
		// --- Fields ---
		#region Private Fields
		private static HelpForm helpForm;												// The Basecode's Help Form
		private static DataTable dtInputHelp;											// DataTable Defining Input Help For The Basecode Help Screen
		private static bool isHelpDisplayed = false;									// Is The Help Screen Currently Displayed?
		private static bool isDisposed = false;											// Has Dispose Been Called?
		#endregion Private Fields

		#region Public Fields
		/// <summary>
		/// Current keyboard state.  The key's integer value, is the index for this array.  If that 
		/// index is true, the key is being pressed, if it's false, the key is not being pressed.  You 
		/// should likely mark the key as handled, by setting it's index to false when you've processed it.
		/// </summary>
		public static bool[] KeyState = new bool[256];									// Keyboard State

		/// <summary>
		/// Current mouse state.
		/// </summary>
		public struct Mouse {
			/// <summary>
			/// X-axis position in the view.
			/// </summary>
			public static int X;

			/// <summary>
			/// Y-axis position in the view.
			/// </summary>
			public static int Y;

			/// <summary>
			/// Previous X-axis position in the view.
			/// </summary>
			public static int LastX;

			/// <summary>
			/// Previous Y-axis position in the view.
			/// </summary>
			public static int LastY;

			/// <summary>
			/// Difference between the current and previous X-axis position in the view.
			/// </summary>
			public static int DifferenceX;

			/// <summary>
			/// Difference between the current and previous Y-axis position in the view.
			/// </summary>
			public static int DifferenceY;

			/// <summary>
			/// Is left mouse button pressed?
			/// </summary>
			public static bool LeftButton;

			/// <summary>
			/// Is middle mouse button pressed?
			/// </summary>
			public static bool MiddleButton;

			/// <summary>
			/// Is right mouse button pressed?
			/// </summary>
			public static bool RightButton;

			/// <summary>
			/// Is X button 1 (Intellimouse) pressed?  Windows 2000 and above only.
			/// </summary>
			public static bool XButton1;

			/// <summary>
			/// Is X button 2 (Intellimouse) pressed?  Windows 2000 and above only.
			/// </summary>
			public static bool XButton2;
		}
		#endregion Public Fields

		#region Public Properties
		/// <summary>
		/// Model-specific description.
		/// </summary>
		/// <remarks>
		/// Override this to provide a description of your application for use in the
		/// basecode's help screen.  If you're not using the basecode's help screen, 
		/// you can either disregard this property, or use it how you like.
		/// </remarks>
		/// <example>
		/// In your inherited <see cref="Model" />:
		/// <code>
		/// public override string Description {
		/// 	get {
		/// 		return "My fancy OpenGL lesson!";
		/// 	}
		/// }
		/// </code>
		/// </example>
		public virtual string Description {
			get {
				return null;
			}
		}

		/// <summary>
		/// Is the basecode's help screen currently displayed?
		/// </summary>
		public bool IsHelpDisplayed {
			get {
				return isHelpDisplayed;
			}
			set {
				isHelpDisplayed = value;
			}
		}

		/// <summary>
		/// The basecode's help form.
		/// </summary>
		public HelpForm HelpForm {
			get {
				return helpForm;
			}
		}

		/// <summary>
		/// The model's input help DataTable.
		/// </summary>
		public DataTable InputHelpDataTable {
			get {
				return dtInputHelp;
			}
		}

		/// <summary>
		/// Model-specific title.
		/// </summary>
		/// <remarks>
		/// Override this to provide a title for your application for use in the
		/// basecode's help screen.  If you're not using the basecode's help screen, 
		/// you can either disregard this property, or use it how you like.
		/// </remarks>
		/// <example>
		/// In your inherited <see cref="Model" />:
		/// <code>
		/// public override string Title {
		/// 	get {
		/// 		return "My OpenGL Lesson";
		/// 	}
		/// }
		/// </code>
		/// </example>
		public virtual string Title {
			get {
				return "A CsGL Example";
			}
		}

		/// <summary>
		/// Model-specific URL.
		/// </summary>
		/// <remarks>
		/// Override this to provide a URL for your application for use in the
		/// basecode's help screen.  If you're not using the basecode's help screen, 
		/// you can either disregard this property, or use it how you like.
		/// </remarks>
		/// <example>
		/// In your inherited <see cref="Model" />:
		/// <code>
		/// public override string Url {
		/// 	get {
		/// 		return "http://csgl.sourceforge.net/";
		/// 	}
		/// }
		/// </code>
		/// </example>
		public virtual string Url {
			get {
				return null;
			}
		}

		/// <summary>
		/// Model-specific window icon.
		/// </summary>
		/// <remarks>
		/// Override this to provide an <see cref="Icon" /> for your application to use 
		/// in place of the basecode's default icon.
		/// </remarks>
		/// <example>
		/// In your inherited <see cref="Model" />:
		/// <code>
		/// public override Icon WindowIcon {
		/// 	get {
		/// 		return new Icon(filename);
		/// 	}
		/// }
		/// </code>
		/// </example>
		public virtual Icon WindowIcon {
			get {
				return null;
			}
		}

		/// <summary>
		/// Model-specific window title.
		/// </summary>
		/// <remarks>
		/// Override this to provide a title for your application in its title bar.  
		/// This title is not used in the basecode's help screen, only in the 
		/// application's title bar.  If this is not supplied in your model, the
		/// basecode will use the <see cref="Title" /> property for the title bar.
		/// </remarks>
		/// <example>
		/// In your inherited <see cref="Model" />:
		/// <code>
		/// public override string WindowTitle {
		/// 	get {
		/// 		return "My OpenGL Lesson's Title Bar!";
		/// 	}
		/// }
		/// </code>
		/// </example>
		public virtual string WindowTitle {
			get {
				return null;
			}
		}
		#endregion Public Properties

		// --- Creation & Destruction Methods ---
		#region Dispose()
		/// <summary>
		/// Disposes of this class.  Implements IDisposable.
		/// </summary>
		public void Dispose() {
			Dispose(true);																// We've Manually Called For A Dispose
			GC.SuppressFinalize(this);													// Prevent Being Added To The Finalization Queue
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
		public void Dispose(bool disposing) {
			if(!isDisposed) {															// Check To See If Dispose Has Already Been Called
				if(disposing) {															// If disposing Equals true, Dispose All Managed And Unmanaged Resources
					if(isHelpDisplayed) {
						helpForm.Hide();
					}
					if(dtInputHelp != null) {
						dtInputHelp.Dispose();
					}
					if(helpForm != null) {
						helpForm.Close();
					}
				}

				// Release Any Unmanaged Resources Here, If disposing Was false, Only The Following Code Is Executed
				helpForm = null;
				dtInputHelp = null;
			}
			isDisposed = true;															// Mark As disposed
		}
		#endregion Dispose(bool disposing)

		#region Finalizer
		/// <summary>
		/// This destructor will run only if the Dispose method does not get called.  It gives 
		/// the class the opportunity to finalize.  Simply calls Dispose(false).
		/// </summary>
		~Model() {
			Dispose(false);																// We've Automatically Called For A Dispose
		}
		#endregion Finalizer

		// --- Private Methods ---
		#region DisplayHelp()
		/// <summary>
		/// Displays a basecode help screen, including model-specific input help information.
		/// </summary>
		private void DisplayHelp() {
			helpForm = new HelpForm();													// Create A Help Form
			UpdateInputHelp();															// Update The Help Form's DataGrid
			helpForm.Show();															// Display The Help Form
			if(App.IsFullscreen) {														// If We're In Fullscreen Mode
				Cursor.Show();															// Show The Cursor
			}
			App.ResetFramerate();														// Reset The Framerate
		}
		#endregion DisplayHelp()

		// --- Public Methods ---
		#region Draw()
		/// <summary>
		/// All drawing occurs here.  Override to provide your scene's OpenGL drawing.
		/// </summary>
		public virtual void Draw() {
		}
		#endregion Draw()

		#region Initialize()
		/// <summary>
		/// All initial setup for OpenGL goes here, override for application-specific setup.
		/// This method is called every time a form for your application is created, including
		/// at startup, when desktop resolution changes, and when you toggle to or from
		/// fullscreen or windowed mode.
		/// </summary>
		public virtual void Initialize() {
			glShadeModel(GL_SMOOTH);													// Enable Smooth Shading
			glClearColor(0.0f, 0.0f, 0.0f, 0.5f);										// Black Background
			glClearDepth(1.0f);															// Depth Buffer Setup
			glEnable(GL_DEPTH_TEST);													// Enables Depth Testing
			glDepthFunc(GL_LEQUAL);														// The Type Of Depth Testing To Do
			glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);							// Really Nice Perspective Calculations
		}
		#endregion Initialize()

		#region InputHelp()
		/// <summary>
		/// Defines the input help information, override to supply model-specific input help.  If
		/// you are not allowing <see cref="ProcessInput" /> to handle the default basecode input, 
		/// then ignore this.
		/// </summary>
		public virtual void InputHelp() {
			DataRow dataRow;															// Row To Add

			dataRow = InputHelpDataTable.NewRow();										// ESC - Quit App
			dataRow["Input"] = "ESC";
			dataRow["Effect"] = "Quit Application";
			dataRow["Current State"] = "";
			InputHelpDataTable.Rows.Add(dataRow);

			dataRow = InputHelpDataTable.NewRow();										// F1 - Show Help Screen
			dataRow["Input"] = "F1";
			dataRow["Effect"] = "Show Help Screen";
			dataRow["Current State"] = "This Screen";
			InputHelpDataTable.Rows.Add(dataRow);

			dataRow = InputHelpDataTable.NewRow();										// F2 - Toggle Fullscreen
			dataRow["Input"] = "F2";
			dataRow["Effect"] = "Toggle Fullscreen / Windowed";
			if(App.IsFullscreen) {
				dataRow["Current State"] = "Fullscreen";
			}
			else {
				dataRow["Current State"] = "Windowed";
			}
			InputHelpDataTable.Rows.Add(dataRow);

			dataRow = InputHelpDataTable.NewRow();										// F3 - Reset FPS
			dataRow["Input"] = "F3";
			dataRow["Effect"] = "Reset FPS";
			dataRow["Current State"] = "";
			InputHelpDataTable.Rows.Add(dataRow);

			dataRow = InputHelpDataTable.NewRow();										// F4 - Toggle StatusBar
			dataRow["Input"] = "F4";
			dataRow["Effect"] = "Toggle Status Bar On / Off";
			if(App.ShowStatusBar) {
				dataRow["Current State"] = "On";
			}
			else {
				dataRow["Current State"] = "Off";
			}
			InputHelpDataTable.Rows.Add(dataRow);
			
			dataRow = InputHelpDataTable.NewRow();										// F5 - Take Screenshot
			dataRow["Input"] = "F5";
			dataRow["Effect"] = "Take Screenshot";
			dataRow["Current State"] = "";
			InputHelpDataTable.Rows.Add(dataRow);
		}
		#endregion InputHelp()

		#region WindowsForm()
		/// <summary>
		/// Provides a hook for you to setup your own application's Windows Form for windowed 
		/// mode.  If you do not override this, then the default basecode's Windows Form will 
		/// be used.  If you do override this, make sure to add <see cref="View" /> to 
		/// your form as a control and don't forget to Show() your form.
		/// </summary>
		public virtual void WindowsForm() {
			App.Form = new AppForm();													// Create The Default Basecode Form
			App.Form.Show();
		}
		#endregion WindowsForm()

		#region ProcessInput()
		/// <summary>
		/// Handles some default user input, override for custom input handling and functionality.
		/// </summary>
		public virtual void ProcessInput() {
			if(KeyState[(int) Keys.Escape]) {											// Is Escape Pressed?
				App.Terminate();														// Kill The App
			}

			if(KeyState[(int) Keys.F1]) {												// Is F1 Pressed?
				KeyState[(int) Keys.F1] = false;										// Set F1 Handled
				if(!isHelpDisplayed) {													// If The Help Screen Isn't Currently Displayed
					DisplayHelp();														// Show Help Screen
				}
				else {																	// If It Is Displayed
					helpForm.Focus();													// Give It Focus
				}
			}

			if(KeyState[(int) Keys.F2]) {												// If F2 Pressed?
				KeyState[(int) Keys.F2] = false;										// Set F2 Handled
				App.ToggleFullscreen();													// Toggle Between Fullscreen And Windowed Mode
				//UpdateInputHelp();
			}

			if(KeyState[(int) Keys.F3]) {												// Is F3 Pressed?
				KeyState[(int) Keys.F3] = false;										// Set F3 Handled 
				App.ResetFramerate();													// Reset The Framerate
			}

			if(KeyState[(int) Keys.F4]) {												// Is F4 Pressed?
				KeyState[(int) Keys.F4] = false;										// Set F4 Handled
				if(App.Form.GetType() == Type.GetType("CsGL.Basecode.AppForm")) {		// If The Form Is AppForm
					((AppForm) App.Form).ToggleStatusBar();								// Toggle Display Of Status Bar
					UpdateInputHelp();													// Update The Input Screens State
				}
			}

			if(KeyState[(int) Keys.F5]) {												// Is F5 Pressed?
				KeyState[(int) Keys.F5] = false;										// Set F5 Handled
				App.View.Screenshot();													// Save A Screenshot
			}
		}
 
		#endregion ProcessInput()

		#region Reshape(int width, int height)
		/// <summary>
		/// Reshaping code for OpenGL goes here, override for application-specific 
		/// resizing functionality.
		/// </summary>
		/// <param name="width">The new width of the window.</param>
		/// <param name="height">The new height of the window.</param>
		public virtual void Reshape(int width, int height) {
			if(height == 0) {															// If Height Is 0
				height = 1;																// Set To 1 To Prevent A Divide By Zero
			}

			glViewport(0, 0, width, height);											// Reset The Current Viewport
			glMatrixMode(GL_PROJECTION);												// Select The Projection Matris
			glLoadIdentity();															// Reset The Projection Matrix

			// Set Up A Perspective Projection Matrix 
			gluPerspective(App.FovY, (float) width / (float) height, App.NearClippingPlane, App.FarClippingPlane);

			glMatrixMode(GL_MODELVIEW);													// Select The Modelview Matrix
			glLoadIdentity();															// Reset The Modelview Matrix
		}
		#endregion Reshape(int width, int height)

		#region Setup()
		/// <summary>
		/// Provides a place to perform any initial or one-time-only setup for your 
		/// application before it begins the main loop.
		/// </summary>
		/// <remarks>
		/// Override to provide your own setup.  You could run a dialog, a splash screen, 
		/// override <see cref="App" /> properties, whatever you like.  The default 
		/// implementation shows the basecode's <see cref="SetupForm" />, which provides 
		/// a basic setup screen for the end-user.
		/// </remarks>
		public virtual void Setup() {
			SetupForm setupForm = new SetupForm();										// Create The Basecode's SetupForm
			setupForm.ShowDialog();														// Show The SetupForm Modal
			setupForm.Dispose();
			setupForm = null;
			App.RunWhileMinimized = false;
			GC.Collect();
		}
		#endregion Setup()

		#region UpdateInputHelp()
		/// <summary>
		/// Updates the help screen's input help, call when you need to update state on the help screen.
		/// </summary>
		public void UpdateInputHelp() {
			if(isHelpDisplayed && helpForm != null) {									// If The HelpForm Has Been Created
				DataSet dsInputHelp = new DataSet();									// DataSet Containing Input Help
				dtInputHelp = new DataTable("InputHelp");								// DataTable Defining Input Help
				DataColumn dcInput = new DataColumn("Input", typeof(string));			// Input DataColumn
				DataColumn dcEffect = new DataColumn("Effect", typeof(string));			// Effect DataColumn
				DataColumn dcState = new DataColumn("Current State", typeof(string));	// Current State DataColumn
				dtInputHelp.Columns.Add(dcInput);										// Add The DataColumn To The DataTable
				dtInputHelp.Columns.Add(dcEffect);										// ""
				dtInputHelp.Columns.Add(dcState);										// ""
				dsInputHelp.Tables.Add(dtInputHelp);									// Add The DataTable To The DataSet

				InputHelp();															// Build The Input Help

				// Bind The DataSet To The DataGrid
				helpForm.InputHelpDataGrid.DataSource = dsInputHelp.Tables["InputHelp"];

				helpForm.InputHelpDataGrid.ScrollToRow();								// Make Sure We Didn't Lose Our Scrolled Position

				App.Form.Focus();														// Give The Main Window Focus
			}
		}
		#endregion UpdateInputHelp()
	}
}