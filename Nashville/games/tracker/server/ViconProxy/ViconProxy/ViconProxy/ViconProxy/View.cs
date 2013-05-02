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
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace CsGL.Basecode {
	/// <summary>
	/// Provides an OpenGL view control for your application's form.
	/// </summary>
	public sealed class View : OpenGLControl {
		// --- Fields ---
		#region Private Fields
		private readonly Model model;													// The Model This View Displays
		private bool isDisposed = false;												// Has Disposed Been Called For This View?
		#endregion Private Fields

		// --- Creation And Destruction Methods ---
		#region Constructor
		/// <summary>
		/// Creates a <see cref="View" /> displaying the supplied, inherited <see cref="Model" />.
		/// </summary>
		/// <param name="inheritedModel">
		/// The <see cref="Model" /> to be displayed by this view.
		/// </param>
		public View(Model inheritedModel) {
			model = inheritedModel;
		}
		#endregion Constructor

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
		protected override void Dispose(bool disposing) {
			if(!isDisposed) {															// Check To See If Dispose Has Already Been Called
				if(disposing) {															// If disposing Equals true, Dispose All Managed And Unmanaged Resources
					GC.SuppressFinalize(this);											// Suppress Finalization
				}

				// Release Any Unmanaged Resources Here, If disposing Was false, Only The Following Code Is Executed
				base.Dispose(disposing);
			}
			isDisposed = true;															// Mark As disposed
		}
		#endregion Dispose(bool disposing)

		#region Finalizer
		/// <summary>
		/// This destructor will run only if the Dispose method does not get called.  It gives 
		/// the class the opportunity to finalize.  Simply calls Dispose(false).
		/// </summary>
		~View() {
			Dispose(false);																// We've Automatically Called For A Dispose
		}
		#endregion Finalizer

		// --- Protected Methods ---
		#region CreateContext()
		/// <summary>
		/// Overrides the creation of the OpenGL context.
		/// </summary>
		/// <returns>An OpenGLContext.</returns>
		protected override OpenGLContext CreateContext() {
			// Setup Our PixelFormat And Context
			ControlGLContext context = new ControlGLContext(this);
			DisplayType displayType = new DisplayType(App.ColorDepth, App.ZDepth, App.StencilDepth, App.AccumDepth);
			context.Create(displayType, null);
			return context;
		}
		#endregion CreateContext()

		// --- Public Methods ---
		#region glDraw()
		/// <summary>
		/// Draws the <see cref="Model" /> on the control, overriding OpenGLControl's glDraw().
		/// This is for the basecode to call itself.  You shouldn't need to call this directly.
		/// </summary>
		public override void glDraw() {
			model.Draw();																// Have The Control Draw Our Model
		}
		#endregion glDraw()

		#region Redraw()
		/// <summary>
		/// Paints the control.  This is for the basecode to call itself.  You shouldn't 
		/// need to call this directly.
		/// </summary>
		public void Redraw() {
			OnPaint(null);																// Force Painting
		}
		#endregion Redraw()

		#region Screenshot()
		/// <summary>
		/// Saves a screenshot of the current view to a JPEG image file named
		/// the current assembly name.  If the file already exists, a numerical counter
		/// is added to the filename.
		/// </summary>
		public void Screenshot() {
			try {
				if(!File.Exists(model.GetType().FullName + ".jpg")) {					// If The Filename Doesn't Exist
					Image image = Context.ToImage();									// Create An Image From The OpenGL Context
					image.Save(model.GetType().FullName + ".jpg", ImageFormat.Jpeg);	// Save The Image
					image.Dispose();													// Get Rid Of The Image Object
				}
				else {																	// Otherwise
					int x = 0;															// Let's Add A Number To The Filename
					while(File.Exists(model.GetType().FullName + "_" + x + ".jpg")) {	// While This New Filename Exists
						x++;															// Increase The Number
					}
					Image image = Context.ToImage();									// Create An Image From The OpenGL Context
					// Save The Image
					image.Save(model.GetType().FullName + "_" + x + ".jpg", ImageFormat.Jpeg);
					image.Dispose();													// Get Rid Of The Image Object
				}
			}
			catch(Exception e) {
				// Handle Any Exceptions While Saving The Screenshot, Exit App
				string errorMsg = "An Error Occurred While Saving Screenshot.\n\n\nStack Trace:\n\t" + e.StackTrace + "\n";
				MessageBox.Show(errorMsg, "Error", MessageBoxButtons.OK, MessageBoxIcon.Stop);
				App.Terminate();
			}
		}
		#endregion Screenshot()

		// --- Events ---
		#region IsInputKey(Keys key)
		/// <summary>
		/// Handle these keys, by default they are ignored.
		/// </summary>
		/// <param name="key">Key to test.</param>
		/// <returns>Key handled?</returns>
		protected override bool IsInputKey(Keys key) { 
			switch(key) {
				case Keys.Up:
				case Keys.Down:
				case Keys.Right:
				case Keys.Left:
				case Keys.Tab:
					return true;
				default:
					return base.IsInputKey(key);
			}
		}
		#endregion IsInputKey(Keys key)

		#region OnKeyDown(KeyEventArgs e)
		/// <summary>
		/// Handles the KeyDown event.
		/// </summary>
		/// <param name="e">The KeyEventArgs.</param>
		protected override void OnKeyDown(KeyEventArgs e) {
			Model.KeyState[e.KeyValue] = true;											// Set The KeyState For This Key To Pressed
		}
		#endregion OnKeyDown(KeyEventArgs e)

		#region OnKeyUp(KeyEventArgs e)
		/// <summary>
		/// Handles the KeyUp event.
		/// </summary>
		/// <param name="e">The KeyEventArgs.</param>
		protected override void OnKeyUp(KeyEventArgs e) {
			Model.KeyState[e.KeyValue] = false;											// Set The KeyState For This Key To Not Pressed
		}
		#endregion OnKeyUp(KeyEventArgs e)

		#region OnMouseDown(MouseEventArgs e)
		/// <summary>
		/// Handles the MouseDown event.
		/// </summary>
		/// <param name="e">The MouseEventArgs.</param>
		protected override void OnMouseDown(MouseEventArgs e) {
			if(e.Button == MouseButtons.Left) {
				Model.Mouse.LeftButton = true;
			}
			else if(e.Button == MouseButtons.Middle) {
				Model.Mouse.MiddleButton = true;
			}
			else if(e.Button == MouseButtons.Right) {
				Model.Mouse.RightButton = true;
			}
		}
		#endregion OnMouseDown(MouseEventArgs e)

		#region OnMouseEnter(EventArgs e)
		/// <summary>
		/// Handles the MouseEnter event.
		/// </summary>
		/// <param name="e">The EventArgs.</param>
		protected override void OnMouseEnter(EventArgs e) {
			Model.Mouse.DifferenceX = 0;
			Model.Mouse.DifferenceY = 0;
		}
		#endregion OnMouseEnter(EventArgs e)

		#region OnMouseMove(MouseEventArgs e)
		/// <summary>
		/// Handles the MouseMove event.
		/// </summary>
		/// <param name="e">The MouseEventArgs.</param>
		protected override void OnMouseMove(MouseEventArgs e) {
			Model.Mouse.LastX = Model.Mouse.X;
			Model.Mouse.LastY = Model.Mouse.Y;
			Model.Mouse.X = e.X;
			Model.Mouse.Y = e.Y;
			Model.Mouse.DifferenceX = Model.Mouse.X - Model.Mouse.LastX;
			Model.Mouse.DifferenceY = Model.Mouse.Y - Model.Mouse.LastY;
		}
		#endregion OnMouseMove(MouseEventArgs e)

		#region OnMouseUp(MouseEventArgs e)
		/// <summary>
		/// Handles the MouseUp event.
		/// </summary>
		/// <param name="e">The MouseEventArgs.</param>
		protected override void OnMouseUp(MouseEventArgs e) {
			if(e.Button == MouseButtons.Left) {
				Model.Mouse.LeftButton = false;
			}
			else if(e.Button == MouseButtons.Middle) {
				Model.Mouse.MiddleButton = false;
			}
			else if(e.Button == MouseButtons.Right) {
				Model.Mouse.RightButton = false;
			}
			else if(e.Button == MouseButtons.XButton1) {
				Model.Mouse.XButton1 = false;
			}
			else if(e.Button == MouseButtons.XButton2) {
				Model.Mouse.XButton2 = false;
			}
		}
		#endregion OnMouseDown(MouseEventArgs e)

		#region OnSizeChanged(EventArgs e)
		/// <summary>
		/// Handles the OnSizeChanged event.  Reshapes the OpenGL control.
		/// </summary>
		/// <param name="e">The EventArgs.</param>
		protected override void OnSizeChanged(EventArgs e) {
			if(App.IsRunning) {															// If The App Is Running
				Size s = Size;	// Get The New Size
				if(s.Width!=0 && s.Height!=0)
					App.Model.Reshape((int) s.Width, (int) s.Height);						// Let The OpenGL Control Resize
			}
			base.OnSizeChanged(e);														// Let The Form Resize
		}
		#endregion OnSizeChanged(EventArgs e)
	}
}
