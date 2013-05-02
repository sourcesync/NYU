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

using System;
using System.Drawing;
using System.Reflection;
using System.Resources;
using System.Windows.Forms;

namespace CsGL.Basecode {
	/// <summary>
	/// Provides a default form for end-user application setup.
	/// </summary>
	/// <remarks>
	/// <b>SetupForm</b> provides a form allowing the end-user to make some 
	/// application setting choices.  Including setting resolution, color depth,
	/// and windowed/fullscreen startup.  This form is called by default 
	/// from <see cref="Model.Setup" />.  You can override <see cref="Model.Setup" /> 
	/// to provide your own form, your own custom setup, or nothing at all.
	/// </remarks>
	public sealed class SetupForm : Form {
		// --- Fields ---
		#region Private Fields
		private static PictureBox pictureBox = new PictureBox();						// Area For Basecode's Logo
		private static GroupBox grpResolution = new GroupBox();							// Group For Resolutions
		private static RadioButton rad640 = new RadioButton();							// 640x480
		private static RadioButton rad800 = new RadioButton();							// 800x600
		private static RadioButton rad1024 = new RadioButton();							// 1024x768
		private static RadioButton rad1600 = new RadioButton();							// 1600x1200
		private static GroupBox grpColorDepth = new GroupBox();							// Group For Color Depths
		private static RadioButton rad16bpp = new RadioButton();						// 16 Bits Per Pixel
		private static RadioButton rad24bpp = new RadioButton();						// 24 Bits Per Pixel
		private static RadioButton rad32bpp = new RadioButton();						// 32 Bits Per Pixel
		private static CheckBox chkFullscreen = new CheckBox();							// Fullscreen? Checkbox
		private static CheckBox chkStatusBar = new CheckBox();							// StatusBar? Checkbox
		private static Button btnOk = new Button();										// OK Button
		private static bool isDisposed = false;											// Has Dispose Been Called?
		#endregion Private Fields

		// --- Creation & Destruction Methods ---
		#region SetupForm()
		/// <summary>
		/// Constructor.
		/// </summary>
		public SetupForm() {
			this.SuspendLayout();

			// pictureBox
			pictureBox.BorderStyle = BorderStyle.Fixed3D;
			ResourceManager rm = new ResourceManager("BasecodeResources", Assembly.GetCallingAssembly());
			//gw pictureBox.Image = ((Bitmap)(rm.GetObject("SetupFormLogo")));
			pictureBox.Location = new Point(2, 2);
			pictureBox.Size = new Size(352, 150);

			// grpResolution
			grpResolution.Controls.AddRange(new Control[] { rad640, rad800, rad1024, rad1600 });
			grpResolution.FlatStyle = FlatStyle.Flat;
			grpResolution.Location = new Point(11, 164);
			grpResolution.Size = new Size(334, 50);
			grpResolution.Text = "Select Resolution";

			// rad640
			rad640.Checked = true;
			rad640.FlatStyle = FlatStyle.Flat;
			rad640.Location = new Point(15, 20);
			rad640.Size = new Size(70, 24);
			rad640.Text = "640x480";
			rad640.Cursor = Cursors.Hand;

			// rad800
			rad800.FlatStyle = FlatStyle.Flat;
			rad800.Location = new Point(91, 20);
			rad800.Size = new Size(66, 24);
			rad800.Text = "800x600";
			rad800.Cursor = Cursors.Hand;

			// rad1024
			rad1024.FlatStyle = FlatStyle.Flat;
			rad1024.Location = new Point(163, 20);
			rad1024.Size = new Size(72, 24);
			rad1024.Text = "1024x768";
			rad1024.Cursor = Cursors.Hand;

			// rad1600
			rad1600.FlatStyle = FlatStyle.Flat;
			rad1600.Location = new Point(241, 20);
			rad1600.Size = new Size(78, 24);
			rad1600.Text = "1600x1200";
			rad1600.Cursor = Cursors.Hand;

			// grpColorDepth
			grpColorDepth.Controls.AddRange(new Control[] { rad16bpp, rad24bpp, rad32bpp });
			grpColorDepth.FlatStyle = FlatStyle.Flat;
			grpColorDepth.Location = new Point(11, 228);
			grpColorDepth.Size = new Size(334, 50);
			grpColorDepth.Text = "Select Color Depth";

			// rad16bpp
			rad16bpp.Checked = true;
			rad16bpp.FlatStyle = FlatStyle.Flat;
			rad16bpp.Location = new Point(72, 20);
			rad16bpp.Size = new Size(56, 24);
			rad16bpp.Text = "16bpp";
			rad16bpp.Cursor = Cursors.Hand;

			// rad24bpp
			rad24bpp.FlatStyle = FlatStyle.Flat;
			rad24bpp.Location = new Point(141, 20);
			rad24bpp.Size = new Size(56, 24);
			rad24bpp.Text = "24bpp";
			rad24bpp.Cursor = Cursors.Hand;

			// rad32bpp
			rad32bpp.FlatStyle = FlatStyle.Flat;
			rad32bpp.Location = new Point(210, 20);
			rad32bpp.Size = new Size(56, 24);
			rad32bpp.Text = "32bpp";
			rad32bpp.Cursor = Cursors.Hand;

			// chkFullscreen
			chkFullscreen.CheckAlign = ContentAlignment.MiddleRight;
			chkFullscreen.FlatStyle = FlatStyle.Flat;
			chkFullscreen.Location = new Point(35, 292);
			chkFullscreen.Size = new Size(81, 24);
			chkFullscreen.Text = "Fullscreen:";
			chkFullscreen.Cursor = Cursors.Hand;

			// chkStatusBar
			chkStatusBar.CheckAlign = ContentAlignment.MiddleRight;
			chkStatusBar.Checked = true;
			chkStatusBar.CheckState = CheckState.Checked;
			chkStatusBar.FlatStyle = FlatStyle.Flat;
			chkStatusBar.Location = new Point(137, 292);
			chkStatusBar.Size = new Size(163, 24);
			chkStatusBar.Text = "Status Bar While Windowed:";
			chkStatusBar.Cursor = Cursors.Hand;

			// btnOk
			btnOk.Cursor = Cursors.Hand;
			btnOk.FlatStyle = FlatStyle.Flat;
			btnOk.Location = new Point(141, 330);
			btnOk.Text = "OK";
			btnOk.Click += new EventHandler(BtnOk_Click);
			btnOk.MouseEnter += new System.EventHandler(BtnOk_MouseEnter);
			btnOk.MouseLeave += new System.EventHandler(BtnOk_MouseLeave);

			// this
			this.AcceptButton = btnOk;
			this.BackColor = Color.FromArgb(((byte)(207)), ((byte)(207)), ((byte)(207)));
			this.ClientSize = new Size(356, 368);
			this.ControlBox = false;
			this.Controls.AddRange(new Control[] { pictureBox, grpResolution, grpColorDepth, chkFullscreen, chkStatusBar, btnOk });
			this.FormBorderStyle = FormBorderStyle.None;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.ShowInTaskbar = false;
			this.StartPosition = FormStartPosition.CenterScreen;
			this.Text = "CsGL Basecode Setup";
			this.TopMost = true;

			this.ResumeLayout();
		}
		#endregion SetupForm()

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
					pictureBox.Dispose();
					rad640.Dispose();
					rad800.Dispose();
					rad1024.Dispose();
					rad1600.Dispose();
					rad16bpp.Dispose();
					rad24bpp.Dispose();
					rad32bpp.Dispose();
					grpResolution.Dispose();
					grpColorDepth.Dispose();
					chkFullscreen.Dispose();
					chkStatusBar.Dispose();
					btnOk.Dispose();
					GC.SuppressFinalize(this);											// Suppress Finalization
				}

				// Release Any Unmanaged Resources Here, If disposing Was false, Only The Following Code Is Executed
				pictureBox = null;
				rad640 = null;
				rad800 = null;
				rad1024 = null;
				rad1600 = null;
				rad16bpp = null;
				rad24bpp = null;
				rad32bpp = null;
				grpResolution = null;
				grpColorDepth = null;
				chkFullscreen = null;
				chkStatusBar = null;
				btnOk = null;
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
		~SetupForm() {
			Dispose(false);																// We've Automatically Called For A Dispose
		}
		#endregion Finalizer

		// --- Events ---
		#region BtnOk_Click(object sender, EventArgs e)
		/// <summary>
		/// Handles the OK button Click event.  Saves user selections to 
		/// <see cref="App" />'s properties.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The EventArgs.</param>
		private void BtnOk_Click(object sender, EventArgs e) {
			// Resolution
			if(rad640.Checked) {														// Did User Choose 640x480?
				App.Width = 640;														// Set Width To 640
				App.Height = 480;														// Set Height To 480
			}
			else if(rad800.Checked) {													// Did User Choose 800x600?
				App.Width = 800;														// Set Width To 800
				App.Height = 600;														// Set Height To 600
			}
			else if(rad1024.Checked) {													// Did User Choose 1024x768?
				App.Width = 1024;														// Set Width To 1024
				App.Height = 768;														// Set Height To 768
			}
			else if(rad1600.Checked) {													// Did User Choose 1600x1200?
				App.Width = 1600;														// Set Width To 1600
				App.Height = 1200;														// Set Height To 1200
			}

			// Color Depth
			if(rad16bpp.Checked) {														// Did User Choose 16bpp?
				App.ColorDepth = 16;													// Set ColorDpeth To 16
			}
			else if(rad24bpp.Checked) {													// Did User Choose 24bpp?
				App.ColorDepth = 24;													// Set ColorDepth To 24
			}
			else if(rad32bpp.Checked) {													// Did User Choose 32bpp?
				App.ColorDepth = 32;													// Set ColorDepth To 32
			}

			// Fullscreen
			if(chkFullscreen.Checked) {													// Did User Choose To Start In Fullscreen?
				App.IsFullscreen = true;												// Set To Fullscreen
			}
			else {																		// Otherwise
				App.IsFullscreen = false;												// Set To Windowed
			}

			// Status Bar
			if(chkStatusBar.Checked) {													// Did User Choose To Use StatusBar In Windowed Mode?
				App.ShowStatusBar = true;												// Show The StatusBar When In Windowed Mode
			}
			else {																		// Otherwise
				App.ShowStatusBar = false;												// Hide The StatusBar When In Windowed Mode
			}

			this.Hide();																// Hide The SetupForm
		}
		#endregion BtnOk_Click(object sender, EventArgs e)

		#region BtnOk_MouseEnter(object sender, EventArgs e)
		/// <summary>
		/// Handles the OK button MouseEnter event.  Changes the button color to a reddish.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The EventArgs.</param>
		private void BtnOk_MouseEnter(object sender, EventArgs e) {
			btnOk.BackColor = Color.FromArgb(((byte)(248)), ((byte)(131)), ((byte)(131)));
		}
		#endregion BtnOk_MouseEnter(object sender, EventArgs e)

		#region BtnOk_MouseLeave(object sender, EventArgs e)
		/// <summary>
		/// Handles the OK button MouseLeave event.  Changes the button color back to a gray.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The EventArgs.</param>
		private void BtnOk_MouseLeave(object sender, EventArgs e) {
			btnOk.BackColor = Color.FromArgb(((byte)(207)), ((byte)(207)), ((byte)(207)));
		}
		#endregion BtnOk_MouseLeave(object sender, EventArgs e)
	}
}