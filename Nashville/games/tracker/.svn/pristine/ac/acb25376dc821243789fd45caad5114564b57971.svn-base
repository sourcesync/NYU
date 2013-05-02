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

using CsGL.Util;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Reflection;
using System.Resources;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace CsGL.Basecode {
	/// <summary>
	/// Provides a default Windows Form for basecode applications.
	/// </summary>
	public sealed class AppForm : Form {
		// --- Fields ---
		#region Private Fields
		private StatusBar statusBar = new StatusBar();									// The StatusBar Area
		private StatusBarPanel sbpInformation = new StatusBarPanel();					// Panel For Help Message
		private StatusBarPanel sbpResolution = new StatusBarPanel();					// Panel For Resolution
		private StatusBarPanel sbpCurrentFps = new StatusBarPanel();					// Panel For Current FPS
		private StatusBarPanel sbpHighestFps = new StatusBarPanel();					// Panel For Highest FPS
		private StatusBarPanel sbpLowestFps = new StatusBarPanel();						// Panel For Lowest FPS
		private Timer timer = new Timer();												// Generic Timer For Updating StatusBar
		private bool isDisposed = false;												// Has Dispose Been Called?
		#endregion Private Fields

		// --- Creation & Destruction Methods ---
		#region AppForm()
		/// <summary>
		/// Constructor.
		/// </summary>
		public AppForm() {
			this.SuspendLayout();

			// sbpInformation
			sbpInformation.Alignment = HorizontalAlignment.Center;
			sbpInformation.AutoSize = StatusBarPanelAutoSize.Contents;
			sbpInformation.Text = "F1 For Help";

			// sbpResolution
			sbpResolution.Alignment = HorizontalAlignment.Center;
			sbpResolution.AutoSize = StatusBarPanelAutoSize.Contents;
			sbpResolution.Text = App.Width.ToString() + "x" + App.Height.ToString() + "x" + App.ColorDepth;

			// sbpCurrentFps
			sbpCurrentFps.Alignment = HorizontalAlignment.Center;
			sbpCurrentFps.AutoSize = StatusBarPanelAutoSize.Contents;
			sbpCurrentFps.Text = "Current: -- FPS";

			// sbpHighestFps
			sbpHighestFps.Alignment = HorizontalAlignment.Center;
			sbpHighestFps.AutoSize = StatusBarPanelAutoSize.Contents;
			sbpHighestFps.Text = "High: -- FPS";

			// sbpLowestFps
			sbpLowestFps.Alignment = HorizontalAlignment.Center;
			sbpLowestFps.AutoSize = StatusBarPanelAutoSize.Contents;
			sbpLowestFps.Text = "Low: -- FPS";

			// statusBar
			statusBar.Panels.AddRange(new StatusBarPanel[] { sbpInformation, sbpResolution, sbpCurrentFps, sbpHighestFps, sbpLowestFps });
			statusBar.ShowPanels = true;

			// view
			App.View.Dock = DockStyle.Fill;

			// timer
			timer.Interval = 500;
			timer.Enabled = true;
			timer.Tick += new EventHandler(Timer_Tick);

			this.Controls.AddRange(new Control[] { App.View, statusBar });

			// this
			if(App.Model.WindowIcon == null) {
				ResourceManager rm = new ResourceManager("BasecodeResources", Assembly.GetCallingAssembly());
				//gw this.Icon = (Icon) rm.GetObject("AppFormIcon");
                this.Icon = (Icon)App.Model.WindowIcon;
                //gw
			}
			else {
				this.Icon = (Icon) App.Model.WindowIcon;
			}

			if(App.Model.WindowTitle == null) {
				this.Text = App.Model.Title;
			}
			else {
				this.Text = App.Model.WindowTitle;
			}

			this.Size = new Size(App.Width, App.Height);
			this.StartPosition = FormStartPosition.CenterScreen;
			this.SizeChanged += new EventHandler(Form_SizeChanged);
			this.Activated += new EventHandler(Form_Activated);

			if(App.ShowStatusBar) {
				statusBar.Visible = true;
			}
			else {
				statusBar.Visible = false;
			}

			this.ResumeLayout();
		}
		#endregion AppForm()

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
					sbpInformation.Dispose();
					sbpResolution.Dispose();
					sbpCurrentFps.Dispose();
					sbpHighestFps.Dispose();
					sbpLowestFps.Dispose();
					timer.Dispose();
					GC.SuppressFinalize(this);											// Suppress Finalization
				}

				// Release Any Unmanaged Resources Here, If disposing Was false, Only The Following Code Is Executed
				sbpInformation = null;
				sbpResolution = null;
				sbpCurrentFps = null;
				sbpHighestFps = null;
				sbpLowestFps = null;
				timer = null;
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
		~AppForm() {
			Dispose(false);																// We've Automatically Called For A Dispose
		}
		#endregion Finalizer

		// --- Public Methods ---
		#region ToggleStatusBar()
		/// <summary>
		/// Toggles the display of the StatusBar.
		/// </summary>
		public void ToggleStatusBar() {
			if(App.ShowStatusBar) {														// Status Bar Is Showing, Let's Hide The Status Bar
				statusBar.Hide();														// Hide It
			}
			else {																		// Status Bar Is Not Showing, Let's Show Status Bar
				statusBar.Show();														// Show It
			}

			App.ShowStatusBar = !App.ShowStatusBar;										// Toggle Our Status Bar Display State
			App.ResetFramerate();														// Reset FPS
		}
		#endregion ToggleStatusBar()

		#region UpdateStatusBar(string current, string highest, string lowest)
		/// <summary>
		/// Updates framerate display in StatusBar.
		/// </summary>
		/// <param name="current">Current FPS.</param>
		/// <param name="highest">Highest FPS.</param>
		/// <param name="lowest">Lowest FPS.</param>
		private void UpdateStatusBar(string current, string highest, string lowest) {
			sbpCurrentFps.Text = "Current: " + current + " FPS";						// Set The Current FPS Text
			sbpHighestFps.Text = "High: " + highest + " FPS";							// Set The Highest FPS Text
			sbpLowestFps.Text = "Low: " + lowest + " FPS";								// Set The Lowest FPS Text
		}
		#endregion UpdateStatusBar(string current, string highest, string lowest)

		// --- Events ---
		#region Form_Activated(object sender, EventArgs e)
		/// <summary>
		/// Handles Form Activated event.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The EventArgs.</param>
		private void Form_Activated(object sender, EventArgs e) {
			App.ResetFramerate();														// Reset FPS
		}
		#endregion Form_Activated(object sender, EventArgs e)

		#region Form_SizeChanged(object sender, EventArgs e)
		/// <summary>
		/// Handles the form's resizing.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The EventArgs.</param>
		private void Form_SizeChanged(object sender, EventArgs e) {
			// Update Resolution Status Bar Panel
			sbpResolution.Text = this.Size.Width + "x" + this.Size.Height + "x" + App.ColorDepth;
			if(App.Model.HelpForm != null) {
				if(App.IsFullscreen) {
					App.Model.HelpForm.UpdateResolution(App.Width + "x" + App.Height + "x" + App.ColorDepth, "Fullscreen");
				}
				else {
					App.Model.HelpForm.UpdateResolution(this.ClientSize.Width + "x" + this.ClientSize.Height + "x" + App.ColorDepth, this.Size.Width + "x" + this.Size.Height);
				}
			}
			App.ResetFramerate();														// Reset FPS
		}
		#endregion Form_SizeChanged(object sender, EventArgs e)

		#region OnClosing(CancelEventArgs e)
		/// <summary>
		/// Handles the Closing event.
		/// </summary>
		/// <param name="e">The CancelEventArgs.</param>
		protected override void OnClosing(CancelEventArgs e) {
			if(App.Model.IsHelpDisplayed) {
				App.Model.HelpForm.Hide();
			}
			this.Hide();
		}
		#endregion OnClosing(CancelEventArgs e)

		#region Timer_Tick(object sender, System.EventArgs e)
		/// <summary>
		/// Handles The timer's Tick event.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The EventArgs.</param>
		private void Timer_Tick(object sender, System.EventArgs e) {
			if(App.FramerateReady) {
				UpdateStatusBar(App.CurrentFramerate.ToString(), App.HighestFramerate.ToString(), App.LowestFramerate.ToString());
			}
			else {
				UpdateStatusBar("--", "--", "--");
			}
		}
		#endregion Timer_Tick(object sender, System.EventArgs e)
	}
}