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
using System.Data;
using System.Drawing;
using System.Reflection;
using System.Resources;
using System.Windows.Forms;

namespace CsGL.Basecode {
	/// <summary>
	/// Provides a Windows Form for example information and help.
	/// </summary>
	public sealed class HelpForm : Form {
		// --- Fields ---
		#region Private Fields
		private RichTextBox rtbInformation = new RichTextBox();							// RichTextBox For Example Information
		private HelpFormDataGrid dgInputHelp = new HelpFormDataGrid();					// DataGrid For Input Help
		private StatusBar statusBar = new StatusBar();									// The StatusBar Area
		private StatusBarPanel sbpCurrentFps = new StatusBarPanel();					// Panel For Current FPS
		private StatusBarPanel sbpHighestFps = new StatusBarPanel();					// Panel For Highest FPS
		private StatusBarPanel sbpLowestFps = new StatusBarPanel();						// Panel For Lowest FPS
		private StatusBarPanel sbpOpenGlResolution = new StatusBarPanel();				// Panel For OpenGL Resolution
		private StatusBarPanel sbpWindowResolution = new StatusBarPanel();				// Panel For Windows Form Resolution
		private Timer timer = new Timer();												// Generic Timer For Updating StatusBar
		private bool isDisposed = false;												// Has Disposed Been Called For This Instance?
		#endregion Private Fields

		#region Public Properties
		/// <summary>
		/// The HelpForm's input help DataGrid.
		/// </summary>
		public HelpFormDataGrid InputHelpDataGrid {
			get {
				return this.dgInputHelp;
			}
		}
		#endregion Public Properties

		// --- Creation & Destruction Methods ---
		#region HelpForm()
		/// <summary>
		/// Constructor.
		/// </summary>
		public HelpForm() {
			this.SuspendLayout();

			// rtbInformation
			this.rtbInformation.Location = new Point(0, 0);
			this.rtbInformation.Multiline = true;
			this.rtbInformation.Name = "rtbInformation";
			this.rtbInformation.ReadOnly = true;
			this.rtbInformation.ScrollBars = RichTextBoxScrollBars.ForcedVertical;
			this.rtbInformation.SelectionIndent = 5;
			this.rtbInformation.SelectionRightIndent = 5;
			this.rtbInformation.ShowSelectionMargin = true;
			this.rtbInformation.Size = new Size(588, 140);
			this.rtbInformation.TabIndex = 0;
			this.rtbInformation.TabStop = false;

			string info;
			info = "\n" + App.Model.Title + "\n";
			if(App.Model.Url != null) {
				info += App.Model.Url + "\n";
			}
			if(App.Model.Description != null) {
				info += "\n" + App.Model.Description + "\n";
			}
			info += "\n";

			this.rtbInformation.Text = info;
			this.rtbInformation.LinkClicked += new System.Windows.Forms.LinkClickedEventHandler(this.RtbInformation_LinkClicked);

			// dgInputHelp
			this.dgInputHelp.AllowNavigation = false;
			this.dgInputHelp.AllowSorting = false;
			this.dgInputHelp.AlternatingBackColor = Color.LightGray;
			this.dgInputHelp.BackColor = Color.GhostWhite;
			this.dgInputHelp.BackgroundColor = Color.FromArgb(((byte)(207)), ((byte)(207)), ((byte)(207)));
			this.dgInputHelp.BorderStyle = BorderStyle.Fixed3D;
			this.dgInputHelp.CaptionVisible = false;
			this.dgInputHelp.DataMember = "";
			this.dgInputHelp.Enabled = true;
			this.dgInputHelp.FlatMode = false;
			this.dgInputHelp.Font = new Font("Tahoma", 8F);
			this.dgInputHelp.ForeColor = Color.Black;
			this.dgInputHelp.GridLineColor = Color.Black;
			this.dgInputHelp.HeaderBackColor = Color.Black;
			this.dgInputHelp.HeaderFont = new Font("Tahoma", 8F, FontStyle.Bold);
			this.dgInputHelp.HeaderForeColor = Color.White;
			this.dgInputHelp.Location = new Point(0, 140);
			this.dgInputHelp.Name = "dgInputHelp";
			this.dgInputHelp.PreferredColumnWidth = 188;
			this.dgInputHelp.ReadOnly = true;
			this.dgInputHelp.RowHeadersVisible = false;
			this.dgInputHelp.SelectionBackColor = Color.FromArgb(((byte)(248)), ((byte)(131)), ((byte)(131)));
			this.dgInputHelp.SelectionForeColor = Color.Black;
			this.dgInputHelp.Size = new Size(588, 202);
			this.dgInputHelp.TabIndex = 0;
			this.dgInputHelp.TabStop = false;

			// sbpCurrentFps
			this.sbpCurrentFps.Alignment = HorizontalAlignment.Center;
			this.sbpCurrentFps.AutoSize = StatusBarPanelAutoSize.Contents;
			this.sbpCurrentFps.Text = "Current: 2000 FPS";

			// sbpHighestFps
			this.sbpHighestFps.Alignment = HorizontalAlignment.Center;
			this.sbpHighestFps.AutoSize = StatusBarPanelAutoSize.Contents;
			this.sbpHighestFps.Text = "High: 2000 FPS";

			// sbpLowestFps
			this.sbpLowestFps.Alignment = HorizontalAlignment.Center;
			this.sbpLowestFps.AutoSize = StatusBarPanelAutoSize.Contents;
			this.sbpLowestFps.Text = "Low: 2000 FPS";

			// statusBar
			this.statusBar.Panels.AddRange(new StatusBarPanel[] { this.sbpCurrentFps, this.sbpHighestFps, this.sbpLowestFps, this.sbpOpenGlResolution, this.sbpWindowResolution });
			this.statusBar.ShowPanels = true;
			this.statusBar.SizingGrip = false;

			// sbpOpenGlResolution
			this.sbpOpenGlResolution.Alignment = HorizontalAlignment.Center;
			this.sbpOpenGlResolution.AutoSize = StatusBarPanelAutoSize.Contents;
			if(App.IsFullscreen) {
				this.sbpOpenGlResolution.Text = "OpenGL: " + App.Width + "x" + App.Height + "x" + App.ColorDepth;
			}
			else {
				this.sbpOpenGlResolution.Text = "OpenGL: " + App.Form.ClientSize.Width + "x" + App.Form.ClientSize.Height + "x" + App.ColorDepth;
			}

			// sbpWindowResolution
			this.sbpWindowResolution.Alignment = HorizontalAlignment.Center;
			this.sbpWindowResolution.AutoSize = StatusBarPanelAutoSize.Contents;
			if(App.IsFullscreen) {
				this.sbpWindowResolution.Text = "Window: Fullscreen";
			}
			else {
				this.sbpWindowResolution.Text = "Window: " + App.Form.Size.Width + "x" + App.Form.Size.Height;
			}

			// timer
			timer.Interval = 500;
			timer.Enabled = true;
			timer.Tick += new EventHandler(Timer_Tick);

			// this
			this.BackColor = Color.FromArgb(((byte)(207)), ((byte)(207)), ((byte)(207)));
			this.Controls.AddRange(new Control[] { this.rtbInformation, this.dgInputHelp, this.statusBar });
			this.FormBorderStyle = FormBorderStyle.Fixed3D;

			if(App.Model.WindowIcon == null) {
				ResourceManager rm = new ResourceManager("BasecodeResources", Assembly.GetCallingAssembly());
				this.Icon = (Icon) rm.GetObject("AppFormIcon");
			}
			else {
				this.Icon = (Icon) App.Model.WindowIcon;
			}

			this.MaximizeBox = false;
			this.Size = new Size(600, 400);
			this.StartPosition = FormStartPosition.CenterScreen;
			this.Text = "CsGL Basecode Help";
			this.TopMost = true;
			this.Closed += new EventHandler(this.HelpForm_Closed);

			App.Model.IsHelpDisplayed = true;

			this.ResumeLayout();
		}
		#endregion HelpForm()

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
					if(rtbInformation != null) {
						rtbInformation.Dispose();
					}
					if(dgInputHelp != null) {
						dgInputHelp.Dispose();
					}
					if(sbpCurrentFps != null) {
						sbpCurrentFps.Dispose();
					}
					if(sbpHighestFps != null) {
						sbpHighestFps.Dispose();
					}
					if(sbpLowestFps != null) {
						sbpLowestFps.Dispose();
					}
					if(sbpOpenGlResolution != null) {
						sbpOpenGlResolution.Dispose();
					}
					if(sbpWindowResolution != null) {
						sbpWindowResolution.Dispose();
					}
					if(statusBar != null) {
						statusBar.Dispose();
					}
					if(timer != null) {
						timer.Dispose();
					}
					GC.SuppressFinalize(this);											// Prevent Being Added To The Finalization Queue
				}

				// Release Any Unmanaged Resources Here, If disposing Was false, Only The Following Code Is Executed
				rtbInformation = null;
				dgInputHelp = null;
				sbpCurrentFps = null;
				sbpHighestFps = null;
				sbpLowestFps = null;
				sbpOpenGlResolution = null;
				sbpWindowResolution = null;
				statusBar = null;
				timer = null;
			}
			isDisposed = true;															// Mark As disposed
		}
		#endregion Dispose(bool disposing)

		#region Finalizer
		/// <summary>
		/// This destructor will run only if the Dispose method does not get called.  It gives 
		/// the class the opportunity to finalize.  Simply calls Dispose(false).
		/// </summary>
		~HelpForm() {
			Dispose(false);																// We've Automatically Called For A Dispose
		}
		#endregion Finalizer

		// --- Methods ---
		#region UpdateStatusBar(string current, string highest, string lowest)
		/// <summary>
		/// Updates framerate display in StatusBar.
		/// </summary>
		/// <param name="current">Current FPS.</param>
		/// <param name="highest">Highest FPS.</param>
		/// <param name="lowest">Lowest FPS.</param>
		public void UpdateStatusBar(string current, string highest, string lowest) {
			sbpCurrentFps.Text = "Current: " + current + " FPS";						// Set The Current FPS Text
			sbpHighestFps.Text = "High: " + highest + " FPS";							// Set The Highest FPS Text
			sbpLowestFps.Text = "Low: " + lowest + " FPS";								// Set The Lowest FPS Text
		}
		#endregion UpdateStatusBar(string current, string highest, string lowest)

		#region UpdateResolution(string opengl, string window)
		/// <summary>
		/// Updates resolution display in StatusBar.
		/// </summary>
		/// <param name="opengl">OpenGL Resolution.</param>
		/// <param name="window">Window Resolution.</param>
		public void UpdateResolution(string opengl, string window) {
			if(sbpOpenGlResolution != null) {
				sbpOpenGlResolution.Text = "OpenGL: " + opengl;							// Set The OpenGL Resolution Text
			}
			if(sbpWindowResolution != null) {
				sbpWindowResolution.Text = "Window: " + window;							// Set The Window Resolution Text
			}
		}
		#endregion UpdateResolution(string opengl, string window)

		// --- Events ---
		#region HelpForm_Closed(object sender, EventArgs e)
		/// <summary>
		/// Handles the help form's Closed event.  Marks the help form as not displayed.
		/// </summary>
		/// <param name="sender">The Sender.</param>
		/// <param name="e">The EventArgs.</param>
		private void HelpForm_Closed(object sender, EventArgs e) {
			App.Model.IsHelpDisplayed = false;											// Mark The Help Screen As Not Displayed
			if(App.IsFullscreen && !App.ShowCursorFullscreen) {							// If We're In Fullscreen Mode & Not Supposed To Show The Cursor
				Cursor.Hide();															// Hide The Cursor
			}
		}
		#endregion HelpForm_Closed(object sender, EventArgs e)

		#region OnKeyDown(KeyEventArgs e)
		/// <summary>
		/// Handles the KeyDown event.
		/// </summary>
		/// <param name="e">The KeyEventArgs.</param>
		protected override void OnKeyDown(KeyEventArgs e) {
			if(e.KeyCode == Keys.Escape) {
				this.Close();
			}
			base.OnKeyDown(e);
		}
		#endregion OnKeyDown(KeyEventArgs e)

		#region RtbInformation_LinkClicked(object sender, LinkClickedEventArgs e)
		/// <summary>
		/// Handles the Link Clicked event, launches URL in browser.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The LinkClickedEventArgs.</param>
		private void RtbInformation_LinkClicked(object sender, System.Windows.Forms.LinkClickedEventArgs e) {
			try {
				// A Normal Process.Start(e.LinkText) Should Work, But Does Not...  Odd.
				System.Diagnostics.Process.Start("iexplore.exe", e.LinkText);
			}
			catch(System.ComponentModel.Win32Exception noBrowser) {
				MessageBox.Show(noBrowser.Message);
			}
		}
		#endregion RtbInformation_LinkClicked(object sender, LinkClickedEventArgs e)

		#region Timer_Tick(object sender, System.EventArgs e)
		/// <summary>
		/// Handles the timer's Tick event.
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