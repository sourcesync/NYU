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
using System.Windows.Forms;

namespace CsGL.Basecode {
	/// <summary>
	/// Custom DataGrid to enable ScrollBar and override editing mode.  This is FUGLY!
	/// </summary>
	public class HelpFormDataGrid : DataGrid {
		// --- Fields ---
		#region Private Fields
		private const int CAPTIONHEIGHT = 21;											// Height Of DataGrid's Caption
		private const int BORDERWIDTH = 2;												// Width Of DataGrid's Border
		private const int WM_KEYDOWN = 0x100;											// The Window Message KeyDown Event Value
		private int currentRow = -1;													// The Currently Selected Row
		private int scrollRow = 0;														// The ScrollBar's Selected Row
		private bool isDisposed = false;												// Has Disposed Been Called For This Instance?
		#endregion Private Fields

		// --- Creation & Destruction Methods ---
		#region HelpFormDataGrid()
		/// <summary>
		/// Constructor.
		/// </summary>
		public HelpFormDataGrid() {
			this.VertScrollBar.Visible = true;											// Force ScrollBar Visible
			this.VertScrollBar.VisibleChanged += new EventHandler(ShowScrollBars);		// ""
			this.VertScrollBar.Scroll += new ScrollEventHandler(Scroller);				// Event Handler For The ScrollBar's Scroll Event
		}
		#endregion HelpFormDataGrid()

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
					GC.SuppressFinalize(this);											// Prevent Being Added To The Finalization Queue
				}

				// Release Any Unmanaged Resources Here, If disposing Was false, Only The Following Code Is Executed
			}
			isDisposed = true;															// Mark As disposed
		}
		#endregion Dispose(bool disposing)

		#region Finalizer
		/// <summary>
		/// This destructor will run only if the Dispose method does not get called.  It gives 
		/// the class the opportunity to finalize.  Simply calls Dispose(false).
		/// </summary>
		~HelpFormDataGrid() {
			Dispose(false);																// We've Automatically Called For A Dispose
		}
		#endregion Finalizer

		// --- Public Methods ---
		#region ScrollToRow()
		/// <summary>
		/// Scrolls to a particular row in the DataGrid.
		/// </summary>
		public void ScrollToRow() {
			if(this.DataSource != null) {
				this.GridVScrolled(this, new ScrollEventArgs(ScrollEventType.LargeIncrement, scrollRow));
			}
		}
		#endregion ScrollToRow()

		// --- Events ---
		#region OnDataSourceChanged(EventArgs e)
		/// <summary>
		/// Overrides the DataGrid's DataSource Changed event.
		/// </summary>
		/// <param name="e">The EventArgs.</param>
		protected override void OnDataSourceChanged(EventArgs e) {
			if(currentRow != -1) {														// If The Previously Selected Row Was Set
				this.Select(currentRow);												// Reselect The Previously Selected Row
			}
		}
		#endregion OnDataSourceChanged(EventArgs e)

		#region OnMouseDown(MouseEventArgs e)
		/// <summary>
		/// Overrides the DataGrid's Mouse Down event.
		/// </summary>
		/// <param name="e">The MouseEventArgs.</param>
		protected override void OnMouseDown(MouseEventArgs e) {
			HitTestInfo hti = this.HitTest(new Point(e.X, e.Y));						// Test Where The Hit Occurred
			if(hti.Type == HitTestType.Cell) {											// If It Was A Cell
				if(hti.Row != currentRow) {												// Was This Not Our Previously Selected Row?
					if(currentRow != -1) {												// Was This Not An Invalid Row?
						this.UnSelect(currentRow);										// Unselect The Previously Selected Row
					}
					currentRow = hti.Row;												// Set The New Row To currentRow
					this.Select(hti.Row);												// Select The New Row
				}
				else {																	// Otherwise, We Selected The Same Row As Previous
					currentRow = -1;													// Set currentRow To Invalid
					this.UnSelect(hti.Row);												// Unselect The Previously Selected Row
				}
			}
			CancelEditing();															// Cancel Editing
		}
		#endregion OnMouseDown(MouseEventArgs e)

		#region OnMouseMove(MouseEventArgs e)
		/// <summary>
		/// Overrides the DataGrid's Mouse Move event.
		/// </summary>
		/// <param name="e">The MouseEventArgs.</param>
		protected override void OnMouseMove(MouseEventArgs e) {
			HitTestInfo hti = this.HitTest(new Point(e.X, e.Y));						// Test Where The Hit Occurred
			// If The Hit Was A Column Resize Or A Row Resize Attempt
			if(hti.Type == HitTestType.ColumnResize || hti.Type == HitTestType.RowResize) {
				return;																	// Exit & Do Nothing, Preventing The Resize
			}
		}
		#endregion OnMouseMove(MouseEventArgs e)

		#region OnMouseWheel(MouseEventArgs e)
		/// <summary>
		/// Overrides the DataGrid's Mouse Wheel event.
		/// </summary>
		/// <param name="e">The MouseEventArgs.</param>
		protected override void OnMouseWheel(MouseEventArgs e) {
			base.OnMouseWheel(e);														// Let The Wheel Do Its Scrolling
			
			// This Is A Bit Of A Hack To Find The Scrolled Position
			if(e.Delta <= -120) {														// MouseWheel Down
				scrollRow += SystemInformation.MouseWheelScrollLines;					// Set The New Scroll Row
			}
			if(e.Delta >= 120) {														// MouseWheel Up
				scrollRow -= SystemInformation.MouseWheelScrollLines;					// Set The New Scroll Row
			}
			
			// However, The Scrolling Sets The CurrentCell, So We Have To Cancel Out Of Editing Mode
			CancelEditing();
		}
		#endregion OnMouseWheel(MouseEventArgs e)

		#region bool PreProcessMessage(ref Message msg)
		/// <summary>
		/// Overrides the PreProcessMessage event.
		/// </summary>
		/// <param name="msg">The message.</param>
		/// <returns>Boolean indicating if the message was processed.</returns>
		public override bool PreProcessMessage(ref Message msg) {
			// We're Doing This To Not Allow Tabbing Within The DataGrid...
			Keys keyCode = (Keys) (int)msg.WParam & Keys.KeyCode;						// Get The KeyCode Of The Pressed Key From The Message
			if((msg.Msg == WM_KEYDOWN) && keyCode == Keys.Tab) {						// If It Was The TAB Key
				return true;															// Mark It As Processed
			}	
			return base.PreProcessMessage(ref msg);										// Otherwise, Handle The Message As Normal
		}
		#endregion bool PreProcessMessage(ref Message msg)

		#region Scroller(object sender, ScrollEventArgs e)
		/// <summary>
		/// Overrides DataGrid's Vertical ScrollBar's Scroll event.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The ScrollEventArgs.</param>
		private void Scroller(object sender, ScrollEventArgs e) {
			// Scrolling Seems To Set CurrentCell, We Can Get Out Of This By Running CancelEdit()
			scrollRow = e.NewValue;														// Get The New Scroll Position
			CancelEditing();															// Cancel Editing
			if(currentRow != -1) {														// If We Had A Row Highlighted Previously
				this.Select(currentRow);												// Reselect That Row
			}
		}
		#endregion Scroller(object sender, ScrollEventArgs e)

		#region ShowScrollBars(object sender, EventArgs e)
		/// <summary>
		/// Show the DataGrid's Vertical ScrollBar at all times.
		/// </summary>
		/// <param name="sender">The sender.</param>
		/// <param name="e">The EventArgs.</param>
		private void ShowScrollBars(object sender, EventArgs e) {
			if(!this.VertScrollBar.Visible) {											// If The ScrollBar Isn't Currently Visible
				// Properly Position The ScrollBar
				int width = this.VertScrollBar.Width;
				this.VertScrollBar.Location = new Point(this.ClientRectangle.Width - width - BORDERWIDTH, 0);
				this.VertScrollBar.Size = new Size(width, this.ClientRectangle.Height - BORDERWIDTH);
				this.VertScrollBar.Show();												// Show The ScrollBar
			}
		}
		#endregion ShowScrollBars(object sender, EventArgs e)
	}
}