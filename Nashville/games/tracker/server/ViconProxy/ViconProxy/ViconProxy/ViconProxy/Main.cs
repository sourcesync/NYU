#region BSD License
/*
 BSD License
Copyright (c) 2002, Randy Ridge, The CsGL Development Team
http://csgl.sourceforge.net/
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

#region Original Credits / License
/*
 * Copyright (c) 1993-1997, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED 
 * Permission to use, copy, modify, and distribute this software for 
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that 
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission. 
 *
 * THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 * 
 * US Government Users Restricted Rights 
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(R) is a registered trademark of Silicon Graphics, Inc.
 */
#endregion Original Credits / License

using CsGL.Basecode;
using System;
using System.Data;
using System.Reflection;
using System.Windows.Forms;
using System.Net;
using System.Net.Sockets;
using System.Collections;
using Newtonsoft.Json;
using System.Runtime.Serialization;
using System.IO;

#region AssemblyInfo
[assembly: AssemblyCompany("The CsGL Development Team (http://csgl.sourceforge.net)")]
[assembly: AssemblyCopyright("2002 The CsGL Development Team (http://csgl.sourceforge.net)")]
[assembly: AssemblyDescription("Redbook Torus")]
[assembly: AssemblyProduct("Redbook Torus")]
[assembly: AssemblyTitle("Redbook Torus")]
[assembly: AssemblyVersion("1.0.0.0")]
#endregion AssemblyInfo







namespace RedbookExamples {



    public enum Result
    {
        Unknown,
        NotImplemented,
        Success,
        InvalidHostName,
        InvalidMulticastIP,
        ClientAlreadyConnected,
        ClientConnectionFailed,
        ServerAlreadyTransmittingMulticast,
        ServerNotTransmittingMulticast,
        NotConnected,
        NoFrame,
        InvalidIndex,
        InvalidSubjectName,
        InvalidSegmentName,
        InvalidMarkerName,
        InvalidDeviceName,
        InvalidDeviceOutputName,
        InvalidLatencySampleName,
        CoLinearAxes,
        LeftHandedAxes
    };

    //gw
    public class ViconMarker
    {
        public String name = String.Empty;
        public int[] t = new int[3] { 0, 0, 0 };
        public bool oc = false;
    }
    public class ViconObject
    {
        public String name = String.Empty;
        public int[] t = new int[3] { 0, 0, 0 }; 
        //public int[] lt = new double[3] { 0.0, 0.0, 0.0 };
        public double[] r = new double[3] { 0.0, 0.0, 0.0 };
        public uint sc = 0;
        public uint mc = 0;
        public System.Collections.ArrayList mks = null;
        //public double[] ct = new double[3] { 0.0, 0.0, 0.0 };
        public bool oc = false;
    }
    public class ViconData
    {
        public uint v = RedbookTorus.PROTOCOL_VERSION;
        public uint fno = 0;
        public uint objno = 0;
        public uint mode = 0;
        public System.Collections.ArrayList objs = new System.Collections.ArrayList();
    }
    //gw

	/// <summary>
	/// Redbook Torus -- Display Lists (http://www.opengl.org/developers/code/examples/redbook/redbook.html)
	/// Implemented In C# By The CsGL Development Team (http://csgl.sourceforge.net)
	/// </summary>
	public sealed class RedbookTorus : Model {
		// --- Fields ---
		#region Private Fields
		private static float PI_ = 3.14159265358979323846f;
		private static uint theTorus;

        //gw

        //  some constants...
        public const int PROTOCOL_VERSION = 1;
        public const ushort DEFAULT_SIMPLE_BROADCAST_SERVER_PORT = 6666;
        public const ushort DEFAULT_JSON_BROADCAST_SERVER_PORT = 6667;
        public const ushort DEFAULT_LISTENER_SERVER_PORT = 6868;
        public const String DEFAULT_VICON_ADDR = "localhost:801";
        public const String CALIB_PREFIX = "calib";
        public const int DEFAULT_TIMER_INTERVAL = 10;

        //  vicon objects...
        public static ViconDataStreamSDK.DotNET.Client MyClient;
        public static int packet_counter = 0;
        //  timer stuff...
        public static System.Timers.ElapsedEventHandler timer_handler;
        public static System.Timers.Timer timer;
        public static int timer_time = DEFAULT_TIMER_INTERVAL;
        //  local udp socket...
        public static IPAddress local_address;
        public static IPEndPoint local_ep;
        public static UdpClient local_socket;
        //  remote broadcast socket stuff...
        public static IPAddress simple_broadcast_address;
        public static IPEndPoint simple_broadcast_ep;
        public static IPAddress json_broadcast_address;
        public static IPEndPoint json_broadcast_ep;
        //  local listener socket...
        public static IPAddress local_listener_address;
        public static IPEndPoint local_listener_ep;
        public static Socket local_listener;
        //  socket collections...
        public static System.Collections.Hashtable sockets = new System.Collections.Hashtable();

        //  stats...
        public static int n_sends = 0;
        public static System.DateTime startime = System.DateTime.Now;
        public static double fps = 0.0;

        //gw

		#endregion Private Fields

		#region Public Properties
		/// <summary>
		/// Example title.
		/// </summary>
		public override string Title {
			get {
				return "Redbook Torus -- Display Lists";
			}
		}

		/// <summary>
		/// Example description.
		/// </summary>
		public override string Description {
			get {
				return "This program demonstrates the creation of a display list.";
			}
		}

		/// <summary>
		/// Example URL.
		/// </summary>
		public override string Url {
			get {
				return "http://www.opengl.org/developers/code/examples/redbook/redbook.html";
			}
		}
		#endregion Public Properties

        public static void Quit( String msg, int val )
        {
            Console.WriteLine(msg);
            System.Threading.Thread.Sleep(5000);
            System.Environment.Exit(val);
        }

		// --- Entry Point ---
		#region Main()
		/// <summary>
		/// Application's entry point, runs this Redbook example.
		/// </summary>
		public static void Main() {														// Entry Point
            bool raw_mode = false;

            //  See if we are explicitly setting reg/raw mode...
            if (System.Environment.GetCommandLineArgs().Length == 5)
            {
                // get reg/raw mode...
                String raw = System.Environment.GetCommandLineArgs()[3];
                if (raw == "1")
                {
                    Console.WriteLine("INFO: Starting raw mode.");
                    raw_mode = true;
                }
                //  get timer time...
                String tt = System.Environment.GetCommandLineArgs()[4];
                timer_time = int.Parse(tt);

            }
            else if (System.Environment.GetCommandLineArgs().Length == 4)
            {
                String raw = System.Environment.GetCommandLineArgs()[3];
                if (raw == "1")
                {
                    Console.WriteLine("INFO: Starting raw mode.");
                    raw_mode = true;
                }
             }
            // else the minimum num args are specified...
            else if (System.Environment.GetCommandLineArgs().Length != 3)
            {
                Quit("ERROR: Usage:  ViconProxy.exe <broadcast ip:port> <vicon_ip:vicon_port> <raw_mode_flag>", 1);
                return;
            }

            //  Parse broadcast address...
            String broadcast_addr = System.Environment.GetCommandLineArgs()[1];
            int broadcast_port = DEFAULT_JSON_BROADCAST_SERVER_PORT;
            //  If has colon, sep into addr and port...
            String[] parts = broadcast_addr.Split( new char[] {':'});
            if (parts.Length > 1)
            {
                broadcast_addr = parts[0];
                broadcast_port = int.Parse(parts[1]);
                Console.WriteLine(String.Format("INFO: overriding default json broadcast port -> {0}...", broadcast_port));
                System.Threading.Thread.Sleep(3000);
            }

            //  Initialize vicon related stuff...
            Console.WriteLine(String.Format("INFO: vicon init start -> {0} ...", System.Environment.GetCommandLineArgs()[2]));
            MyClient = new ViconDataStreamSDK.DotNET.Client();
            

            //  Check if already connected...
            ViconDataStreamSDK.DotNET.Output_IsConnected output2 = MyClient.IsConnected();
            //if (!output2.Connected)
            //{
            //    Console.WriteLine("WARNING: That's weird, already connected...");
            //    System.Threading.Thread.Sleep(3);
            //}
            //else
            {

                //MyClient.Connect(VICON_ADDR);
                
                MyClient.Connect(System.Environment.GetCommandLineArgs()[2]);
                ;
                //  Better be connected now...
                ViconDataStreamSDK.DotNET.Output_IsConnected output3 = MyClient.IsConnected();
                if (!output3.Connected)
                {
                    Quit("ERROR: Cannot connect to vicon.  Check that the system is on and Vicon software sees it.... ", 1);
                    return;
                }
            }

            MyClient.EnableDeviceData();
            MyClient.EnableSegmentData();

            if (raw_mode)
            {
                MyClient.EnableUnlabeledMarkerData();
            }

            MyClient.GetFrame();
            ViconDataStreamSDK.DotNET.Output_GetFrameNumber frame_no = MyClient.GetFrameNumber();
            MyClient.EnableMarkerData();
            ViconDataStreamSDK.DotNET.Output_IsUnlabeledMarkerDataEnabled Output4 = MyClient.IsUnlabeledMarkerDataEnabled();
            ViconDataStreamSDK.DotNET.Output_GetDeviceCount dc = MyClient.GetDeviceCount();
            Console.WriteLine("INFO: vicon init done...");

            //  Initialize udp socket broadcast related stuff...
            try
            {
                local_address = IPAddress.Parse(broadcast_addr);
                local_ep = new IPEndPoint(local_address, 0);
                local_socket = new UdpClient(local_ep);
                simple_broadcast_address = IPAddress.Parse("255.255.255.255");
                simple_broadcast_ep = new IPEndPoint(simple_broadcast_address, DEFAULT_SIMPLE_BROADCAST_SERVER_PORT);
                json_broadcast_address = IPAddress.Parse("255.255.255.255");
                json_broadcast_ep = new IPEndPoint(json_broadcast_address, broadcast_port);

                //  Initialize tcp socket server related stuff...
                local_listener_address = IPAddress.Parse(broadcast_addr);
                local_listener_ep = new IPEndPoint(local_listener_address, DEFAULT_LISTENER_SERVER_PORT);
                local_listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                local_listener.Bind(local_listener_ep);
                local_listener.Listen(100);
                local_listener.Blocking = false;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
                Quit("ERROR: Something went wrong with setting up sockets...", 1);
                return;
            }  


            // Initialize timer stuff...
            if (raw_mode)
            {
                timer_handler = new System.Timers.ElapsedEventHandler(timer_elapsed_raw);
            }
            else
            {
                timer_handler = new System.Timers.ElapsedEventHandler(timer_elapsed);
            }
            timer = new System.Timers.Timer(timer_time);  // TODO: should be arg...
            timer.Elapsed += timer_handler;
            timer.Enabled = true;
            timer.Start();

            //  start the app...
			//App.Run(new RedbookTorus());												// Run Our Example As A Windows Forms Application
            Application.Run();
		}
		#endregion Main()

		// --- Basecode Methods ---
		#region Initialize()
		/// <summary>
		/// Overrides OpenGL's initialization.
		/// </summary>
		public override void Initialize() {
			// Create display list with Torus and initialize state
			theTorus = glGenLists(1);
			glNewList(theTorus, GL_COMPILE);
				Torus(8, 25);
			glEndList();

			glShadeModel(GL_FLAT);
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		}
		#endregion Initialize()

		#region Draw()
		/// <summary>
		/// Draws Redbook Torus scene.
		/// </summary>
		public override void Draw() {													// Here's Where We Do All The Drawing
			// Clear window and draw torus
			glClear(GL_COLOR_BUFFER_BIT);
			glColor3f(1.0f, 1.0f, 1.0f);
			glCallList(theTorus);
			glFlush();
		}
		#endregion Draw()

		#region InputHelp()
		/// <summary>
		/// Overrides default input help, supplying example-specific help information.
		/// </summary>
		public override void InputHelp() {
			base.InputHelp();															// Set Up The Default Input Help

			DataRow dataRow;															// Row To Add

			dataRow = InputHelpDataTable.NewRow();										// X - Rotate About X-Axis
			dataRow["Input"] = "X";
			dataRow["Effect"] = "Rotate About X-Axis";
			dataRow["Current State"] = "";
			InputHelpDataTable.Rows.Add(dataRow);

			dataRow = InputHelpDataTable.NewRow();										// Y - Rotate About Y-Axis
			dataRow["Input"] = "Y";
			dataRow["Effect"] = "Rotate About Y-Axis";
			dataRow["Current State"] = "";
			InputHelpDataTable.Rows.Add(dataRow);

			dataRow = InputHelpDataTable.NewRow();										// R - Reset
			dataRow["Input"] = "R";
			dataRow["Effect"] = "Reset";
			dataRow["Current State"] = "";
			InputHelpDataTable.Rows.Add(dataRow);
		}
		#endregion InputHelp()

		#region ProcessInput()
		/// <summary>
		/// Overrides default input handling, adding example-specific input handling.
		/// </summary>
		public override void ProcessInput() {
			base.ProcessInput();														// Handle The Default Basecode Keys

			if(KeyState[(int) Keys.X]) {												// Is X Key Being Pressed?
				KeyState[(int) Keys.X] = false;											// Mark As Handled
				glRotatef(30.0f, 1.0f, 0.0f, 0.0f);										// Rotate About X-Axis
			}

			if(KeyState[(int) Keys.Y]) {												// Is Y Key Being Pressed?
				KeyState[(int) Keys.Y] = false;											// Mark As Handled
				glRotatef(30.0f, 0.0f, 1.0f, 0.0f);										// Rotate About Y-Axis
			}

			if(KeyState[(int) Keys.R]) {												// Is R Key Being Pressed?
				KeyState[(int) Keys.R] = false;											// Mark As Handled
				glLoadIdentity();														// Reset View
				gluLookAt(0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
			}
		}
		#endregion ProcessInput()

		#region Reshape(int width, int height)
		/// <summary>
		/// Overrides OpenGL reshaping.
		/// </summary>
		/// <param name="width">New width.</param>
		/// <param name="height">New height.</param>
		public override void Reshape(int width, int height) {							// Resize And Initialize The GL Window
			glViewport(0, 0, width, height);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluPerspective(30.0f, (float) width / (float) height, 1.0f, 100.0f);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			gluLookAt(0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
		}
		#endregion Reshape(int width, int height)

		// --- Example Methods ---
		#region Torus(int numc, int numt)
		/// <summary>
		/// Draws a torus.
		/// </summary>
		/// <param name="numc"></param>
		/// <param name="numt"></param>
		private static void Torus(int numc, int numt) {
			int i, j, k;
			double s, t, x, y, z, twopi;

			twopi = 2 * PI_;
			for(i = 0; i < numc; i++) {
				glBegin(GL_QUAD_STRIP);
					for(j = 0; j <= numt; j++) {
						for(k = 1; k >= 0; k--) {
							s = (i + k) % numc + 0.5;
							t = j % numt;

							x = (1 + 0.1 * Math.Cos(s * twopi / numc)) * Math.Cos(t * twopi / numt);
							y = (1 + 0.1 * Math.Cos(s * twopi / numc)) * Math.Sin(t * twopi / numt);
							z = 0.1 * Math.Sin(s * twopi / numc);
							glVertex3f((float) x, (float) y, (float) z);
						}
					}
				glEnd();
			}
		}
		#endregion Torus(int numc, int numt)

        //gw - timer that fires to query vicon and broadcast to udp...
        public static void timer_elapsed(object sender, System.Timers.ElapsedEventArgs args)
        {
            try
            {
                //  stats stuff...
                n_sends++;
                long diffticks= System.DateTime.Now.Ticks - startime.Ticks;
                float diffsecs = diffticks * 1.0f / System.TimeSpan.TicksPerSecond;
                if (diffsecs > 1.0f)
                {
                        fps = n_sends * 1.0f / diffsecs;
                        startime = System.DateTime.Now;
                        n_sends = 0;
                        String statsstr = String.Format("INFO: fps={0}...", fps.ToString());
                        Console.WriteLine(statsstr);
                }

                //  Get the frame number...
                MyClient.GetFrame();
                ViconDataStreamSDK.DotNET.Output_GetFrameNumber frame_no = MyClient.GetFrameNumber();

                String fno_str = String.Format("INFO: broadcasting data from vicon frame_no={0}...", frame_no.FrameNumber.ToString());
                Console.WriteLine(fno_str);

                //  Get the subject count...
                ViconDataStreamSDK.DotNET.Output_GetSubjectCount sc = MyClient.GetSubjectCount();

                packet_counter++;
                //  Start a simple packet...
                //  String packet = String.Format("ViconProxy,{0},Data,{1},{2},{3}\n", PROTOCOL_VERSION, packet_counter, frame_no.FrameNumber, sc.SubjectCount);

                //  Start an object for json...
                ViconData vdata = new ViconData();
                vdata.v = PROTOCOL_VERSION;
                vdata.fno = frame_no.FrameNumber;

                //  Iterate over subjects and complete the data packet...
                for (uint i=0;i<sc.SubjectCount;i++)
                {
                    //  Get subject name...
                    ViconDataStreamSDK.DotNET.Output_GetSubjectName OutputGSN = MyClient.GetSubjectName(i);

                    //  Get subject count...
                    ViconDataStreamSDK.DotNET.Output_GetSegmentCount segc = MyClient.GetSegmentCount(OutputGSN.SubjectName);

                    //  Get coordinates of first segment, if any...
                    if (segc.SegmentCount == 1)
                    {
                        //  Get segment name of 0th segment...
                        ViconDataStreamSDK.DotNET.Output_GetSegmentName segname = MyClient.GetSegmentName(OutputGSN.SubjectName, 0);

                        //  Get translation...
                        ViconDataStreamSDK.DotNET.Output_GetSegmentGlobalTranslation dt = MyClient.GetSegmentGlobalTranslation(
                            OutputGSN.SubjectName, segname.SegmentName);
                      
                        //  Get local translation...
                        ViconDataStreamSDK.DotNET.Output_GetSegmentLocalTranslation dlt = MyClient.GetSegmentLocalTranslation(
                            OutputGSN.SubjectName, segname.SegmentName);

                        //  Get rotation...
                        ViconDataStreamSDK.DotNET.Output_GetSegmentGlobalRotationEulerXYZ dr = MyClient.GetSegmentGlobalRotationEulerXYZ(
                            OutputGSN.SubjectName, segname.SegmentName);

                        //  append to simple packet...
                        //packet += String.Format("{0},{1},{2},{3},{4}\n", OutputGSN.SubjectName, segc.SegmentCount,
                        //    dt.Translation[0], dt.Translation[1], dt.Translation[2]);

                        //  append to object for json...
                        vdata.objno += 1;
                        ViconObject tdata = new ViconObject();
                        tdata.name = OutputGSN.SubjectName;
                        tdata.sc = segc.SegmentCount;
                        tdata.t[0] = (int)dt.Translation[0];
                        tdata.t[1] = (int)dt.Translation[1];
                        tdata.t[2] = (int)dt.Translation[2];
                        //tdata.lt[0] = dlt.Translation[0];
                        //tdata.lt[1] = dlt.Translation[1];
                        //tdata.lt[2] = dlt.Translation[2];
                        tdata.r[0] = (double)dr.Rotation[0];
                        tdata.r[1] = (double)dr.Rotation[1];
                        tdata.r[2] = (double)dr.Rotation[2];
                        tdata.oc = dt.Occluded;
                        
                        vdata.objs.Add(tdata);

                        //  Get marker count...
                        ViconDataStreamSDK.DotNET.Output_GetMarkerCount mc = MyClient.GetMarkerCount(OutputGSN.SubjectName);
                        tdata.mc = mc.MarkerCount;

                        /*
                        //  Iterate over markers and compute avg...
                        if (mc.MarkerCount > 0)
                        {
                            uint avg_count = mc.MarkerCount;
                            for (uint m = 0; m < mc.MarkerCount; m++)
                            {
                                //  Get marker's name...
                                ViconDataStreamSDK.DotNET.Output_GetMarkerName OutputGMN = MyClient.GetMarkerName(OutputGSN.SubjectName, m);
                                //  Get its translation...
                                ViconDataStreamSDK.DotNET.Output_GetMarkerGlobalTranslation mtr =
                                    MyClient.GetMarkerGlobalTranslation(OutputGSN.SubjectName, OutputGMN.MarkerName);
                                if (mtr.Translation[0] < 0.00001 && mtr.Translation[1] < 0.00001 && mtr.Translation[2] < 0.00001)
                                {
                                    avg_count--;
                                }
                                else
                                {
                                    tdata.ct[0] += mtr.Translation[0];
                                    tdata.ct[1] += mtr.Translation[1];
                                    tdata.ct[2] += mtr.Translation[2];
                                }
                            }
                            tdata.ct[0] = tdata.ct[0] / avg_count;
                            tdata.ct[1] = tdata.ct[1] / avg_count;
                            tdata.ct[2] = tdata.ct[2] / avg_count;
                        }
                         * */

                        //  Iterate over subject markers and get info if its a special calibration object...
                        if ( (mc.MarkerCount > 0) && ( OutputGSN.SubjectName.ToLower().StartsWith(CALIB_PREFIX) ) )
                        {
                            tdata.mks = new System.Collections.ArrayList();
                            for (uint m = 0; m < mc.MarkerCount; m++)
                            {
                                //  Get marker's name...
                                ViconDataStreamSDK.DotNET.Output_GetMarkerName OutputGMN = MyClient.GetMarkerName(OutputGSN.SubjectName, m);
                                ViconMarker mdata = new ViconMarker();
                                mdata.name = OutputGMN.MarkerName;
                                //  Get its translation...
                                ViconDataStreamSDK.DotNET.Output_GetMarkerGlobalTranslation mtr =
                                    MyClient.GetMarkerGlobalTranslation(OutputGSN.SubjectName, OutputGMN.MarkerName);
                                mdata.t[0] = (int)mtr.Translation[0];
                                mdata.t[1] = (int)mtr.Translation[1];
                                mdata.t[2] = (int)mtr.Translation[2];
                                mdata.oc = mtr.Occluded;
                                tdata.mks.Add(mdata);
                            }
                        }
                    }
                    else
                    {
                        //  append to simple packet...
                        // packet += String.Format("{0},{1}\n", OutputGSN.SubjectName, segc.SegmentCount);

                        //  append to object for json...
                        vdata.objno += 1;
                        ViconObject tdata = new ViconObject();
                        tdata.name = OutputGSN.SubjectName;
                        tdata.sc = segc.SegmentCount;
                        vdata.objs.Add(tdata);
                    }
                }

                /*
                //  Broadcast the simple packet...
                System.Text.ASCIIEncoding encoding = new System.Text.ASCIIEncoding();
                byte[] b = encoding.GetBytes(packet);
                int sent = local_socket.Send(b, b.Length, simple_broadcast_ep);
                if (sent!=b.Length)
                {
                    Console.WriteLine("WARNING: packet send/sent size mismatch!");
                }
                 * */
                vdata.mode = 0;

                //  Broadcast the json packet...
                System.Text.ASCIIEncoding encoding = new System.Text.ASCIIEncoding();
                JsonSerializer serializer = new JsonSerializer();
                StringWriter str = new StringWriter();
                JsonWriter writer = new JsonTextWriter(str);
                serializer.Serialize(writer, vdata);
                System.Text.ASCIIEncoding encoding2 = new System.Text.ASCIIEncoding();
                byte[] b2 = encoding.GetBytes(str.ToString());
                int sent = local_socket.Send(b2, b2.Length, json_broadcast_ep);
                if (sent != b2.Length)
                {
                    Console.WriteLine("WARNING: packet send/sent size mismatch!");
                }

                /*
                //  Check for direct tcp server connects...
                try
                {
                    //  Check for explicit connects...
                    Socket accepted_socket = local_listener.Accept();
                    accepted_socket.Blocking = false;
                    accepted_socket.NoDelay = true;
                    sockets.Add(accepted_socket, accepted_socket);
                }
                catch
                {
                    //  Got here, means nothing to accept (likely)...
                }

                //  Send to any tcp connections...
                foreach (Socket s in sockets.Keys)
                {
                    Console.WriteLine("INFO: Also broadcasting to...");
                    sent = s.Send(b);
                    if (sent != b.Length)
                    {
                        Console.WriteLine("WARNING: packet send/sent size mismatch!");
                    }
                }
                 */

                //Console.WriteLine("timer end...");
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }  
        }



        //gw - timer that fires to query vicon and broadcast to udp...
        public static void timer_elapsed_raw(object sender, System.Timers.ElapsedEventArgs args)
        {
            try
            {
                //  stats stuff...
                n_sends++;
                long diffticks = System.DateTime.Now.Ticks - startime.Ticks;
                float diffsecs = diffticks * 1.0f / System.TimeSpan.TicksPerSecond;
                if (diffsecs > 1.0f)
                {
                    fps = n_sends * 1.0f / diffsecs;
                    startime = System.DateTime.Now;
                    n_sends = 0;
                    String statsstr = String.Format("INFO: fps={0}...", fps.ToString());
                    Console.WriteLine(statsstr);
                }

                //  Get the frame number...
                
                MyClient.GetFrame();
                ViconDataStreamSDK.DotNET.Output_GetFrameNumber frame_no = MyClient.GetFrameNumber();

                String fno_str = String.Format("INFO: broadcasting data from vicon frame_no={0}...", frame_no.FrameNumber.ToString());
                Console.WriteLine(fno_str);

                //  Get the subject count...
                ViconDataStreamSDK.DotNET.Output_GetSubjectCount sc = MyClient.GetSubjectCount();

                packet_counter++;
                //  Start a simple packet...
                String packet = String.Format("ViconProxy,{0},Data,{1},{2},{3}\n", PROTOCOL_VERSION, packet_counter, frame_no.FrameNumber, sc.SubjectCount);
                //  Start an object for json...
                ViconData vdata = new ViconData();
                vdata.v = PROTOCOL_VERSION;
                vdata.fno = frame_no.FrameNumber;
                /*
                //  Iterate over subjects and complete the data packet...
                for (uint i = 0; i < sc.SubjectCount; i++)
                {
                    //  Get subject name...
                    ViconDataStreamSDK.DotNET.Output_GetSubjectName OutputGSN = MyClient.GetSubjectName(i);

                    //  Get subject count...
                    ViconDataStreamSDK.DotNET.Output_GetSegmentCount segc = MyClient.GetSegmentCount(OutputGSN.SubjectName);

                    //  Get coordinates of first segment, if any...
                    if (segc.SegmentCount == 1)
                    {
                        //  Get segment name of 0th segment...
                        ViconDataStreamSDK.DotNET.Output_GetSegmentName segname = MyClient.GetSegmentName(OutputGSN.SubjectName, 0);

                        //  Get translation...
                        ViconDataStreamSDK.DotNET.Output_GetSegmentGlobalTranslation dt = MyClient.GetSegmentGlobalTranslation(
                            OutputGSN.SubjectName, segname.SegmentName);

                        //  Get local translation...
                        ViconDataStreamSDK.DotNET.Output_GetSegmentLocalTranslation dlt = MyClient.GetSegmentLocalTranslation(
                            OutputGSN.SubjectName, segname.SegmentName);

                        //  Get rotation...
                        ViconDataStreamSDK.DotNET.Output_GetSegmentGlobalRotationEulerXYZ dr = MyClient.GetSegmentGlobalRotationEulerXYZ(
                            OutputGSN.SubjectName, segname.SegmentName);

                        //  append to simple packet...
                        //packet += String.Format("{0},{1},{2},{3},{4}\n", OutputGSN.SubjectName, segc.SegmentCount,
                        //    dt.Translation[0], dt.Translation[1], dt.Translation[2]);

                        //  append to object for json...
                        vdata.objno += 1;
                        ViconObject tdata = new ViconObject();
                        tdata.name = OutputGSN.SubjectName;
                        tdata.sc = segc.SegmentCount;
                        tdata.t[0] = (int)dt.Translation[0];
                        tdata.t[1] = (int)dt.Translation[1];
                        tdata.t[2] = (int)dt.Translation[2];
                        //tdata.lt[0] = dlt.Translation[0];
                        //tdata.lt[1] = dlt.Translation[1];
                        //tdata.lt[2] = dlt.Translation[2];
                        //tdata.r[0] = dr.Rotation[0];
                        //tdata.r[1] = dr.Rotation[1];
                        //tdata.r[2] = dr.Rotation[2];
                        tdata.oc = dt.Occluded;

                        vdata.objs.Add(tdata);

                        //  Get marker count...
                        ViconDataStreamSDK.DotNET.Output_GetMarkerCount mc = MyClient.GetMarkerCount(OutputGSN.SubjectName);
                        tdata.mc = mc.MarkerCount;

                        /*
                        //  Iterate over markers and compute avg...
                        if (mc.MarkerCount > 0)
                        {
                            uint avg_count = mc.MarkerCount;
                            for (uint m = 0; m < mc.MarkerCount; m++)
                            {
                                //  Get marker's name...
                                ViconDataStreamSDK.DotNET.Output_GetMarkerName OutputGMN = MyClient.GetMarkerName(OutputGSN.SubjectName, m);
                                //  Get its translation...
                                ViconDataStreamSDK.DotNET.Output_GetMarkerGlobalTranslation mtr =
                                    MyClient.GetMarkerGlobalTranslation(OutputGSN.SubjectName, OutputGMN.MarkerName);
                                if (mtr.Translation[0] < 0.00001 && mtr.Translation[1] < 0.00001 && mtr.Translation[2] < 0.00001)
                                {
                                    avg_count--;
                                }
                                else
                                {
                                    tdata.ct[0] += mtr.Translation[0];
                                    tdata.ct[1] += mtr.Translation[1];
                                    tdata.ct[2] += mtr.Translation[2];
                                }
                            }
                            tdata.ct[0] = tdata.ct[0] / avg_count;
                            tdata.ct[1] = tdata.ct[1] / avg_count;
                            tdata.ct[2] = tdata.ct[2] / avg_count;
                        }
                         * 

                        //  Iterate over subject markers and get info if its a special calibration object...
                        if ((mc.MarkerCount > 0) && (OutputGSN.SubjectName.ToLower().StartsWith(CALIB_PREFIX)))
                        {
                            tdata.mks = new System.Collections.ArrayList();
                            for (uint m = 0; m < mc.MarkerCount; m++)
                            {
                                //  Get marker's name...
                                ViconDataStreamSDK.DotNET.Output_GetMarkerName OutputGMN = MyClient.GetMarkerName(OutputGSN.SubjectName, m);
                                ViconMarker mdata = new ViconMarker();
                                mdata.name = OutputGMN.MarkerName;
                                //  Get its translation...
                                ViconDataStreamSDK.DotNET.Output_GetMarkerGlobalTranslation mtr =
                                    MyClient.GetMarkerGlobalTranslation(OutputGSN.SubjectName, OutputGMN.MarkerName);
                                mdata.t[0] = (int)mtr.Translation[0];
                                mdata.t[1] = (int)mtr.Translation[1];
                                mdata.t[2] = (int)mtr.Translation[2];
                                mdata.oc = mtr.Occluded;
                                tdata.mks.Add(mdata);
                            }
                        }
                    }
                    else
                    {
                        //  append to simple packet...
                        //packet += String.Format("{0},{1}\n", OutputGSN.SubjectName, segc.SegmentCount);

                        //  append to object for json...
                        vdata.objno += 1;
                        ViconObject tdata = new ViconObject();
                        tdata.name = OutputGSN.SubjectName;
                        tdata.sc = segc.SegmentCount;
                        vdata.objs.Add(tdata);
                    }
                }*/

                /*
                //  Broadcast the simple packet...
                System.Text.ASCIIEncoding encoding = new System.Text.ASCIIEncoding();
                byte[] b = encoding.GetBytes(packet);
                int sent = local_socket.Send(b, b.Length, simple_broadcast_ep);
                if (sent!=b.Length)
                {
                    Console.WriteLine("WARNING: packet send/sent size mismatch!");
                }
                 * */



                ViconData unlabeled = new ViconData();
                unlabeled.fno = frame_no.FrameNumber;
                uint ct = MyClient.GetUnlabeledMarkerCount().MarkerCount;
                int[] pt = { -1, -1, -1 };
   
                unlabeled.mode = 0;
                for (uint i = 0; i < ct; i++)
                {
                        //Inting off
                        double[] res = MyClient.GetUnlabeledMarkerGlobalTranslation(i).Translation;
                        pt[0] = (int)res[0];
                        pt[1] = (int)res[1];
                        pt[2] = (int)res[2];
                        unlabeled.objs.Add(pt);
                  
                    //double[] res = MyClient.GetUnlabeledMarkerGlobalTranslation(i).Translation;
                    //unlabeled.objs.Add(res);
                }

                Console.WriteLine(String.Format("INFO: Setting mode to: {0}", unlabeled.mode));
               
                String output = String.Format("INFO: Number unlabeled: {0}", ct);
                Console.WriteLine(output);


                //  Broadcast the json packet...
                System.Text.ASCIIEncoding encoding = new System.Text.ASCIIEncoding();
                JsonSerializer serializer = new JsonSerializer();
                StringWriter str = new StringWriter();
                JsonWriter writer = new JsonTextWriter(str);
                serializer.Serialize(writer, unlabeled);

                System.Text.ASCIIEncoding encoding2 = new System.Text.ASCIIEncoding();
                byte[] b2 = encoding.GetBytes(str.ToString());
                int sent = local_socket.Send(b2, b2.Length, json_broadcast_ep);
                if (sent != b2.Length)
                {
                    Console.WriteLine("WARNING: packet send/sent size mismatch!");
                }

                /*
                //  Check for direct tcp server connects...
                try
                {
                    //  Check for explicit connects...
                    Socket accepted_socket = local_listener.Accept();
                    accepted_socket.Blocking = false;
                    accepted_socket.NoDelay = true;
                    sockets.Add(accepted_socket, accepted_socket);
                }
                catch
                {
                    //  Got here, means nothing to accept (likely)...
                }

                //  Send to any tcp connections...
                foreach (Socket s in sockets.Keys)
                {
                    Console.WriteLine("INFO: Also broadcasting to...");
                    sent = s.Send(b);
                    if (sent != b.Length)
                    {
                        Console.WriteLine("WARNING: packet send/sent size mismatch!");
                    }
                }
                 */

                //Console.WriteLine("timer end...");
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
        }
	}
}