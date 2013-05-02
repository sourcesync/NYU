using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Data;
using System.Reflection;
using System.Windows.Forms;
using System.Net;
using System.Net.Sockets;
using System.Collections;
using Newtonsoft.Json;
using System.Runtime.Serialization;
using System.IO;
using System.Runtime.InteropServices;

namespace WallE_Proxy
{
    class Program
    {
        //  some constants...
        public const int PROTOCOL_VERSION = 1;
        public const ushort DEFAULT_SIMPLE_BROADCAST_SERVER_PORT = 6666;
        public const ushort DEFAULT_JSON_BROADCAST_SERVER_PORT = 6667;
        public const ushort DEFAULT_LISTENER_SERVER_PORT = 6868;
        public const String DEFAULT_VICON_ADDR = "localhost:801";
        public const String CALIB_PREFIX = "calib";
        public const int DEFAULT_TIMER_INTERVAL = 100;
        //  vicon objects...
        public static int packet_counter = 0;
        //  timer stuff...
        //public static System.Timers.ElapsedEventHandler timer_handler = nul
        //public static System.Timers.Timer timer;
        //public static int timer_time = DEFAULT_TIMER_INTERVAL;
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
        //  current key press...
        public static char keyp = '0';
        //  dump to file...
        public static String dumpPath = "c:\\temp\\walledump.txt";
        //  form
        public static WallE frm = null;

        public const float MULT_FACTOR = 10.0f;

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
            public uint sc = 0;
            public uint mc = 0;
            public System.Collections.ArrayList mks = null;
            public bool oc = false;
        }
        public class ViconData
        {
            public uint v = Program.PROTOCOL_VERSION;
            public uint fno = 0;
            public uint objno = 0;
            public uint mode = 0;
            public System.Collections.ArrayList objs = new System.Collections.ArrayList();
        }

        [DllImport("blobslib.dll")]
        static extern int blobslib_init(int cam);

        [DllImport("blobslib.dll")]
        static extern int blobslib_do(IntPtr pblobs, IntPtr num_blobs, char keyp);

        [DllImport("blobslib.dll")]
        //static extern int blobslib_doall(IntPtr pblobs, IntPtr num_blobs, char keyp,
        //    int rmin, int rmax, int gmin, int gmax, int bmin, int bmax, int gain, int exposure);
        static extern int blobslib_doall(IntPtr pblobs, IntPtr num_blobs, IntPtr pblobparams);

        [DllImport("User32.dll")]
        public static extern int FindWindow(string strClassName, string strWindowName);

        [DllImport("blobslib.dll")]
        static extern int fnblobslib();

        const int MAX_BLOBS = 1000;

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, Pack = 0)]
        public struct blobtype
        {
            public float x;
            public float y;
            public float z;
        };

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, Pack=0)]
        public struct blobparamstype
        {
            public int rmin;
            public int rmax;
            public int gmin;
            public int gmax;
            public int bmin;
            public int bmax;
            public int gain;
            public int exposure;
            public int erode;
            public int dilate;
            public int drawmode;
            public int sig_setmask;
            public int sig_resetmask;
            public int flipmode;
            public int eshape;
            public int erows;
            public int ecols;
            public int dshape;
            public int drows;
            public int dcols;
            public int sig_savemask;
            public int sig_loadmask;
            public char keyp;
        };

        public static int camid;
        public static blobtype bt = new blobtype();
        public static IntPtr pblobs = System.Runtime.InteropServices.Marshal.AllocHGlobal(System.Runtime.InteropServices.Marshal.SizeOf(bt) * MAX_BLOBS);
        public static IntPtr num_blobs = System.Runtime.InteropServices.Marshal.AllocHGlobal(sizeof(int));
        public static blobparamstype blobparams = new blobparamstype();
        public static IntPtr pblobparams = System.Runtime.InteropServices.Marshal.AllocHGlobal(System.Runtime.InteropServices.Marshal.SizeOf(blobparams));

        public static void Quit(String msg, int val)
        {
            Console.WriteLine(msg);
            System.Threading.Thread.Sleep(5000);
            System.Environment.Exit(val);
        }

        //gw - timer that fires to query vicon and broadcast to udp...
        public static void timer_elapsed_raw(object sender, System.Timers.ElapsedEventArgs args)
        {
            //timer.Stop();
            //do_it(keyp);
            //timer.Start();
        }


        //gw - timer that fires to query vicon and broadcast to udp...
        public static int do_it(char keyp)
        {
            try
            {
                blobparamstype blp =
                    (blobparamstype)System.Runtime.InteropServices.Marshal.PtrToStructure(pblobparams, typeof(blobparamstype));

                //  Get blob data from blob tracker...
                //int retv = blobslib_do(pblobs, num_blobs, keyp);
                //int retv = blobslib_doall(pblobs, num_blobs, keyp, rmin, rmax, gmin, gmax, bmin, gmax, gain, exposure);
                int retv = blobslib_doall(pblobs, num_blobs, pblobparams);
                if (retv <= 0)
                {
                    Quit("WARNING: Some error or termination request at blobslib_do...", 1);
                    return 0;
                }
                

                //  Do some stats...
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
                uint fno = 0;
                String fno_str = String.Format("INFO: broadcasting data from vicon frame_no={0}...", fno);
                //Console.WriteLine(fno_str);

                //  Get number of blobs...
                uint nblobs = (uint)System.Runtime.InteropServices.Marshal.ReadInt32(num_blobs);
                //System.Console.WriteLine(String.Format("INFO: num_blobs={0}", nblobs));

                //  Populate vicon data structure...
                ViconData unlabeled = new ViconData();
                unlabeled.fno = fno;
                uint ct = nblobs;
                unlabeled.mode = 0;
                for (uint i = 0; i < ct; i++)
                {
                    IntPtr arr = new IntPtr(pblobs.ToInt64() + i * System.Runtime.InteropServices.Marshal.SizeOf(bt));
                    blobtype rblob = (blobtype)System.Runtime.InteropServices.Marshal.PtrToStructure(arr, typeof(blobtype));
                    //System.Console.WriteLine(String.Format("INFO: blob x={0},y={1},z={2}", rblob.x, rblob.y, rblob.z));

                    double[] res = new double[] { 
                        rblob.x * MULT_FACTOR, 
                        rblob.y * MULT_FACTOR, 
                        rblob.z * MULT_FACTOR }; // MyClient.GetUnlabeledMarkerGlobalTranslation(i).Translation;
                    int[] pt = new int[] { -1, -1, -1 };
                    pt[0] = (int)res[0];
                    pt[1] = (int)res[1];
                    pt[2] = (int)res[2];
                    unlabeled.objs.Add( pt );
                }

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

                //  Possibly dump to file path...
                if (dumpPath != String.Empty)
                {
                    StreamWriter sw;
                    sw = System.IO.File.AppendText(dumpPath);
                    sw.WriteLine(str);
                    sw.Close();
                }
                return 1;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
                return -1;
            }
        }

        static public void UpdateThresholds()
        {
            blobparamstype blp = (blobparamstype)System.Runtime.InteropServices.Marshal.PtrToStructure(pblobparams, typeof(blobparamstype));
            blp.rmin = (int)frm.rmin.Value;
            blp.rmax = (int)frm.rmax.Value;
            blp.gmin = (int)frm.gmin.Value;
            blp.gmax = (int)frm.gmax.Value;
            blp.bmin = (int)frm.bmin.Value;
            blp.bmax = (int)frm.bmax.Value;
            blp.gain = (int)frm.gain.Value;
            blp.exposure = (int)frm.exposure.Value;
            blp.erode = (int)frm.erode.Value;
            blp.dilate = (int)frm.dilate.Value;
            blp.eshape = 0;
            blp.erows = (int)frm.erows.Value;
            blp.ecols = (int)frm.ecols.Value; 
            blp.dshape = 0;
            blp.drows = (int)frm.drows.Value;
            blp.dcols = (int)frm.dcols.Value;

            System.Runtime.InteropServices.Marshal.StructureToPtr(blp, pblobparams, false);
        }

        static void ShowForm()
        {
            if (frm != null)
            {
                frm.Close();
                frm = null;
            }
            frm = new WallE();

            blobparamstype blp = (blobparamstype)System.Runtime.InteropServices.Marshal.PtrToStructure(pblobparams, typeof(blobparamstype));
            frm.rmin.Value = blp.rmin;
            frm.rmax.Value = blp.rmax;
            frm.gmin.Value = blp.gmin;
            frm.gmax.Value = blp.gmax;
            frm.bmin.Value = blp.bmin;
            frm.bmax.Value = blp.bmax;
            frm.gain.Value = blp.gain;
            frm.exposure.Value = blp.exposure;
            frm.erode.Value = blp.erode;
            frm.dilate.Value = blp.dilate;
            frm.erows.Value = blp.erows;
            frm.ecols.Value = blp.ecols;
            frm.drows.Value = blp.drows;
            frm.dcols.Value = blp.dcols;

            frm.Update();
            frm.Show();
        }

        static void Usage()
        {
            Quit("ERROR: Usage:  WallEProxy.exe <broadcast ip:port> <cam id> [dump path]", 1);
        }

        static char ProcessKeyBoard()
        {
            char k = '0';
            if (System.Console.KeyAvailable)
            {
                System.ConsoleKeyInfo ck = System.Console.ReadKey();
                
                k = ck.KeyChar;

                if (k == 'f')
                {
                    ShowForm();
                }
            }
            return k;
        }

        static void Main(string[] args)
        {
            //  Parse command line args...
            if (System.Environment.GetCommandLineArgs().Length == 4)
            {
                dumpPath = System.Environment.GetCommandLineArgs()[3];
            }
            else if (System.Environment.GetCommandLineArgs().Length != 3)
            {
                Usage();
            }

            //  Get cam id...
            camid = int.Parse(System.Environment.GetCommandLineArgs()[2]);

            
            //  blob params...
            blobparamstype parms = 
                (blobparamstype)System.Runtime.InteropServices.Marshal.PtrToStructure(pblobparams, typeof(blobparamstype));
            parms.rmin = 100;
            parms.rmax = 256;
            parms.gmin = 100;
            parms.gmax = 256;
            parms.bmin = 100;
            parms.bmax = 256;
            parms.gain = 0;
            parms.exposure = 511;
            parms.erode = 0;
            parms.dilate = 15;
            parms.drawmode = 0;
            parms.sig_setmask = 0;
            parms.sig_resetmask = 0;
            parms.flipmode = 0;
            parms.eshape = 0;
            parms.erows = 3;
            parms.ecols = 3;
            parms.dshape = 0;
            parms.drows = 3;
            parms.dcols = 3;
            parms.keyp = (char)0;
            parms.sig_loadmask = 0;
            parms.sig_savemask = 0;
            System.Runtime.InteropServices.Marshal.StructureToPtr(parms, pblobparams, false);
            
            //  Initialize blob tracker and cam...
            if (blobslib_init(0) <= 0)
            {
                Quit("ERROR: Cannot init blobs tracker", 1);
                return;
            }

            //  Init any file dump...
            if (dumpPath != String.Empty)
            {
                StreamWriter sw;
                sw = System.IO.File.CreateText(dumpPath);
                sw.Close();
            }

            //  Parse broadcast address...
            String broadcast_addr = System.Environment.GetCommandLineArgs()[1];
            int broadcast_port = DEFAULT_JSON_BROADCAST_SERVER_PORT;
            //  If has colon, sep into addr and port...
            String[] parts = broadcast_addr.Split(new char[] { ':' });
            if (parts.Length > 1)
            {
                broadcast_addr = parts[0];
                broadcast_port = int.Parse(parts[1]);
                Console.WriteLine(String.Format("INFO: overriding default json broadcast port -> {0}...", broadcast_port));
                System.Threading.Thread.Sleep(1500);
            }

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
                Quit("ERROR: Something went wrong with setting up sockets - is it a valid ip address and port?", 1);
                return;
            }

            while (true)
            {
                char k = ProcessKeyBoard();

                int retv = do_it(k);
                if (retv <= 0)
                {
                    break;
                }
            }

            Console.WriteLine("INFO: Done.");

        }
    }
}
