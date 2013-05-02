using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace blobslibtest_cs
{
    class Program
    {
        [DllImport("blobslib.dll")]
        static extern int blobslib_init(int cam);

        [DllImport("blobslib.dll")]
        static extern int blobslib_do(IntPtr pblobs, IntPtr num_blobs, char keyp);

        [DllImport("User32.dll")]
        public static extern int FindWindow(string strClassName, string strWindowName);

        [DllImport("blobslib.dll")]
        static extern int fnblobslib();

        const int MAX_BLOBS = 1000;

        public struct blobtype
        {
            public float x;
            public float y;
            public float z;
        };

        static void Main(string[] args)
        {
            string curdir = System.Environment.CurrentDirectory;
            System.Console.WriteLine("before init");

            FindWindow("yo", "yo");

            fnblobslib();

            int retv = blobslib_init(0);
            if (retv <= 0)
            {
                System.Console.WriteLine("blobslib_init failed.");
            }

            blobtype bt;
            bt.x = 0.0f;
            bt.y = 0.0f;
            bt.z = 0.0f;
            IntPtr pblobs = System.Runtime.InteropServices.Marshal.AllocHGlobal( System.Runtime.InteropServices.Marshal.SizeOf(bt) * MAX_BLOBS);
            IntPtr num_blobs = System.Runtime.InteropServices.Marshal.AllocHGlobal(sizeof(int));
            while (true)
            {
                //IntPtr pblobs = IntPtr.Zero;
                //IntPtr num_blobs = IntPtr.Zero;
                char keyp = '0';
                if (System.Console.KeyAvailable)
                {
                    System.ConsoleKeyInfo info = System.Console.ReadKey();
                    keyp = info.KeyChar;
                }
                retv = blobslib_do( pblobs, num_blobs, keyp );
                if (retv <= 0)
                {
                    break;
                }
                int nblobs = System.Runtime.InteropServices.Marshal.ReadInt32(num_blobs);
                System.Console.WriteLine(String.Format("INFO: num_blobs={0}", nblobs));
                for (int i=0; i<nblobs; i++)
                {
                    //IntPtr arr = System.Runtime.InteropServices.Marshal.ReadIntPtr( pblobs, i* System.Runtime.InteropServices.Marshal.SizeOf( bt ) );
                    IntPtr arr = new IntPtr(pblobs.ToInt64() + i*System.Runtime.InteropServices.Marshal.SizeOf(bt));
                    blobtype rblob = (blobtype)System.Runtime.InteropServices.Marshal.PtrToStructure(arr, typeof(blobtype));
                    System.Console.WriteLine(String.Format("INFO: blob x={0},y={1},z={2}", rblob.x, rblob.y, rblob.z));
                }
            }
        }
    }
}
