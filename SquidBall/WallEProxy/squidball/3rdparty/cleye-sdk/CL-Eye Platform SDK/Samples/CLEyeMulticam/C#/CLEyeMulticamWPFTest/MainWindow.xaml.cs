//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// This file is part of CL-EyeMulticam SDK
//
// WPF C# CLEyeMulticamWPFTest Sample Application
//
// It allows the use of multiple CL-Eye cameras in your own applications
//
// For updates and file downloads go to: http://codelaboratories.com
//
// Copyright 2008-2010 (c) Code Laboratories, Inc. All rights reserved.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Interop;
using CLEyeMulticam;

namespace CLEyeMulticamWPFTest
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        int numCameras = 0;
        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += new RoutedEventHandler(MainWindow_Loaded);
            this.Closing += new System.ComponentModel.CancelEventHandler(MainWindow_Closing);
        }

        void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (numCameras >= 1)
            {
                cameraImage1.Device.Stop();
                cameraImage1.Device.Destroy();
            }
            if (numCameras == 2)
            {
                cameraImage2.Device.Stop();
                cameraImage2.Device.Destroy();
            }
        }

        void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            // Query for number of connected cameras
            numCameras = CLEyeCameraDevice.CameraCount;
            if(numCameras == 0)
            {
                MessageBox.Show("Could not find any PS3Eye cameras!");
                return;
            }
            output.Items.Add(string.Format("Found {0} CLEyeCamera devices", numCameras));
            // Show camera's UUIDs
            for (int i = 0; i < numCameras; i++)
            {
                output.Items.Add(string.Format("CLEyeCamera #{0} UUID: {1}", i + 1, CLEyeCameraDevice.CameraUUID(i)));
            }
            // Create cameras, set some parameters and start capture
            if (numCameras >= 1)
            {
                cameraImage1.Device.Create(CLEyeCameraDevice.CameraUUID(0));
                cameraImage1.Device.Zoom = -50;
                cameraImage1.Device.Start();
            }
            if (numCameras == 2)
            {
                cameraImage2.Device.Create(CLEyeCameraDevice.CameraUUID(1));
                cameraImage2.Device.Rotation = 200;
                cameraImage2.Device.Start();
            }
        }
    }
}
