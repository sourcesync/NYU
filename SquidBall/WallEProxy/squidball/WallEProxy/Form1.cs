using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace WallE_Proxy
{
    public partial class WallE : Form
    {
        public WallE()
        {
            InitializeComponent();
        }

        private void WallE_Load(object sender, EventArgs e)
        {
            this.rmin.ValueChanged += new System.EventHandler(ValueChanged);
            this.rmax.ValueChanged += new System.EventHandler(ValueChanged);
            this.gmin.ValueChanged += new System.EventHandler(ValueChanged);
            this.gmax.ValueChanged += new System.EventHandler(ValueChanged);
            this.bmin.ValueChanged += new System.EventHandler(ValueChanged);
            this.bmax.ValueChanged += new System.EventHandler(ValueChanged);
            this.gain.ValueChanged += new System.EventHandler(ValueChanged);
            this.exposure.ValueChanged += new System.EventHandler(ValueChanged);
            this.erode.ValueChanged += new System.EventHandler(ValueChanged);
            this.dilate.ValueChanged += new System.EventHandler(ValueChanged);
            this.erows.ValueChanged += new System.EventHandler(ValueChanged);
            this.ecols.ValueChanged += new System.EventHandler(ValueChanged);
            this.drows.ValueChanged += new System.EventHandler(ValueChanged);
            this.dcols.ValueChanged += new System.EventHandler(ValueChanged);
        }

        private void ValueChanged(object sender, EventArgs e)
        {
            Program.UpdateThresholds();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            this.Close();
        }

    }
}
