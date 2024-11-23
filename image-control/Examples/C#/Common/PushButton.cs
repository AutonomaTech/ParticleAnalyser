using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Text;
using System.Windows.Forms;

    public partial class PushButton : UserControl, IControlBase
    {
        public PushButton( TIS.Imaging.VCDButtonProperty itf )
        {
            InitializeComponent();

            PushItf = itf;
            button1.Text = itf.Parent.Name;
            UpdateControl();
        }

        // The interface this button controls
        public TIS.Imaging.VCDButtonProperty PushItf;

        // Collection of controls connected to interfaces of the same item
        // These controls have to be updated when this button is pushed
        public System.Collections.ArrayList sisterControls;

        private void button1_Click(object sender, EventArgs e)
        {
            try
            {
                // Notify the interface
                PushItf.Push();

                // If we know about controls of the same item, update them
                if (!(sisterControls == null))
                {
                    foreach (IControlBase ctl in sisterControls)
                    {
                        ctl.UpdateControl();
                    }
                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void button1_Resize(object sender, EventArgs e)
        {
            // The button fills the whole user control
            button1.Width = Width;
            button1.Height = Height;
        }

        public void UpdateControl()
        {

            // Check whether the button property is available
            button1.Enabled = PushItf.Available;
        }


        public void setSisterControls(System.Collections.ArrayList controls)
        {
            sisterControls = controls;
        }



    }
