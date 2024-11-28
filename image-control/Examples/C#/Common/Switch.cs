using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Text;
using System.Windows.Forms;

    public partial class Switch : UserControl, IControlBase
    {
        public Switch( TIS.Imaging.VCDSwitchProperty itf )
        {
            InitializeComponent();

            _switchItf = itf;
            Check.Text = itf.Parent.Name;
            UpdateControl();
        }

        // The interface this switch controls
        public TIS.Imaging.VCDSwitchProperty _switchItf;

        public bool updating;

        // Collection of controls connected to interfaces of the same item
        // These controls have to be updated when this button is pushed
        public System.Collections.ArrayList sisterControls;


        private void checkBox1_CheckStateChanged(object sender, EventArgs e)
        {
            try
            {
                // Only change the property if the click event was caused by the user
                if (updating)
                {
                    return;
                }

                if (!_switchItf.ReadOnly)
                {
                    // Assign the new value to the property
                    _switchItf.Switch = Check.Checked;
                }

                // If we know about controls of the same item, update them
                if (!(sisterControls == null))
                {
                    foreach (IControlBase chk in sisterControls)
                    {
                        chk.UpdateControl();
                    }
                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void Switch_Resize(object sender, EventArgs e)
        {
            // The button fills the whole user control
            Check.Width = Width;
            Check.Height = Height;
        }

        public void UpdateControl()
        {

            updating = true;

            Check.Enabled = _switchItf.Available;
            Check.Checked = _switchItf.Switch;

            updating = false;
        }


        public void setSisterControls(System.Collections.ArrayList ctrls)
        {
            sisterControls = ctrls;
        }

    }

