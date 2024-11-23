using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Text;
using System.Windows.Forms;


    public partial class StringCombo : UserControl, IControlBase
    {
        public StringCombo( TIS.Imaging.VCDMapStringsProperty itf )
        {
            InitializeComponent();

            _prop_itf = itf;

            InitialUpdate();
            UpdateControl();
        }

        // The interface this combo controls
        private TIS.Imaging.VCDMapStringsProperty _prop_itf;

        // Flag to indicate that we are updating ourself, so that we have to ignore changes to the slider
        private bool _updating;

        // Collection of sliders connected to interfaces of the same item
        // These sliders have to be updated when this combo is changed
        private System.Collections.ArrayList _sisterSliders;


        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {

                // Only change the property if the click event was caused by the user
                if (!_updating)
                {

                    // Assign the new string
                    if (!_prop_itf.ReadOnly)
                    {
                        _prop_itf.String = Combo.Text;
                    }

                    // If we know about possibly connected sliders, update them
                    if (!(_sisterSliders == null))
                    {
                        foreach (IControlBase ctl in _sisterSliders)
                        {
                            ctl.UpdateControl();
                        }
                    }

                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        public void UpdateControl()
        {
            _updating = true;

            // Check whether the property is available
            Combo.Enabled = _prop_itf.Available;

            // Select the current string
            ScrollUpdate();

            _updating = false;
        }

        private void InitialUpdate()
        {
            // Fill the combo box with the available strings
            Combo.Items.Clear();
            foreach (string s in _prop_itf.Strings)
            {
                Combo.Items.Add(s);
            }

        }

        public void ScrollUpdate()
        {

            _updating = true;

            // Calculate the new position
            Combo.SelectedIndex = _prop_itf.Value - _prop_itf.RangeMin;

            _updating = false;

        }

        private void StringCombo_Resize(object eventSender, System.EventArgs eventArgs)
        {
            Combo.Width = Width;
        }

        public void setSisterSliders(System.Collections.ArrayList sliders)
        {
            _sisterSliders = sliders;
        }
    }
