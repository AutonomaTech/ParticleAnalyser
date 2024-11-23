using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Text;
using System.Windows.Forms;

    public partial class AbsValSlider : UserControl , IControlBase, IControlSlider
    {
        public AbsValSlider( TIS.Imaging.VCDAbsoluteValueProperty itf )
        {
            _prop_itf = itf;

            InitializeComponent();

            Slider.Minimum = 0;
            Slider.Maximum = 100;

            UpdateControl();
        }

        // The interface this slider controls
        private TIS.Imaging.VCDAbsoluteValueProperty _prop_itf;

        // Flag to indicate that we are updating ourself, so that we have to ignore changes to the slider
        private bool _updating;

        // Flag to indicate that this control was changed by the user, so that we do not have to update it
        private bool _selfClicked;

        // Collection of sliders connected to interfaces of the same item
        // These sliders have to be updated when this slider is changed
        private System.Collections.ArrayList _sisterSliders;


        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            try
            {

                // Only change the property if the scroll event was caused by the user
                if (!_updating)
                {

                    // Get the new absolute value depending on the slider position
                    // and set the new value to the interface
                    if (!_prop_itf.ReadOnly)
                    {
                        _prop_itf.Value = GetAbsVal();
                    }

                    // Read the new value back from the absolute value interface
                    // This has to be done because we do not know the granularity of the absolute values
                    // and the value that has really been set normally differs from the value we
                    // assigned to the property
                    ScrollUpdate();

                    // If we know about possibly connected sliders, update them
                    _selfClicked = true;
                    if (!(_sisterSliders == null))
                    {
                        foreach (IControlSlider sld in _sisterSliders)
                        {
                            sld.ScrollUpdate();
                        }
                    }
                    _selfClicked = false;

                }

            }
            catch (Exception ex)
            {

                MessageBox.Show(ex.Message);

            }
        }

        private void AbsValSlider_Load(object sender, EventArgs e)
        {
            _updating = false;
            _selfClicked = false;
        }

        // This function calculates the needed position of the slider based on the current absolute value
        private int GetSliderPos()
        {
            double rmin = 0;
            double rmax = 0;
            double absval = 0;
            double rangelen = 0;
            double p = 0;

            // Get the property data from the interface
            rmin = _prop_itf.RangeMin;
            rmax = _prop_itf.RangeMax;
            absval = _prop_itf.Value;

            // Do calculation depending of the dimension function of the property
            if (_prop_itf.DimFunction == TIS.Imaging.AbsDimFunction.eAbsDimFunc_Log)
            {

                rangelen = System.Math.Log(rmax) - System.Math.Log(rmin);
                p = 100 / rangelen * (System.Math.Log(absval) - System.Math.Log(rmin));
            }
            else // AbsValItf.DimFunction = AbsDimFunction.eAbsDimFunc_Linear
            {
                rangelen = rmax - rmin;
                p = 100 / rangelen * (absval - rmin);
            }

            // Round to integer
            return (int)System.Math.Round(p, 0);
        }

        // This function calculates the current absolute value based on the position of the slider
        private double GetAbsVal()
        {

            double rmin = 0;
            double rmax = 0;
            double rangelen = 0;
            double value = 0;

            // Get the property data from the interface
            rmin = _prop_itf.RangeMin;
            rmax = _prop_itf.RangeMax;

            // Do calculation depending of the dimension function of the property
            if (_prop_itf.DimFunction == TIS.Imaging.AbsDimFunction.eAbsDimFunc_Log)
            {

                rangelen = System.Math.Log(rmax) - System.Math.Log(rmin);
                value = System.Math.Exp(System.Math.Log(rmin) + rangelen / 100 * Slider.Value);

            }
            else // AbsValItf.DimFunction = AbsDimFunction.eAbsDimFunc_Linear
            {

                rangelen = rmax - rmin;
                value = rmin + rangelen / 100 * Slider.Value;

            }

            // Correct the value if it is out of bounds
            if (value > rmax)
            {
                value = rmax;
            }
            if (value < rmin)
            {
                value = rmin;
            }

            return value;

        }

        public void UpdateControl()
        {
            _updating = true;

            // Check whether the property is available
            if (_prop_itf.Available)
            {

                // Enable the slider
                Slider.Enabled = true;

                ScrollUpdate();
            }
            else
            {
                // Disable the slider
                Slider.Enabled = false;

                // Disable the text box
                ValueText.Text = "";
                ValueText.Enabled = false;
            }

            _updating = false;
        }

        public void ScrollUpdate()
        {

            // Do not update if this event was caused by this control
            if (_selfClicked)
            {
                return;
            }

            // Do not update if the property is not avilable
            if (!_prop_itf.Available)
            {
                return;
            }

            _updating = true;

            // Assign a text representation of the current value to the text box
            ValueText.Text = _prop_itf.Value.ToString();
            if (!ValueText.Enabled)
            {
                ValueText.Enabled = true;
            }

            // Set the slider position
            Slider.Value = GetSliderPos();

            _updating = false;
        }


        public void setSisterSliders(System.Collections.ArrayList sliders)
        {
            _sisterSliders = sliders;
        }

        private void AbsValSlider_Layout(object sender, System.Windows.Forms.LayoutEventArgs e)
        {
            if (_prop_itf == null)
            {
                return;
            }

            // Determine the length of the describing text at some points to estimate
            // a good width for the edit box
            int lenmin = 0;
            int lenmid = 0;
            int lenmax = 0;

            System.Drawing.Graphics g = this.CreateGraphics();

            lenmin = (int)g.MeasureString(_prop_itf.RangeMin.ToString(), this.Font).Width;
            double valmid = _prop_itf.RangeMax - _prop_itf.RangeMin / 2;
            lenmid = (int)g.MeasureString(valmid.ToString(), this.Font).Width;
            lenmax = (int)g.MeasureString(_prop_itf.RangeMax.ToString(), this.Font).Width;

            g.Dispose();

            int textlen = 0;
            textlen = lenmin;
            if (lenmid > textlen)
            {
                textlen = lenmid;
            }
            if (lenmax > textlen)
            {
                textlen = lenmax;
            }

            // Resize the slider and the edit box
            Slider.Width = Width - (textlen + 20);
            Slider.Height = Height;
            ValueText.Height = Height;
            ValueText.Left = Width - (textlen + 20);
            ValueText.Width = textlen + 20;
        }
    }

