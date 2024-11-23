using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Text;
using System.Windows.Forms;


    public partial class RangeSlider : UserControl, IControlBase, IControlSlider
    {
        public RangeSlider( TIS.Imaging.VCDRangeProperty itf )
        {
            InitializeComponent();
            _rangeProperty = itf;
            InitialUpdate();
            UpdateControl();

        }

        // The interface this slider controls
        private TIS.Imaging.VCDRangeProperty _rangeProperty;

        // Collection of sliders connected to interfaces of the same item
        // These sliders have to be updated when this slider is changed
        private System.Collections.ArrayList _sisterSliders;

        private void Slider_Scroll(object sender, EventArgs e)
        {
            try
            {
                // Assign the new value to the property
                if (!_rangeProperty.ReadOnly)
                {
                    _rangeProperty.Value = Slider.Value * _rangeProperty.Delta;
                }

                // Update the text box
                ValueText.Text = _rangeProperty.Value.ToString();

                // If we know about possibly connected sliders, update them
                if (!(_sisterSliders == null))
                {
                    foreach (IControlSlider sld in _sisterSliders)
                    {
                        if (!(sld == this))
                        {
                            sld.ScrollUpdate();
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

            // Check whether the property is available
            if (_rangeProperty.Available)
            {

                // Enable slider and textbox
                Slider.Enabled = true;
                ValueText.Enabled = true;

                // Update slider position
                ScrollUpdate();
            }
            else
            {
                // Disable slider
                Slider.Enabled = false;

                // Disable text
                ValueText.Text = "";
                ValueText.Enabled = false;
            }

        }

        private void InitialUpdate()
        {
            // Initialize the slider range
            int min = _rangeProperty.RangeMin / _rangeProperty.Delta;
            int max = _rangeProperty.RangeMax / _rangeProperty.Delta;

            Slider.TickFrequency = 1;
            if (max - min > 50)
            {
                Slider.TickFrequency = 10;
            }
            if (max - min > 500)
            {
                Slider.TickFrequency = 100;
            }

            Slider.Minimum = min;
            Slider.Maximum = max;

            if (min == max)
            {
                Slider.Enabled = false;
            }

        }

        public void ScrollUpdate()
        {
            if (!_rangeProperty.Available) return;

            // Calculate the new slider position
            int pos = _rangeProperty.Value / _rangeProperty.Delta;

            if (pos < Slider.Minimum || pos > Slider.Maximum )
            {
                Slider.Enabled = false;
            }
            else
            {
                Slider.Value = pos;
            }

            ValueText.Text = pos.ToString();
        }
        public void setSisterSliders(System.Collections.ArrayList sliders)
        {
            _sisterSliders = sliders;
        }

        private void RangeSlider_Layout(object sender, System.Windows.Forms.LayoutEventArgs e)
        {
            // Adjust the slider and textbox position
            Slider.Width = Width * 80 / 100;
            Slider.Height = Height;
            ValueText.Height = Height;
            ValueText.Left = Width * 80 / 100;
            ValueText.Width = Width * 20 / 100;
        }

    }

