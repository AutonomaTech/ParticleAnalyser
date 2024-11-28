using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using TIS.Imaging;

namespace VCD_Property_Page
{
    public partial class VCDPropertiesDlg : Form
    {
        private TIS.Imaging.ICImagingControl icImagingControl;

        // The frame of the category that is currently filled with controls
        private System.Windows.Forms.GroupBox currentFrame;

        // The controls of an item are put into a collection to allow fast updates after checking switches
        private System.Collections.ArrayList currentSisterControls;

        // The Sliders of an item are put into a separate collection to allow even faster updates of
        // connected sliders
        private System.Collections.ArrayList currentSisterSliders;

        // This is a collection of all controls for the global update event
        private readonly System.Collections.ArrayList allControls = new System.Collections.ArrayList();

        // These constants control the layout of the property page
        private const int CTRL_HEIGHT = 27;
        private const int ROW_HEIGHT = 28;
        private const int PAGE_WIDTH = 500;

        private const int LABEL_WIDTH = 100;
        private const int SLIDER_WIDTH = 180;

        private const int COMBO_EXTRA_WIDTH = 33;
        private const int BUTTON_EXTRA_WIDTH = 20;
        private const int CHECK_EXTRA_WIDTH = 27;

        private const int CTRL_MARGIN = 3;

        public VCDPropertiesDlg( TIS.Imaging.ICImagingControl ic )
        {
            icImagingControl = ic;
            InitializeComponent();
        }

        private void VCDPropertiesDlg_Load(object sender, EventArgs e)
        {
            int height = 7;

            // Iterate through the categories
            foreach (string Cat in icImagingControl.VCDPropertyItems.CategoryMap.Categories)
            {
                int subHeight;
                // Insert a frame for the category and controls for its items
                InsertCategory(Cat, 7, height, PAGE_WIDTH - 14, out subHeight);
                height = height + subHeight;
            }

            // Resize the page and adjust the button's positions
            this.Width = PAGE_WIDTH;
            this.Height = height + OKButton.Height + 14 + 14 + 7;
            OKButton.Top = height;
            OKButton.Left = this.Width - 14 - OKButton.Width;
            UpdateButton.Top = height;
            UpdateButton.Left = OKButton.Left - 7 - UpdateButton.Width;
        }

        private void UpdateControls()
        {
            // Update all controls
            icImagingControl.VCDPropertyItems.Update();
            foreach (IControlBase ctl in allControls)
            {
                ctl.UpdateControl();
            }
        }

        private void UpdateButton_Click(object sender, EventArgs e)
        {
            UpdateButton.Enabled = false;
            UpdateControls();
            UpdateButton.Enabled = true;
        }

        private void OKButton_Click(object sender, EventArgs e)
        {
            this.Close();
        }     

        private void CreateCategoryFrame(string name, int left, int top, int width)
        {

            // Create a frame object
            System.Windows.Forms.GroupBox frm = new System.Windows.Forms.GroupBox();
            frm.Name = "Group" + allControls.Count;
            this.Controls.Add(frm);

            frm.Text = Name;
            frm.SetBounds(left, top, Width, 0);

            // Set the global frame reference to this new frame
            currentFrame = frm;
        }

        private void CreatePropertyLabel(string name, int left, int top, int width, int height)
        {

            // Create a label object
            System.Windows.Forms.Label lbl = new System.Windows.Forms.Label();
            lbl.Name = "Label" + allControls.Count;
            currentFrame.Controls.Add(lbl);

            lbl.SetBounds(left, top, width, height);
            lbl.Text = name;
        }

        private void CreateRangeSlider(TIS.Imaging.VCDRangeProperty itf, int left, int top, int width, int height)
        {
            // Create a RangeSlider control
            RangeSlider Slider = new RangeSlider( itf );
            Slider.Name = "slider" + allControls.Count;
            currentFrame.Controls.Add(Slider);

            Slider.SetBounds(left, top, width, height);

            // Add the control to the current sistercontrols collection
            currentSisterControls.Add(Slider);
            // Add the control to the current sistersliders collection
            currentSisterSliders.Add(Slider);

            // Tell the control the sistersliders group it belongs to
            Slider.setSisterSliders(currentSisterSliders);

            // Add the control to the global list of all controls
            allControls.Add(Slider);
        }

        private void CreateAbsValSlider(TIS.Imaging.VCDAbsoluteValueProperty itf, int left, int top, int width, int height)
        {

            // Create a AbsValSlider control
            AbsValSlider Slider = new AbsValSlider( itf );
            Slider.Name = "absvalslider" + allControls.Count;
            currentFrame.Controls.Add(Slider);

            Slider.SetBounds(left, top, width, height);

            // Add the control to the current sistercontrols collection
            currentSisterControls.Add(Slider);
            // Add the control to the current sistersliders collection
            currentSisterSliders.Add(Slider);

            // Tell the control the sistersliders group it belongs to
            Slider.setSisterSliders(currentSisterSliders);

            // Add the control to the global list of all controls
            allControls.Add(Slider);
        }

        private void CreateSwitch(TIS.Imaging.VCDSwitchProperty itf, int left, int top, int width, int height)
        {
            // Create a Switch Control
            Switch swtch = new Switch( itf );
            swtch.Name = "switch" + allControls.Count;
            currentFrame.Controls.Add(swtch);

            swtch.SetBounds(left, top, width, height);

            // Add the control to the current sistercontrols collection
            currentSisterControls.Add(swtch);

            // Tell the control the sister controls group it belongs to
            swtch.setSisterControls(currentSisterControls);

            // Add the control to the global list of all controls
            allControls.Add(swtch);
        }

        private void CreateButton(TIS.Imaging.VCDButtonProperty itf, int left, int top, int width, int height)
        {
            // Create a PushButton control
            PushButton btn = new PushButton( itf );
            btn.Name = "button" + allControls.Count;
            currentFrame.Controls.Add(btn);

            btn.SetBounds(left, top, width, height);

            // Add the control to the current sistercontrols collection
            currentSisterControls.Add(btn);

            // Tell the control the sister controls group it belongs to
            btn.setSisterControls(currentSisterControls);

            // Add the control to the global list of all controls
            allControls.Add(btn);
        }

        private void CreateCombo(TIS.Imaging.VCDMapStringsProperty itf, int left, int top, int width, int height)
        {
            // Create a StringCombo control
            StringCombo cbo = new StringCombo( itf );
            cbo.Name = "combo" + allControls.Count;
            currentFrame.Controls.Add(cbo);

            cbo.SetBounds(left, top, width, height);

                        // Add the control to the current sistercontrols collection
            currentSisterControls.Add(cbo);
            // Add the control to the current sistersliders collection
            currentSisterSliders.Add(cbo);

            // Tell the control the sistersliders group it belongs to
            cbo.setSisterSliders(currentSisterSliders);

            // Add the control to the global list of all controls
            allControls.Add(cbo);
        }

        private void InsertElement(TIS.Imaging.VCDPropertyElement elem, ref int x, ref int y, string txt)
        {
            // Try to get all known interfaces. Some may be 'null'
            VCDAbsoluteValueProperty AbsValItf = elem.Find<VCDAbsoluteValueProperty>();
            VCDMapStringsProperty MapStringsItf = elem.Find<VCDMapStringsProperty>();
            VCDRangeProperty RangeItf = elem.Find<VCDRangeProperty>();
            VCDSwitchProperty SwitchItf = elem.Find<VCDSwitchProperty>();
            VCDButtonProperty buttonItf = elem.Find<VCDButtonProperty>();

            // A label is only needed if the interface is AbsValItf, MapStringsItf or RangeItf,
            // because they are represented by sliders and sliders don't show the name of a property
            int labelWidth = 0;
            if ((AbsValItf != null || MapStringsItf != null || RangeItf != null) && (txt != ""))
            {
                labelWidth = TextWidth(txt);
                if (labelWidth < LABEL_WIDTH)
                {
                    labelWidth = LABEL_WIDTH;
                }

                CreatePropertyLabel(txt, x, y + CTRL_HEIGHT / 4, labelWidth, CTRL_HEIGHT / 4 * 3);
                x = x + labelWidth;
            }

            int w = 0;

            // If we were able to acquire an Absolute Value Interface, create an appropriate slider
            if (AbsValItf != null)
            {

                CreateAbsValSlider(AbsValItf, x, y, SLIDER_WIDTH, CTRL_HEIGHT);
                x = x + SLIDER_WIDTH + CTRL_MARGIN;

                // If we were able to acquire an MapStrings Interface, create a combo box
            }
            else if (MapStringsItf != null)
            {

                // Calculate the needed with of the combo box
                // based on the width of the entries
                int boxWidth = 0;
                foreach (string s in MapStringsItf.Strings)
                {
                    w = TextWidth(s);
                    if (w > boxWidth)
                    {
                        boxWidth = w;
                    }
                }

                // Add some more space for the dropdown button
                boxWidth = boxWidth + COMBO_EXTRA_WIDTH;

                CreateCombo(MapStringsItf, x, y, boxWidth, CTRL_HEIGHT);
                x = x + boxWidth + CTRL_MARGIN;

                // If we were able to acquire a range interface, create a range slider
            }
            else if (RangeItf != null)
            {

                CreateRangeSlider(RangeItf, x, y, SLIDER_WIDTH, CTRL_HEIGHT);
                x = x + SLIDER_WIDTH + CTRL_MARGIN;

                // If we acquired a switch interface, create a switch checkbox
            }
            else if (SwitchItf != null)
            {

                // Determine the needed width of the switch control
                int switchWidth = TextWidth(elem.Name);
                switchWidth = switchWidth + CHECK_EXTRA_WIDTH;

                CreateSwitch(SwitchItf, x, y, switchWidth, CTRL_HEIGHT);
                x = x + switchWidth + CTRL_MARGIN;

                // If we got a button interface, create a button control
            }
            else if (buttonItf != null)
            {

                // Determine the needed width of the button control
                int buttonWidth = TextWidth(elem.Name);
                buttonWidth = buttonWidth + BUTTON_EXTRA_WIDTH;

                CreateButton(buttonItf, x, y, buttonWidth, CTRL_HEIGHT);
                x = x + buttonWidth + CTRL_MARGIN;
            }

        }

        private void InsertItem(TIS.Imaging.VCDPropertyItem item, ref int y)
        {
            int x = 20;

            // Try to find the 'default' elements (Value, Auto and One Push)
            VCDPropertyElement valueElem = item.Elements.FindElement(TIS.Imaging.VCDGUIDs.VCDElement_Value);
            VCDPropertyElement autoElem = item.Elements.FindElement(TIS.Imaging.VCDGUIDs.VCDElement_Auto);
            VCDPropertyElement onepushElem = item.Elements.FindElement(TIS.Imaging.VCDGUIDs.VCDElement_OnePush);

            if (item.ItemGUID == TIS.Imaging.VCDGUIDs.VCDID_WhiteBalance)
            {
                if (item.Elements.FindElement(TIS.Imaging.VCDGUIDs.VCDElement_WhiteBalanceRed) != null)
                {
                    valueElem = null;
                }
            }

            // Create a label with appropriate size
            int labelWidth = TextWidth(item.Name);
            if (labelWidth < LABEL_WIDTH)
            {
                labelWidth = LABEL_WIDTH;
            }

            CreatePropertyLabel(item.Name, x, y + CTRL_HEIGHT / 4, labelWidth, CTRL_HEIGHT / 4 * 3);
            x = x + labelWidth + CTRL_MARGIN;

            // If we found 'default' elements, create controls for them
            if (valueElem != null)
            {
                InsertElement(valueElem, ref x, ref y, "");
            }
            if (autoElem != null)
            {
                InsertElement(autoElem, ref x, ref y, "");
            }
            if (onepushElem != null)
            {
                InsertElement(onepushElem, ref x, ref y, "");
            }

            y = y + ROW_HEIGHT;

            // Find any non-default elements we did not create controls for yet
            foreach (TIS.Imaging.VCDPropertyElement elem in item.Elements)
            {
                Guid guid = elem.ElementGUID;

                if (guid != TIS.Imaging.VCDGUIDs.VCDElement_Value & guid != TIS.Imaging.VCDGUIDs.VCDElement_Auto & guid != VCDGUIDs.VCDElement_OnePush)
                {

                    x = 25;
                    InsertElement(elem, ref x, ref y, elem.Name);
                    y = y + ROW_HEIGHT;

                }
            }
        }

        private void InsertCategory(string category, int left, int top, int right, out int height)
        {
            int y = 20;

            // Create a frame for the category
            CreateCategoryFrame(category, left, top, right - left);

            // Create a new control group for this category
            currentSisterControls = new System.Collections.ArrayList();

            // Iterate through the items in this category
            foreach (Guid itemID in icImagingControl.VCDPropertyItems.CategoryMap.ItemIDs[category])
            {
                TIS.Imaging.VCDPropertyItem item = icImagingControl.VCDPropertyItems.FindItem(itemID);

                // If we got a valid item, insert controls for it
                if (item != null)
                {
                    currentSisterSliders = new System.Collections.ArrayList();
                    InsertItem(item, ref y);
                }
            }

            // Adjust the frame height
            currentFrame.Height = y + 7;

            // Return the vertical space the created frame occupies
            height = y + 14;
        }

        private void UpdateTimer_Tick(object sender, EventArgs e)
        {
            UpdateControls();
        }

        private int TextWidth(string s)
        {
            System.Drawing.Graphics g = this.CreateGraphics();
            int w = (int)g.MeasureString(s, this.Font).Width + 7;
            g.Dispose();
            return w;
        }


    }
}