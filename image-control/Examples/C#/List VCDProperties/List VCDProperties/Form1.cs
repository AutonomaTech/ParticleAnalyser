using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace List_VCD_Properties
{
    public partial class Form1 : Form
    {
        private List<Control> _currentControls = new List<Control>();

        public Form1()
        {
            InitializeComponent();
        }

        
//<<formload
        private void Form1_Load(object sender, EventArgs e)
        {
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.");
                this.Close();
                return;
            }
            UpdateStateAfterDeviceSelect();
        }
//>>

//<<btnselectdevice
        private void btnSelectDevice_Click(object sender, EventArgs e)
        {
            // The device settings dialog needs the live mode to be stopped
            if (icImagingControl1.LiveVideoRunning)
            {
                icImagingControl1.LiveStop();
            }

            // Show the device settings dialog
            icImagingControl1.ShowDeviceSettingsDialog();

            // If no device was selected, exit
            if (!icImagingControl1.DeviceValid)
            {
                MessageBox.Show("No device was selected.");
                this.Close();
                return;
            }

            UpdateStateAfterDeviceSelect();
        }

        private void UpdateStateAfterDeviceSelect()
        {
            // If no device was selected, exit
            // Print all properties into the debug window
            ListAllPropertyItems();

            // Start live mode
            icImagingControl1.LiveStart();

            // (re-)initialize the tree view
            BuildVCDPropertiesTree();
        }
//>>

//<<btnshowpage
        private void btnShowPage_Click(object sender, EventArgs e)
        {
            if (icImagingControl1.DeviceValid)
            {
                // Show the built-in properties dialog
                icImagingControl1.ShowPropertyDialog();
            }
        }
//>>


//<<queryitems
        private void BuildVCDPropertiesTreeItems(TreeNode pp, TIS.Imaging.VCDPropertyItems props)
        {
            // Iterate through all VCDPropertyItems and insert them into the tree
            foreach (TIS.Imaging.VCDPropertyItem item in props)
            {
                // Create a new tree node for the item
                TreeNode newNode = new TreeNode(item.Name, 0, 0);
                pp.Nodes.Add(newNode);

                QueryVCDPropertyElements(newNode, item);
            }
        }
//>>
//<<queryelements
        private void QueryVCDPropertyElements(TreeNode pp, TIS.Imaging.VCDPropertyItem item)
        {
            foreach (TIS.Imaging.VCDPropertyElement elem in item.Elements)
            {
                TreeNode newNode = null;

                if( elem.ElementGUID == TIS.Imaging.VCDGUIDs.VCDElement_Value )
                    newNode = new TreeNode("VCDElement_Value: '" + elem.Name + "'", 1, 1);
                else if ( elem.ElementGUID == TIS.Imaging.VCDGUIDs.VCDElement_Auto )
                    newNode = new TreeNode("VCDElement_Auto: '" + elem.Name + "'", 2, 2);       
                else if ( elem.ElementGUID == TIS.Imaging.VCDGUIDs.VCDElement_OnePush )
                    newNode = new TreeNode("VCDElement_OnePush: '" + elem.Name + "'", 3, 3);
                else if ( elem.ElementGUID == TIS.Imaging.VCDGUIDs.VCDElement_WhiteBalanceRed )
                    newNode = new TreeNode("VCDElement_WhiteBalanceRed: '" + elem.Name + "'", 4, 4);
                else if ( elem.ElementGUID == TIS.Imaging.VCDGUIDs.VCDElement_WhiteBalanceBlue )
                    newNode = new TreeNode("VCDElement_WhiteBalanceBlue: '" + elem.Name + "'", 4, 4);
                else
                    newNode = new TreeNode("Other Element ID: '" + elem.Name + "'", 4, 4);

                pp.Nodes.Add(newNode);

                // Insert all interfaces
                QueryVCDPropertyInterface(newNode, elem);
            }
        }
//>>

//<<queryinterface
        private void QueryVCDPropertyInterface(TreeNode pp, TIS.Imaging.VCDPropertyElement elem)
        {
            foreach (TIS.Imaging.VCDPropertyInterface itf in elem)
            {
                TreeNode newNode;
                if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_AbsoluteValue )
                    newNode = new TreeNode("AbsoluteValue", 4, 4);
                else if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_MapStrings )
                    newNode = new TreeNode("MapStrings", 6, 6);
                else if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Range )
                    newNode = new TreeNode("Range", 4, 4);
                else if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Switch )
                    newNode = new TreeNode("Switch", 5, 5);
                else if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Button )
                    newNode = new TreeNode("Button", 3, 3);
                else
                {
                    newNode = new TreeNode(itf.InterfaceGUID.ToString(), 4, 4);
                }

                // The Tag property holds the interface at the node.
                newNode.Tag = itf;
                pp.Nodes.Add(newNode);
            }
        }
//>>

//<<queryprops
        private void BuildVCDPropertiesTree()
        {
            // Erase the complete tree.
            Tree.Nodes.Clear();

            // Fill the tree.
            TreeNode root = new TreeNode("VCDPropertyItems");
            Tree.Nodes.Add(root);

            BuildVCDPropertiesTreeItems(root, icImagingControl1.VCDPropertyItems);

            root.ExpandAll();
            Tree.SelectedNode = root;
        }
//>>

//<<tree_nodeclick
        private void treeView1_AfterSelect(object sender, TreeViewEventArgs e)
        {
            // If the Tag property is empty, no leaf node was selected.
            if (Tree.SelectedNode.Tag == null)
            {
                return;
            }

            // Hide all controls
            foreach( var ctrl in _currentControls )
            {
                ctrl.Dispose();
            }
            _currentControls.Clear();

            TIS.Imaging.VCDPropertyInterface itf = Tree.SelectedNode.Tag as TIS.Imaging.VCDPropertyInterface;
            if (itf != null)
            {
                itf.Update();
                // Show the control group matching the type of the selected interface
                // and initialize it.
                if ( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_AbsoluteValue )
                     ShowAbsoluteValueControl(itf);
                else if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_MapStrings)
                    ShowComboBoxControl(itf);
                else if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Range)
                    ShowRangeControl(itf);
                else if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Switch)
                    ShowSwitchControl(itf);
                else if( itf.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Button)
                    ShowButtonControl(itf);
            }
        }
//>>

//<<showcontrol
        private void ShowAbsoluteValueControl(TIS.Imaging.VCDPropertyInterface itf)
        {
            var newCtrl = new AbsValSlider((TIS.Imaging.VCDAbsoluteValueProperty)itf);
            CtrlFrame.Controls.Add(newCtrl);
            newCtrl.SetBounds(20, 20, 500, 27);
            CtrlFrame.Text = "Absolute Value";
            _currentControls.Add(newCtrl);
        }
//>>
        private void ShowComboBoxControl(TIS.Imaging.VCDPropertyInterface itf)
        {
            var newCtrl = new StringCombo((TIS.Imaging.VCDMapStringsProperty)itf);
            CtrlFrame.Controls.Add(newCtrl);
            newCtrl.SetBounds(20, 20, 200, 27);
            CtrlFrame.Text = "MapStrings";
            _currentControls.Add(newCtrl);
        }

        private void ShowRangeControl(TIS.Imaging.VCDPropertyInterface itf)
        {
            var newCtrl = new RangeSlider((TIS.Imaging.VCDRangeProperty)itf);
            CtrlFrame.Controls.Add(newCtrl);
            newCtrl.SetBounds(20, 20, 200, 27);
            CtrlFrame.Text = "Range";
            _currentControls.Add(newCtrl);
        }

        private void ShowSwitchControl(TIS.Imaging.VCDPropertyInterface itf)
        {
            var newCtrl = new Switch((TIS.Imaging.VCDSwitchProperty)itf);
            CtrlFrame.Controls.Add(newCtrl);
            newCtrl.SetBounds(20, 20, 200, 27);
            CtrlFrame.Text = "Switch";
            _currentControls.Add(newCtrl);
        }

        private void ShowButtonControl(TIS.Imaging.VCDPropertyInterface itf)
        {
            var newCtrl = new PushButton((TIS.Imaging.VCDButtonProperty)itf);
            CtrlFrame.Controls.Add(newCtrl);
            newCtrl.SetBounds(20, 20, 100, 27);
            CtrlFrame.Text = "Button";
            _currentControls.Add(newCtrl);
        }


        //
        // ListAllPropertyItems
        //
        // This sub builds an item - element - lists all names and values of the properties in the
        // debug window. It shows, how to enumerate all properties, elements and interfaces. The
        // interfaces have to be "casted" to the appropriate interface types like range, absolute
        // value etc. to get a correct access to the current interface's properties.
        //
//<<listallitems
        private void ListAllPropertyItems()
        {
            // Get all property items
            foreach (TIS.Imaging.VCDPropertyItem PropertyItem in icImagingControl1.VCDPropertyItems)
            {
                System.Diagnostics.Debug.WriteLine(PropertyItem.Name);

                // Get all property elements of the current property item.
                foreach (TIS.Imaging.VCDPropertyElement PropertyElement in PropertyItem.Elements)
                {
                    System.Diagnostics.Debug.WriteLine("    Element : " + PropertyElement.Name);

                    // Get all interfaces of the current property element.
                    foreach (TIS.Imaging.VCDPropertyInterface PropertyInterFace in PropertyElement)
                    {
                        System.Diagnostics.Debug.Write("        Interface ");

                        try
                        {
                            // Cast the current interface into the appropriate type to access
                            // the special interface properties.
                            if( PropertyInterFace.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_AbsoluteValue )
                            {
                                var AbsoluteValue = (TIS.Imaging.VCDAbsoluteValueProperty)PropertyInterFace;
                                System.Diagnostics.Debug.Write("Absolute Value : ");
                                System.Diagnostics.Debug.WriteLine(AbsoluteValue.Value.ToString());
                            }
                            else if( PropertyInterFace.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_MapStrings )
                            {
                                var MapString = (TIS.Imaging.VCDMapStringsProperty)PropertyInterFace;
                                System.Diagnostics.Debug.Write("Mapstring : ");
                                System.Diagnostics.Debug.WriteLine(MapString.String);
                            }
                            else if( PropertyInterFace.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Switch )
                            {
                                var Switch = (TIS.Imaging.VCDSwitchProperty)PropertyInterFace;
                                System.Diagnostics.Debug.Write("Switch : ");
                                System.Diagnostics.Debug.WriteLine(Switch.Switch.ToString());
                            }
                            else if( PropertyInterFace.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Button )
                            {
                                var Button = (TIS.Imaging.VCDButtonProperty)PropertyInterFace;
                                System.Diagnostics.Debug.WriteLine("Button");
                            }
                            else if( PropertyInterFace.InterfaceGUID == TIS.Imaging.VCDGUIDs.VCDInterface_Range )
                            {
                                var Range = (TIS.Imaging.VCDRangeProperty)PropertyInterFace;
                                System.Diagnostics.Debug.Write("Range : ");
                                System.Diagnostics.Debug.WriteLine(Range.Value.ToString());
                            }

                        }
                        catch( Exception ex )
                        {
                            System.Diagnostics.Debug.WriteLine("<error>:" + ex);
                        }
                    }
                }
            }
        }
        //>>

    }
}