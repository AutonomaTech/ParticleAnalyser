namespace List_VCD_Properties
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.Tree = new System.Windows.Forms.TreeView();
            this.btnShowPage = new System.Windows.Forms.Button();
            this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
            this.btnSelectDevice = new System.Windows.Forms.Button();
            this.CtrlFrame = new System.Windows.Forms.GroupBox();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // Tree
            // 
            this.Tree.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)));
            this.Tree.Location = new System.Drawing.Point(8, 41);
            this.Tree.Name = "Tree";
            this.Tree.Size = new System.Drawing.Size(248, 458);
            this.Tree.TabIndex = 0;
            this.Tree.AfterSelect += new System.Windows.Forms.TreeViewEventHandler(this.treeView1_AfterSelect);
            // 
            // btnShowPage
            // 
            this.btnShowPage.Location = new System.Drawing.Point(139, 12);
            this.btnShowPage.Name = "btnShowPage";
            this.btnShowPage.Size = new System.Drawing.Size(117, 23);
            this.btnShowPage.TabIndex = 2;
            this.btnShowPage.Text = "Show Property Page";
            this.btnShowPage.UseVisualStyleBackColor = true;
            this.btnShowPage.Click += new System.EventHandler(this.btnShowPage_Click);
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(262, 12);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(540, 409);
            this.icImagingControl1.TabIndex = 3;
            // 
            // btnSelectDevice
            // 
            this.btnSelectDevice.Location = new System.Drawing.Point(8, 12);
            this.btnSelectDevice.Name = "btnSelectDevice";
            this.btnSelectDevice.Size = new System.Drawing.Size(125, 23);
            this.btnSelectDevice.TabIndex = 4;
            this.btnSelectDevice.Text = "Select Device";
            this.btnSelectDevice.UseVisualStyleBackColor = true;
            this.btnSelectDevice.Click += new System.EventHandler(this.btnSelectDevice_Click);
            // 
            // CtrlFrame
            // 
            this.CtrlFrame.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.CtrlFrame.Location = new System.Drawing.Point(268, 429);
            this.CtrlFrame.Name = "CtrlFrame";
            this.CtrlFrame.Size = new System.Drawing.Size(533, 69);
            this.CtrlFrame.TabIndex = 5;
            this.CtrlFrame.TabStop = false;
            this.CtrlFrame.Text = "Range";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(813, 511);
            this.Controls.Add(this.CtrlFrame);
            this.Controls.Add(this.btnSelectDevice);
            this.Controls.Add(this.icImagingControl1);
            this.Controls.Add(this.btnShowPage);
            this.Controls.Add(this.Tree);
            this.Name = "Form1";
            this.Text = "List VCD Properties";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TreeView Tree;
        private System.Windows.Forms.Button btnShowPage;
        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Button btnSelectDevice;
        private System.Windows.Forms.GroupBox CtrlFrame;
    }
}

