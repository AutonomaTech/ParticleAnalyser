namespace StandardProperties
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
            this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
            this.label1 = new System.Windows.Forms.Label();
            this.trackBarExposure = new System.Windows.Forms.TrackBar();
            this.checkBoxExposure = new System.Windows.Forms.CheckBox();
            this.checkBoxGain = new System.Windows.Forms.CheckBox();
            this.trackBarGain = new System.Windows.Forms.TrackBar();
            this.label2 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExposure)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarGain)).BeginInit();
            this.SuspendLayout();
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.DeviceListChangedExecutionMode = TIS.Imaging.EventExecutionMode.Invoke;
            this.icImagingControl1.DeviceLostExecutionMode = TIS.Imaging.EventExecutionMode.AsyncInvoke;
            this.icImagingControl1.ImageAvailableExecutionMode = TIS.Imaging.EventExecutionMode.MultiThreaded;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(14, 17);
            this.icImagingControl1.Margin = new System.Windows.Forms.Padding(6);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(953, 540);
            this.icImagingControl1.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(24, 577);
            this.label1.Margin = new System.Windows.Forms.Padding(6, 0, 6, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(103, 25);
            this.label1.TabIndex = 1;
            this.label1.Text = "Exposure";
            // 
            // trackBarExposure
            // 
            this.trackBarExposure.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.trackBarExposure.Location = new System.Drawing.Point(148, 569);
            this.trackBarExposure.Margin = new System.Windows.Forms.Padding(6);
            this.trackBarExposure.Name = "trackBarExposure";
            this.trackBarExposure.Size = new System.Drawing.Size(572, 90);
            this.trackBarExposure.TabIndex = 2;
            this.trackBarExposure.Scroll += new System.EventHandler(this.TrackBarExposure_Scroll);
            // 
            // checkBoxExposure
            // 
            this.checkBoxExposure.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxExposure.AutoSize = true;
            this.checkBoxExposure.Location = new System.Drawing.Point(732, 583);
            this.checkBoxExposure.Margin = new System.Windows.Forms.Padding(6);
            this.checkBoxExposure.Name = "checkBoxExposure";
            this.checkBoxExposure.Size = new System.Drawing.Size(88, 29);
            this.checkBoxExposure.TabIndex = 3;
            this.checkBoxExposure.Text = "Auto";
            this.checkBoxExposure.UseVisualStyleBackColor = true;
            this.checkBoxExposure.CheckedChanged += new System.EventHandler(this.CheckBoxExposure_CheckedChanged);
            // 
            // checkBoxGain
            // 
            this.checkBoxGain.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxGain.AutoSize = true;
            this.checkBoxGain.Location = new System.Drawing.Point(732, 672);
            this.checkBoxGain.Margin = new System.Windows.Forms.Padding(6);
            this.checkBoxGain.Name = "checkBoxGain";
            this.checkBoxGain.Size = new System.Drawing.Size(88, 29);
            this.checkBoxGain.TabIndex = 6;
            this.checkBoxGain.Text = "Auto";
            this.checkBoxGain.UseVisualStyleBackColor = true;
            this.checkBoxGain.CheckedChanged += new System.EventHandler(this.CheckBoxGain_CheckedChanged);
            // 
            // trackBarGain
            // 
            this.trackBarGain.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.trackBarGain.Location = new System.Drawing.Point(148, 658);
            this.trackBarGain.Margin = new System.Windows.Forms.Padding(6);
            this.trackBarGain.Name = "trackBarGain";
            this.trackBarGain.Size = new System.Drawing.Size(572, 90);
            this.trackBarGain.TabIndex = 5;
            this.trackBarGain.Scroll += new System.EventHandler(this.TrackBarGain_Scroll);
            // 
            // label2
            // 
            this.label2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(24, 666);
            this.label2.Margin = new System.Windows.Forms.Padding(6, 0, 6, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(57, 25);
            this.label2.TabIndex = 4;
            this.label2.Text = "Gain";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(12F, 25F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(981, 763);
            this.Controls.Add(this.checkBoxGain);
            this.Controls.Add(this.trackBarGain);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.checkBoxExposure);
            this.Controls.Add(this.trackBarExposure);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.icImagingControl1);
            this.Margin = new System.Windows.Forms.Padding(6);
            this.Name = "Form1";
            this.Text = "StandardProperties";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExposure)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarGain)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TrackBar trackBarExposure;
        private System.Windows.Forms.CheckBox checkBoxExposure;
        private System.Windows.Forms.CheckBox checkBoxGain;
        private System.Windows.Forms.TrackBar trackBarGain;
        private System.Windows.Forms.Label label2;
    }
}

