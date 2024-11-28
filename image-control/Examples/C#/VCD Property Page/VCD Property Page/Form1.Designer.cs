namespace VCD_Property_Page
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
            this.cmdSelectDevice = new System.Windows.Forms.Button();
            this.cmdShowMyDialog = new System.Windows.Forms.Button();
            this.cmdShowOriginalDialog = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(8, 8);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(461, 358);
            this.icImagingControl1.TabIndex = 0;
            // 
            // cmdSelectDevice
            // 
            this.cmdSelectDevice.Location = new System.Drawing.Point(475, 8);
            this.cmdSelectDevice.Name = "cmdSelectDevice";
            this.cmdSelectDevice.Size = new System.Drawing.Size(98, 23);
            this.cmdSelectDevice.TabIndex = 1;
            this.cmdSelectDevice.Text = "Select Device";
            this.cmdSelectDevice.UseVisualStyleBackColor = true;
            this.cmdSelectDevice.Click += new System.EventHandler(this.cmdSelectDevice_Click);
            // 
            // cmdShowMyDialog
            // 
            this.cmdShowMyDialog.Location = new System.Drawing.Point(475, 53);
            this.cmdShowMyDialog.Name = "cmdShowMyDialog";
            this.cmdShowMyDialog.Size = new System.Drawing.Size(98, 23);
            this.cmdShowMyDialog.TabIndex = 2;
            this.cmdShowMyDialog.Text = "Show My Dialog";
            this.cmdShowMyDialog.UseVisualStyleBackColor = true;
            this.cmdShowMyDialog.Click += new System.EventHandler(this.cmdShowMyDialog_Click);
            // 
            // cmdShowOriginalDialog
            // 
            this.cmdShowOriginalDialog.Location = new System.Drawing.Point(475, 82);
            this.cmdShowOriginalDialog.Name = "cmdShowOriginalDialog";
            this.cmdShowOriginalDialog.Size = new System.Drawing.Size(98, 23);
            this.cmdShowOriginalDialog.TabIndex = 3;
            this.cmdShowOriginalDialog.Text = "Show Original Page";
            this.cmdShowOriginalDialog.UseVisualStyleBackColor = true;
            this.cmdShowOriginalDialog.Click += new System.EventHandler(this.cmdShowOriginalDialog_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(578, 381);
            this.Controls.Add(this.cmdShowOriginalDialog);
            this.Controls.Add(this.cmdShowMyDialog);
            this.Controls.Add(this.cmdSelectDevice);
            this.Controls.Add(this.icImagingControl1);
            this.Name = "Form1";
            this.Text = "VCDPropertyDialog Example";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Button cmdSelectDevice;
        private System.Windows.Forms.Button cmdShowMyDialog;
        private System.Windows.Forms.Button cmdShowOriginalDialog;
    }
}

