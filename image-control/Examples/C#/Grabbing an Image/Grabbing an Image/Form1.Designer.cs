namespace Grabbing_an_Image
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
            this.cmdStartLive = new System.Windows.Forms.Button();
            this.cmdSaveBitmap = new System.Windows.Forms.Button();
            this.cmdStopLive = new System.Windows.Forms.Button();
            this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // cmdStartLive
            // 
            this.cmdStartLive.Location = new System.Drawing.Point(7, 246);
            this.cmdStartLive.Name = "cmdStartLive";
            this.cmdStartLive.Size = new System.Drawing.Size(75, 23);
            this.cmdStartLive.TabIndex = 1;
            this.cmdStartLive.Text = "Start Live";
            this.cmdStartLive.UseVisualStyleBackColor = true;
            this.cmdStartLive.Click += new System.EventHandler(this.cmdStartLive_Click);
            // 
            // cmdSaveBitmap
            // 
            this.cmdSaveBitmap.Location = new System.Drawing.Point(261, 246);
            this.cmdSaveBitmap.Name = "cmdSaveBitmap";
            this.cmdSaveBitmap.Size = new System.Drawing.Size(75, 23);
            this.cmdSaveBitmap.TabIndex = 2;
            this.cmdSaveBitmap.Text = "Save Bitmap";
            this.cmdSaveBitmap.UseVisualStyleBackColor = true;
            this.cmdSaveBitmap.Click += new System.EventHandler(this.cmdSaveBitmap_Click);
            // 
            // cmdStopLive
            // 
            this.cmdStopLive.Location = new System.Drawing.Point(135, 246);
            this.cmdStopLive.Name = "cmdStopLive";
            this.cmdStopLive.Size = new System.Drawing.Size(75, 23);
            this.cmdStopLive.TabIndex = 3;
            this.cmdStopLive.Text = "Stop Live";
            this.cmdStopLive.UseVisualStyleBackColor = true;
            this.cmdStopLive.Click += new System.EventHandler(this.cmdStopLive_Click);
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(9, 8);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(326, 228);
            this.icImagingControl1.TabIndex = 4;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(344, 277);
            this.Controls.Add(this.icImagingControl1);
            this.Controls.Add(this.cmdStopLive);
            this.Controls.Add(this.cmdSaveBitmap);
            this.Controls.Add(this.cmdStartLive);
            this.Name = "Form1";
            this.Text = "Grabbing an Image";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button cmdStartLive;
        private System.Windows.Forms.Button cmdSaveBitmap;
        private System.Windows.Forms.Button cmdStopLive;
        private TIS.Imaging.ICImagingControl icImagingControl1;
    }
}

