namespace Strobe
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
            this.chkStrobe = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(10, 10);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(528, 368);
            this.icImagingControl1.TabIndex = 0;
            // 
            // chkStrobe
            // 
            this.chkStrobe.AutoSize = true;
            this.chkStrobe.Location = new System.Drawing.Point(10, 385);
            this.chkStrobe.Name = "chkStrobe";
            this.chkStrobe.Size = new System.Drawing.Size(57, 17);
            this.chkStrobe.TabIndex = 1;
            this.chkStrobe.Text = "Strobe";
            this.chkStrobe.UseVisualStyleBackColor = true;
            this.chkStrobe.CheckedChanged += new System.EventHandler(this.chkStrobe_CheckedChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(552, 414);
            this.Controls.Add(this.chkStrobe);
            this.Controls.Add(this.icImagingControl1);
            this.Name = "Form1";
            this.Text = "Strobe";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.CheckBox chkStrobe;
    }
}

