namespace BrightnessProperty
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
            this.sldBrightness = new System.Windows.Forms.TrackBar();
            this.chkBrightnessAuto = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.sldBrightness)).BeginInit();
            this.SuspendLayout();
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(7, 9);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(424, 280);
            this.icImagingControl1.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 299);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(56, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Brightness";
            // 
            // sldBrightness
            // 
            this.sldBrightness.Location = new System.Drawing.Point(74, 295);
            this.sldBrightness.Name = "sldBrightness";
            this.sldBrightness.Size = new System.Drawing.Size(286, 45);
            this.sldBrightness.TabIndex = 2;
            this.sldBrightness.Scroll += new System.EventHandler(this.sldBrightness_Scroll);
            // 
            // chkBrightnessAuto
            // 
            this.chkBrightnessAuto.AutoSize = true;
            this.chkBrightnessAuto.Location = new System.Drawing.Point(366, 302);
            this.chkBrightnessAuto.Name = "chkBrightnessAuto";
            this.chkBrightnessAuto.Size = new System.Drawing.Size(48, 17);
            this.chkBrightnessAuto.TabIndex = 3;
            this.chkBrightnessAuto.Text = "Auto";
            this.chkBrightnessAuto.UseVisualStyleBackColor = true;
            this.chkBrightnessAuto.CheckedChanged += new System.EventHandler(this.chkBrightnessAuto_CheckedChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(438, 331);
            this.Controls.Add(this.chkBrightnessAuto);
            this.Controls.Add(this.sldBrightness);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.icImagingControl1);
            this.Name = "Form1";
            this.Text = "BrightnessProperty";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.sldBrightness)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TrackBar sldBrightness;
        private System.Windows.Forms.CheckBox chkBrightnessAuto;
    }
}

