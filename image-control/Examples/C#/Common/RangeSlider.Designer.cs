
    partial class RangeSlider
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.ValueText = new System.Windows.Forms.TextBox();
            this.Slider = new System.Windows.Forms.TrackBar();
            ((System.ComponentModel.ISupportInitialize)(this.Slider)).BeginInit();
            this.SuspendLayout();
            // 
            // ValueText
            // 
            this.ValueText.Enabled = false;
            this.ValueText.Location = new System.Drawing.Point(315, 7);
            this.ValueText.Name = "ValueText";
            this.ValueText.Size = new System.Drawing.Size(87, 20);
            this.ValueText.TabIndex = 3;
            this.ValueText.Visible = false;
            // 
            // Slider
            // 
            this.Slider.Location = new System.Drawing.Point(3, 3);
            this.Slider.Name = "Slider";
            this.Slider.Size = new System.Drawing.Size(306, 45);
            this.Slider.TabIndex = 2;
            this.Slider.Scroll += new System.EventHandler(this.Slider_Scroll);
            // 
            // RangeSlider
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.ValueText);
            this.Controls.Add(this.Slider);
            this.Name = "RangeSlider";
            this.Size = new System.Drawing.Size(410, 45);
            ((System.ComponentModel.ISupportInitialize)(this.Slider)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox ValueText;
        private System.Windows.Forms.TrackBar Slider;
    }

