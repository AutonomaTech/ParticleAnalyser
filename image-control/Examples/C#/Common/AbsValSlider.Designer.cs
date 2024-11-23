
    partial class AbsValSlider
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
            this.Slider = new System.Windows.Forms.TrackBar();
            this.ValueText = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.Slider)).BeginInit();
            this.SuspendLayout();
            // 
            // Slider
            // 
            this.Slider.Location = new System.Drawing.Point(3, 3);
            this.Slider.Name = "Slider";
            this.Slider.Size = new System.Drawing.Size(306, 45);
            this.Slider.TabIndex = 0;
            this.Slider.Scroll += new System.EventHandler(this.trackBar1_Scroll);
            // 
            // ValueText
            // 
            this.ValueText.Location = new System.Drawing.Point(315, 7);
            this.ValueText.Name = "ValueText";
            this.ValueText.Size = new System.Drawing.Size(87, 20);
            this.ValueText.TabIndex = 1;
            // 
            // AbsValSlider
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.ValueText);
            this.Controls.Add(this.Slider);
            this.Name = "AbsValSlider";
            this.Size = new System.Drawing.Size(408, 40);
            this.Load += new System.EventHandler(this.AbsValSlider_Load);
            ((System.ComponentModel.ISupportInitialize)(this.Slider)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TrackBar Slider;
        private System.Windows.Forms.TextBox ValueText;
    }

