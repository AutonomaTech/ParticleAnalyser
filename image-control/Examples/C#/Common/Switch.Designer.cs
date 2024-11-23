
    partial class Switch
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
            this.Check = new System.Windows.Forms.CheckBox();
            this.SuspendLayout();
            // 
            // Check
            // 
            this.Check.AutoSize = true;
            this.Check.Location = new System.Drawing.Point(7, 6);
            this.Check.Name = "Check";
            this.Check.Size = new System.Drawing.Size(80, 17);
            this.Check.TabIndex = 0;
            this.Check.Text = "checkBox1";
            this.Check.UseVisualStyleBackColor = true;
            this.Check.CheckStateChanged += new System.EventHandler(this.checkBox1_CheckStateChanged);
            // 
            // Switch
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.Check);
            this.Name = "Switch";
            this.Size = new System.Drawing.Size(88, 27);
            this.Resize += new System.EventHandler(this.Switch_Resize);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.CheckBox Check;
    }

