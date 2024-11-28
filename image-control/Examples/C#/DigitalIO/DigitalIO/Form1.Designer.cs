namespace DigitalIO
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
            this.cmdReadDigitalInput = new System.Windows.Forms.Button();
            this.cmdWriteDigitalOutput = new System.Windows.Forms.Button();
            this.chkDigitalInputState = new System.Windows.Forms.CheckBox();
            this.chkDigitalOutputState = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(8, 8);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(535, 384);
            this.icImagingControl1.TabIndex = 0;
            // 
            // cmdReadDigitalInput
            // 
            this.cmdReadDigitalInput.Location = new System.Drawing.Point(8, 398);
            this.cmdReadDigitalInput.Name = "cmdReadDigitalInput";
            this.cmdReadDigitalInput.Size = new System.Drawing.Size(75, 23);
            this.cmdReadDigitalInput.TabIndex = 1;
            this.cmdReadDigitalInput.Text = "Read Input";
            this.cmdReadDigitalInput.UseVisualStyleBackColor = true;
            this.cmdReadDigitalInput.Click += new System.EventHandler(this.cmdReadDigitalInput_Click);
            // 
            // cmdWriteDigitalOutput
            // 
            this.cmdWriteDigitalOutput.Location = new System.Drawing.Point(8, 427);
            this.cmdWriteDigitalOutput.Name = "cmdWriteDigitalOutput";
            this.cmdWriteDigitalOutput.Size = new System.Drawing.Size(75, 23);
            this.cmdWriteDigitalOutput.TabIndex = 2;
            this.cmdWriteDigitalOutput.Text = "Write Input";
            this.cmdWriteDigitalOutput.UseVisualStyleBackColor = true;
            this.cmdWriteDigitalOutput.Click += new System.EventHandler(this.cmdWriteDigitalOutput_Click);
            // 
            // chkDigitalInputState
            // 
            this.chkDigitalInputState.AutoSize = true;
            this.chkDigitalInputState.Location = new System.Drawing.Point(89, 402);
            this.chkDigitalInputState.Name = "chkDigitalInputState";
            this.chkDigitalInputState.Size = new System.Drawing.Size(78, 17);
            this.chkDigitalInputState.TabIndex = 3;
            this.chkDigitalInputState.Text = "Input State";
            this.chkDigitalInputState.UseVisualStyleBackColor = true;
            // 
            // chkDigitalOutputState
            // 
            this.chkDigitalOutputState.AutoSize = true;
            this.chkDigitalOutputState.Location = new System.Drawing.Point(89, 431);
            this.chkDigitalOutputState.Name = "chkDigitalOutputState";
            this.chkDigitalOutputState.Size = new System.Drawing.Size(86, 17);
            this.chkDigitalOutputState.TabIndex = 4;
            this.chkDigitalOutputState.Text = "Output State";
            this.chkDigitalOutputState.UseVisualStyleBackColor = true;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(552, 454);
            this.Controls.Add(this.chkDigitalOutputState);
            this.Controls.Add(this.chkDigitalInputState);
            this.Controls.Add(this.cmdWriteDigitalOutput);
            this.Controls.Add(this.cmdReadDigitalInput);
            this.Controls.Add(this.icImagingControl1);
            this.Name = "Form1";
            this.Text = "DigitalIO";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Button cmdReadDigitalInput;
        private System.Windows.Forms.Button cmdWriteDigitalOutput;
        private System.Windows.Forms.CheckBox chkDigitalInputState;
        private System.Windows.Forms.CheckBox chkDigitalOutputState;
    }
}

