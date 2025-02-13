using System;
using System.IO;
using System.Diagnostics;
using Python.Runtime;
using System.Text.RegularExpressions;

class Program
{
    static void InitialiseProcessing(dynamic startModel, string imagePath, string programNumber)
    {
        Console.WriteLine($"Image {imagePath} sent to Python for processing...");
        dynamic modelInstance = startModel.ProcessStartModel();
        modelInstance.saveNewPicture(imagePath, programNumber);
        modelInstance.analyse();
    }
    static void activateConda()
    {
        // Set working directory and create process
        var workingDirectory = @"C:\\Users\\marco\\Desktop\\ParticleAnalyser";
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                RedirectStandardInput = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                WorkingDirectory = workingDirectory
            }
        };
        process.Start();

        // Pass multiple commands to cmd.exe
        using (var sw = process.StandardInput)
        {
            if (sw.BaseStream.CanWrite)
            {
                // Activate the environment
                sw.WriteLine("C:\\Users\\marco\\anaconda3\\Scripts\\activate.bat");
                sw.WriteLine("activate processStart");
            }
        }

        // Read multiple output lines to ensure successful activation
        while (!process.StandardOutput.EndOfStream)
        {
            var line = process.StandardOutput.ReadLine();
            Console.WriteLine(line);
        }
    }

    static void Main(string[] args)
    {
        //activateConda();
        // Dynamically find Python DLL path in the current directory
        string pythonDllPath = "C:\\Users\\marco\\anaconda3\\envs\\processStart\\python310.dll";
        //string pythonDllPath = "C:\\Users\\marco\\Desktop\\ParticleAnalyser\\bin\\Debug\\net9.0\\python311.dll";
        if (!File.Exists(pythonDllPath))
        {
            Console.WriteLine("Error: python311.dll not found in the current directory.");
            return;
        }

        Runtime.PythonDLL = pythonDllPath;
        
        // Initialize Python.NET
        PythonEngine.Initialize();
        Console.WriteLine("------------------");

        string scriptPath = @"C:\\Users\\marco\\Desktop\\ParticleAnalyser\\ProcessStartModel.py";
        Console.WriteLine(scriptPath);
        string imagePath1 = Path.Combine(scriptPath, "picture.jpg");
        string programNumber1 = "program1";

        string imagePath2 = Path.Combine(scriptPath, "image.jpg");
        string programNumber2 = "";

        using (Py.GIL())
        {
            dynamic sys = Py.Import("sys");
            sys.path.append(scriptPath);

            // Import the correct Python script (without .py extension)
            dynamic imageProcessor = Py.Import("ProcessStartModel"); // Change this to match your actual script

            // Add images to the processing queue
            InitialiseProcessing(imageProcessor, imagePath1, programNumber1);
            InitialiseProcessing(imageProcessor, imagePath2, programNumber2);
        }
    }
}
