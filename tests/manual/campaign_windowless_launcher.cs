// GUI-subsystem bootstrap only. Recovery and singleton ownership stay in Python.
// Compile with Framework csc /target:winexe /reference:System.Web.Extensions.dll.
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;

class CampaignWindowlessLauncher
{
    public class Configuration
    {
        public string interpreter;
        public string arguments;
        public string working_directory;
        public string log_directory;
        public string progress;
        public string completion_status;
    }

    [STAThread]
    static int Main(string[] args)
    {
        string errorFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "launcher_error.log");
        try
        {
            if (args.Length != 1) throw new ArgumentException("Expected one launcher configuration path");
            // Full campaign receipts can exceed the serializer's 2 MB default.
            var json = new JavaScriptSerializer { MaxJsonLength = Int32.MaxValue };
            var config = json.Deserialize<Configuration>(File.ReadAllText(args[0], Encoding.UTF8));
            if (!String.Equals(Path.GetFullPath(config.interpreter), @"F:\conda_env\flyagent\python.exe", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Only the configured flyagent interpreter is allowed");
            // Completion prevents periodic triggers from starting new watchdogs.
            if (File.Exists(config.progress))
            {
                var state = json.Deserialize<System.Collections.Generic.Dictionary<string, object>>(File.ReadAllText(config.progress, Encoding.UTF8));
                object value;
                if (state.TryGetValue("status", out value))
                {
                    string status = Convert.ToString(value);
                    if (status == "seen_complete" || status == "campaign_complete" || status == "awaiting_image_budget_choice" ||
                        (!String.IsNullOrEmpty(config.completion_status) && status == config.completion_status)) return 0;
                }
            }
            Directory.CreateDirectory(config.log_directory);
            string prefix = Path.Combine(config.log_directory, DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fffffff") + "_" + Process.GetCurrentProcess().Id);
            using (var stdout = new StreamWriter(prefix + ".stdout.log", true, new UTF8Encoding(false)))
            using (var stderr = new StreamWriter(prefix + ".stderr.log", true, new UTF8Encoding(false)))
            using (var child = new Process())
            {
                stdout.AutoFlush = stderr.AutoFlush = true;
                child.StartInfo = new ProcessStartInfo(config.interpreter, config.arguments) {
                    UseShellExecute = false, CreateNoWindow = true,
                    WorkingDirectory = config.working_directory,
                    RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true
                };
                child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) stdout.WriteLine(e.Data); };
                child.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) stderr.WriteLine(e.Data); };
                child.Start();
                child.StandardInput.Close();
                child.BeginOutputReadLine(); child.BeginErrorReadLine();
                child.WaitForExit();
                return child.ExitCode;
            }
        }
        catch (Exception exception)
        {
            // Never display an error dialog from a background scheduled task.
            try { File.AppendAllText(errorFile, DateTime.UtcNow.ToString("o") + " " + exception + Environment.NewLine); } catch { }
            return 1;
        }
    }
}
