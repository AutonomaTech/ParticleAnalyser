import glob, os
import shutil  # Import shutil for file moving

# Define the SMB server and share
SMB_SERVER = "AT-SERVER"
SMB_SHARE = "ImageDataShare"

# Define the local destination folder where the file will be moved
local_destination_folder = r"C:\Users\Autonoma\Desktop\ParticleAnalyser\Samples"

# Open the file in read mode with correctly formatted f-string
file_path = f"\\\\{SMB_SERVER}\\\\{SMB_SHARE}\\\\ABCD_20250227_081709.bmp"  # Use f-string here
with open(file_path, 'r') as f:
    content = f.read()  # Read the file content
    print(content)  # Print the file content

# Define the new local path for moving the file
new_file_path = os.path.join(local_destination_folder, "ABCD_20250227_081709.json")

# Move the file to the local folder
shutil.move(file_path, new_file_path)
print(f"File moved to {new_file_path}")

# List all files recursively on a specific share
for f in glob.glob(r'\\USER1-PC\Users\**\*', recursive=True):
    print(f)  # Print each file path
    if os.path.isfile(f):
        print(os.path.getmtime(f))  # Print the last modified time of the file