import subprocess
from datetime import date
import sys
import os
version = "python3"

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(parentddir)
sys.path.append(parentddir + "/bs_ml")


input_path = "bayesian_opt/"
input_file = 'bayesian_optimization'

#input_path = input("Enter your input path of input file starting at bs_ml/.. : ")
#input_file = input("Enter your input file : ")

from bayesian_opt.bayesian_optimization import init_points, n_iter

output_path = "outputs/"
output_file = f'{input_file}_output_ip={init_points}_ni={n_iter}_{date.today()}.txt'

###########################################

total_ipath = input_path + input_file + ".py"
total_opath = output_path + output_file

# Execute the Python script and capture its output
 #capture_output=True,

# Write the output to a text file
try:
    result = subprocess.run([version, total_ipath], text=True, stderr = subprocess.STDOUT, stdout = subprocess.PIPE)
    output_text = result.stdout
    with open(total_opath, 'w') as file:
        file.write(output_text)
    
    print(output_text)
    
    if result.returncode != 0: 
        file.write(f"Script execution error (return code: {result.returncode})")

except FileNotFoundError:
    file.write("Error: Python executable 'python' not found. Make sure Python is installed and added to PATH.")


print(f"Output written to {total_opath}.")