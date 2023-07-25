import subprocess
from datetime import date
import sys
version = "python3"

sys.path.append('../')
input_path = "bayesian_opt/"
input_file = 'bayesian_optimization'

#input_path = input("Enter your input path of input file starting at bs_ml/.. : ")
#input_file = input("Enter your input file : ")

output_path = "outputs/"
output_file = f'{input_file}_output_{date.today()}.txt'

###########################################

total_ipath = input_path + input_file + ".py"
total_opath = output_path + output_file

# Execute the Python script and capture its output
result = subprocess.run([version, total_ipath], capture_output=True, text=True)
output_text = result.stdout

# Write the output to a text file
with open(total_opath, 'wb') as file:
    file.write(output_text)

print(f"Output written to {total_opath}.")