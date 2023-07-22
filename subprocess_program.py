import subprocess
from datetime import time

version = "python3"

input_path = "bayesian_opt/"
input_file = 'bayesian_optimization.py'

output_path = "output/"
output_file = f'bayes_opt_output_{time.today()}.txt'

###########################################

total_ipath = input_path + input_file
total_opath = output_path + output_file

# Execute the Python script and capture its output
result = subprocess.run([version, total_ipath], capture_output=True, text=True)
output_text = result.stdout

# Write the output to a text file
with open(output_file, 'w') as file:
    file.write(total_opath)

print(f"Output written to {total_opath}.")