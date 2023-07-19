import subprocess

version = "python3"
path = "bayesian_opt/"
input_script = 'bayesian_optimization.py'
output_file = 'bayes_opt_output.txt'

total_path = path + input_script

# Execute the Python script and capture its output
result = subprocess.run(['python', total_path], capture_output=True, text=True)
output_text = result.stdout

# Write the output to a text file
with open(output_file, 'w') as file:
    file.write(output_text)

print(f"Output written to {output_file}.")