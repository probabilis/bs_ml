
import subprocess

path = "bayesian_opt/"
program = "bayesian_optimization.py"

with open("output.txt", "wb") as f:
    subprocess.check_call(["python3", path + program], stdout=f)
