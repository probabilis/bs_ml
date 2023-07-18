
import subprocess

with open("output.txt", "wb") as f:
    subprocess.check_call(["python", "decision_trees.py"], stdout=f)
