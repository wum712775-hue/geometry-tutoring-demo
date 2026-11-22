"""Run the complete synthetic-data demonstration from any working directory."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Use small resampling budgets for an initial smoke run.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    commands = [
        ["src/generate_data.py"],
        ["src/analysis.py"],
        ["src/analysis.py", "--dataset", "iid"],
        ["src/validate.py"],
        ["src/calibrate.py"],
        ["src/predict_next_event.py"],
        ["src/plot_results.py"],
        ["src/tutoring_replay.py"],
        ["src/build_visual_tutor.py"],
        ["-m", "unittest", "discover", "-s", "tests", "-v"],
    ]
    if args.quick:
        for command in commands:
            if command[0] == "src/analysis.py":
                command.extend(["--permutations", "199", "--bootstrap", "199"])
            elif command[0] == "src/validate.py":
                command.extend(["--replicates", "10"])
            elif command[0] == "src/calibrate.py":
                command.extend(["--replicates", "10", "--permutations", "99"])
    for command in commands:
        print(f"Running: {sys.executable} {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, *command], cwd=root, check=True)
    print("Pipeline completed. Open README.md and figures/ to review the results.")
    if args.quick:
        print("This was a smoke run. Use the default command for the documented full budgets.")


if __name__ == "__main__":
    main()
