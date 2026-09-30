import subprocess
import sys

def execute_pipeline():
    print("=========================================")
    print("Starting Lab 5: Production Data Pipeline")
    print("=========================================")

    scripts = [
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py",
    ]

    for script in scripts:
        print(f"\n---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted at {script}.")
            sys.exit(result.returncode)

    print("\n[SUCCESS] Lab 5 Production Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
