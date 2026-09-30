import subprocess
import sys

def execute_pipeline():
    print("=========================================")
    print("Starting Lab 6: Model Registry & Lifecycle")
    print("=========================================")

    scripts = [
        "src/train_registry.py",
        "src/automate_lifecycle.py",
        "src/generate_registry_report.py",
    ]

    for script in scripts:
        print(f"\n---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted at {script}.")
            sys.exit(result.returncode)

    print("\n[SUCCESS] Lab 6 Model Registry Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
