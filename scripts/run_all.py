import subprocess


def run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    run(["python", "scripts/run_eda.py"])
    run(["python", "scripts/build_panel.py"])


if __name__ == "__main__":
    main()