import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

def select_mode():
    """
    Allows to choose between:
    [1] Single-stream mode  (autonomous_car.py)
    [2] Multi-stream mode   (3-camera grid: front, left, right)
    """
    print()
    print(" Autonomous Car Perception System")
    print(" [1]  Single-Stream Mode")
    print(" [2]  Multi-Stream Mode (3 cameras)")
    print()

    choice = input("Select mode (1/2): ").strip()
    return "multi" if choice == "2" else "single"

def main():
    mode = select_mode()

    if mode == "single":
        # Run the original pipeline
        print("\n>>> Launching Single-Stream mode...\n")

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../.."))
        os.chdir(project_root)

        # Import and run original main
        from project import autonomous_car
        autonomous_car.main()

    else:
        # Run the multi-stream pipeline
        print("\nLaunching Multi-Stream mode (FRONT / LEFT / RIGHT)\n")
        import stream_manager
        stream_manager.run()


if __name__ == "__main__":
    main()
