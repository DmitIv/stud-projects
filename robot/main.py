import pyFiles.robot3d as robot
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Searching sequence of mathematical operations for reaching one number from another. Only for positive integer"
    )
    parser.add_argument("start", type=int, help="Start number")
    parser.add_argument("end", type=int, help="Target number")
    args = parser.parse_args()

    a: int = args.start
    b: int = args.end
    r: robot.Robot2D = robot.Robot2D(a, b)

    try:
        start_time: float = time.time()
        result = r.run()
        execution_time: float = round(time.time() - start_time, ndigits=6)
        print(result)
        print("------ {} s ------".format(execution_time))
    except Exception as e:
        print(e)

    r: robot.RobotReverse = robot.RobotReverse(a, b)

    try:
        start_time: float = time.time()
        result = r.run()
        execution_time: float = round(time.time() - start_time, ndigits=6)
        print(result)
        print("------ {} s ------".format(execution_time))
    except Exception as e:
        print(e)

    r: robot.Robot3D = robot.Robot3D(a, b)

    # try:
    start_time: float = time.time()
    result = r.run()
    execution_time: float = round(time.time() - start_time, ndigits=6)
    print(result)
    print("------ {} s ------".format(execution_time))
    # except Exception as e:
    #     print(e)