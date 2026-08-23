import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils import gpu as gpu_mod  # noqa: E402

gpu_mod._present_gpus = lambda: {1, 2, 3, 4}


def main():
    root = sys.argv[1]
    pool = [int(x) for x in sys.argv[2].split(",")]
    hold = float(sys.argv[3])
    detach = "--detach" in sys.argv
    with gpu_mod.gpu_lease(pool, root) as lease:
        print(lease.index, flush=True)
        if detach:
            fd = lease.fileno()
            grandchild = subprocess.Popen(["/bin/sleep", str(int(hold))],
                                          pass_fds=(fd,),
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
            sys.exit(0)  # parent dies; the fd lives on in the grandchild
        time.sleep(hold)


if __name__ == "__main__":
    main()
