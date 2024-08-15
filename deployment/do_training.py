import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/main')))

from training.train import main

sys.argv += ["-c", "config.properties"]

if __name__ == '__main__':
    main()
