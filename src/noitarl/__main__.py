from __future__ import annotations

import sys

from noitarl.settings import Settings
from noitarl.worker import Worker


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    Worker(Settings()).run(command)


if __name__ == "__main__":
    main()
