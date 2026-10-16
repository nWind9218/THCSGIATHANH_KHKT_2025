"""Print a CONSOLE_ADMIN_PASSWORD_HASH value for the teacher console.

Usage: python scripts/hash_console_password.py
"""

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.console_auth import hash_password  # noqa: E402

MIN_LENGTH = 10


def main() -> int:
    password = getpass.getpass("Mật khẩu mới: ")
    if getpass.getpass("Nhập lại mật khẩu: ") != password:
        print("Hai lần nhập không giống nhau.", file=sys.stderr)
        return 1
    if len(password) < MIN_LENGTH:
        print(f"Mật khẩu cần có ít nhất {MIN_LENGTH} ký tự.", file=sys.stderr)
        return 1
    print("\nDán dòng sau vào biến môi trường của server:\n")
    print(f"CONSOLE_ADMIN_PASSWORD_HASH={hash_password(password)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
