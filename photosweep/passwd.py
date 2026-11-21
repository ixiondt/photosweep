"""Set the sign-in password (AUTH_MODE=password).

    docker compose exec web python -m photosweep.passwd
    printf '%s\\n' "$PW" | docker compose exec -T web python -m photosweep.passwd --stdin

Setting a new one signs every browser out.
"""
import argparse
import getpass
import sys

from . import auth, config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdin", action="store_true", help="read the password from one line of standard input")
    args = ap.parse_args()
    if args.stdin:
        pw = sys.stdin.readline().rstrip("\r\n")
    else:
        pw = getpass.getpass("New password: ")
        if getpass.getpass("Again: ") != pw:
            sys.exit("The two did not match; nothing changed.")
    try:
        auth.set_password(pw)
    except ValueError as exc:
        sys.exit(f"Not set: {exc}.")
    print(f"Password set in {config.PASSWORD_FILE}. Any browser signed in before must sign in again.")


if __name__ == "__main__":
    main()
