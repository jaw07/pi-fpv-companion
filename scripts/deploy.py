#!/usr/bin/env python3
"""Cross-platform deploy of pi-fpv-companion to the Pi over WiFi — Windows / macOS / Linux.

Pushes the working tree to the Pi and restarts the systemd service, then verifies
it came up healthy. Uses the system `ssh` (built into Windows 10+, macOS, Linux).
File transfer uses `rsync` when available (fast, incremental — macOS/Linux/WSL);
otherwise falls back to tar-over-ssh, which works on native Windows too (Win10+
ships `tar`). Neither path deletes Pi-only files (.venv / models / var).

Auth (in order of preference):
  1. SSH keys — works on every platform. Run `deploy.py keys` once to set up.
  2. PI_PASS env — on macOS/Linux uses `sshpass` if installed; the sudo password is
     fed to `sudo -S` on all platforms.
  3. Otherwise `ssh` prompts for the password interactively (fine on Windows too,
     just not unattended).

Usage:
  python scripts/deploy.py                  # sync + restart + verify  (default)
  python scripts/deploy.py --no-restart     # sync only
  python scripts/deploy.py --venv           # also reinstall the venv (after dep changes)
  python scripts/deploy.py --dry-run        # show what would sync, change nothing
  python scripts/deploy.py setup            # FIRST-TIME: sync + run install-pi.sh on the Pi
  python scripts/deploy.py keys             # install this host's SSH key (passwordless)
  python scripts/deploy.py status|logs|restart|stop

Env: PI_HOST (default vidtest@192.168.8.160), PI_DIR (/opt/pi-fpv-companion),
     SERVICE (pi-fpv-companion), PI_PASS (optional).
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PI_HOST = os.environ.get("PI_HOST", "vidtest@192.168.8.160")
PI_DIR = os.environ.get("PI_DIR", "/opt/pi-fpv-companion")
SERVICE = os.environ.get("SERVICE", "pi-fpv-companion")
PI_PASS = os.environ.get("PI_PASS")
SRC = Path(__file__).resolve().parent.parent
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=12"]
EXCLUDES = [".git", ".venv", "__pycache__", ".pytest_cache", "*.pyc", "*.egg-info",
            "var", ".DS_Store", "tmp"]

_USE_SSHPASS = False   # decided in choose_auth()


# Colour only on a real terminal that supports ANSI (Mac/Linux, or Windows
# Terminal); plain text when piped or on legacy Windows console.
_COLOR = (sys.stdout.isatty() and "NO_COLOR" not in os.environ
          and (os.name != "nt" or "WT_SESSION" in os.environ))
def _w(code, s): return f"\033[{code}m{s}\033[0m" if _COLOR else s
def info(msg): print(f"{_w('1;36', '==>')} {msg}")
def good(msg): print(f"{_w('1;32', '  OK')} {msg}")
def die(msg): print(f"{_w('1;31', 'ERROR:')} {msg}", file=sys.stderr); sys.exit(1)


def _ssh_prefix() -> list[str]:
    if _USE_SSHPASS:
        return ["sshpass", "-p", PI_PASS, "ssh", *SSH_OPTS, PI_HOST]
    return ["ssh", *SSH_OPTS, PI_HOST]


def rsh(remote_cmd: str, check=True, capture=False) -> subprocess.CompletedProcess:
    """Run a command on the Pi."""
    return subprocess.run(_ssh_prefix() + [remote_cmd], check=check,
                          text=True, capture_output=capture)


def rsu(remote_cmd: str, check=True, capture=False) -> subprocess.CompletedProcess:
    """Run a command on the Pi via sudo (feeds PI_PASS to `sudo -S`; needs NOPASSWD
    if PI_PASS is unset)."""
    return subprocess.run(_ssh_prefix() + [f"sudo -S -p '' {remote_cmd}"],
                          input=(PI_PASS + "\n") if PI_PASS else "",
                          check=check, text=True, capture_output=capture)


def choose_auth():
    """SSH keys if they work; else sshpass (unix + PI_PASS); else interactive ssh."""
    global _USE_SSHPASS, PI_PASS
    keys_ok = subprocess.run(["ssh", "-o", "BatchMode=yes", *SSH_OPTS, PI_HOST, "true"],
                             capture_output=True).returncode == 0
    if keys_ok:
        return
    if PI_PASS and shutil.which("sshpass"):
        _USE_SSHPASS = True
    elif PI_PASS and not shutil.which("sshpass"):
        info("PI_PASS set but sshpass not found — ssh will prompt (or run `deploy.py keys`).")
    # else: no keys, no PI_PASS -> ssh will prompt interactively (works everywhere).


def reachable() -> bool:
    return rsh("true", check=False).returncode == 0


def sync(dry=False):
    info(f"Sync  {SRC}{os.sep}  ->  {PI_HOST}:{PI_DIR}/")
    if shutil.which("rsync"):
        rsh_e = " ".join((["sshpass", "-p", PI_PASS] if _USE_SSHPASS else []) + ["ssh", *SSH_OPTS])
        cmd = ["rsync", "-az", "--human-readable"]
        if dry:
            cmd.append("--dry-run")
        for e in EXCLUDES:
            cmd += ["--exclude", e]
        cmd += ["-e", rsh_e, f"{SRC}{os.sep}", f"{PI_HOST}:{PI_DIR}/"]
        subprocess.run(cmd, check=True)
        return
    # No rsync (typical on native Windows): stream a tar over ssh. Full copy each
    # time, but the tree is tiny and it never deletes Pi-only files.
    info("rsync not found — using tar-over-ssh (cross-platform fallback)")
    if dry:
        good("dry-run: would tar the working tree (excluding venv/models/var) to the Pi")
        return
    tar = ["tar", "czf", "-"]
    for e in EXCLUDES:
        tar += ["--exclude", e]
    tar += ["-C", str(SRC), "."]
    p1 = subprocess.Popen(tar, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(_ssh_prefix() + [f"mkdir -p {PI_DIR} && tar xzf - -C {PI_DIR}"],
                          stdin=p1.stdout)
    p1.stdout.close()
    if p2.wait() != 0 or p1.wait() != 0:
        die("tar-over-ssh transfer failed")


def verify():
    info("Verify service")
    active = rsh(f"systemctl is-active {SERVICE}", check=False, capture=True).stdout.strip()
    nr = rsh(f"systemctl show -p NRestarts --value {SERVICE}", check=False, capture=True).stdout.strip()
    print(f"  service: active={active} NRestarts={nr}")
    if rsh(f"cd {PI_DIR} && .venv/bin/python -m compileall -q src", check=False).returncode == 0:
        good("remote code compiles")
    else:
        die("remote code does not compile")
    logs = rsu(f"journalctl -u {SERVICE} --since '-90 sec' --no-pager", check=False, capture=True).stdout
    errs = sum(1 for ln in logs.splitlines() if any(k in ln.lower() for k in ("error", "traceback", "exception")))
    print(f"  recent error lines (90s): {errs}")
    if active != "active":
        die("service is not active")
    good("deploy healthy")


def main():
    ap = argparse.ArgumentParser(description="Cross-platform deploy to the Pi.")
    ap.add_argument("action", nargs="?", default="deploy",
                    choices=["deploy", "setup", "keys", "status", "logs", "restart", "stop"])
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--venv", action="store_true", help="reinstall the venv (after dep changes)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.action == "keys":
        return do_keys()

    choose_auth()
    if args.action in ("deploy", "setup") and not reachable():
        die(f"cannot reach {PI_HOST} (is the Pi on the network? wrong PI_HOST? auth?)")

    if args.action == "deploy":
        sync(dry=args.dry_run)
        if args.dry_run:
            return info("dry-run only — nothing changed")
        if args.venv:
            info("Update venv"); rsh(f"cd {PI_DIR} && .venv/bin/pip install -q -e ."); good("venv updated")
        if not args.no_restart:
            info(f"Restart {SERVICE}"); rsu(f"systemctl restart {SERVICE}"); good("restarted")
        verify()
    elif args.action == "setup":
        sync()
        info("Running install-pi.sh on the Pi (interactive)")
        ssh_t = (["sshpass", "-p", PI_PASS] if _USE_SSHPASS else []) + ["ssh", "-t", *SSH_OPTS, PI_HOST]
        subprocess.run(ssh_t + [f"cd {PI_DIR} && bash scripts/install-pi.sh"])
    elif args.action == "status":
        rsh(f"systemctl status {SERVICE} --no-pager", check=False)
    elif args.action == "logs":
        rsu(f"journalctl -u {SERVICE} -n 60 -f --no-pager", check=False)
    elif args.action == "restart":
        info(f"Restart {SERVICE}"); rsu(f"systemctl restart {SERVICE}"); good("restarted"); verify()
    elif args.action == "stop":
        info(f"Stop {SERVICE}"); rsu(f"systemctl stop {SERVICE}"); good("stopped")


def do_keys():
    """Install this host's SSH public key on the Pi for passwordless deploys."""
    pub = next((p for p in (Path.home() / ".ssh" / "id_ed25519.pub",
                            Path.home() / ".ssh" / "id_rsa.pub") if p.exists()), None)
    if pub is None:
        die("no SSH key found — create one first:  ssh-keygen -t ed25519")
    key = pub.read_text()
    info(f"Installing {pub.name} on {PI_HOST} (you may be prompted for the password once)")
    prefix = (["sshpass", "-p", PI_PASS] if (PI_PASS and shutil.which("sshpass")) else []) + ["ssh", *SSH_OPTS, PI_HOST]
    r = subprocess.run(prefix + ["umask 077; mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"],
                       input=key, text=True)
    if r.returncode == 0:
        good("key installed — future deploys are passwordless")
    else:
        die("failed to install key")


if __name__ == "__main__":
    main()
