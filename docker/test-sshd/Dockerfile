# Throwaway SSH target that mimics the Pi enough to test scripts/deploy.py
# end-to-end on a dev host — ssh + rsync/tar sync + sudo + a STUBBED systemd
# service (so restart/verify work without real systemd) + a .venv whose python is
# real (so the compileall verify step runs). NOT for production — test only.
#
#   docker build -t pifpv-testsshd docker/test-sshd
#   docker run -d --rm --name pifpv-testsshd -p 127.0.0.1:22:22 pifpv-testsshd
#   PI_HOST=vidtest@127.0.0.1 PI_PASS=password python scripts/deploy.py
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server rsync python3 sudo ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# user matches deploy.py defaults: vidtest / password, with (password) sudo
RUN useradd -m -s /bin/bash vidtest && echo 'vidtest:password' | chpasswd && adduser vidtest sudo
COPY systemctl-stub /usr/local/bin/systemctl
COPY journalctl-stub /usr/local/bin/journalctl
RUN chmod +x /usr/local/bin/systemctl /usr/local/bin/journalctl \
    && mkdir -p /opt/pi-fpv-companion/.venv/bin /run/sshd \
    && ln -s /usr/bin/python3 /opt/pi-fpv-companion/.venv/bin/python \
    && chown -R vidtest:vidtest /opt/pi-fpv-companion
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
