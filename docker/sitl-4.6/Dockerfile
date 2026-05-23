# ArduCopter 4.6 SITL — for validating the GPS-denied ALT_HOLD RC-override path
# on a build close to the likely flight firmware (EKF3 default, current ALT_HOLD /
# RC-override behaviour). Builds NATIVE arm64 on Apple Silicon (no emulation), and
# runs headless with NO MAVProxy so raw MAVLink is free on TCP 5760 — exactly what
# scripts/validate_sitl.py / fly_sitl.py / probe_nogps_modes.py connect to.
#
#   docker build --platform linux/arm64 -t pifpv-sitl:4.6 docker/sitl-4.6
#   docker run -d --rm --name pifpv-sitl -p 5760:5760 pifpv-sitl:4.6
#   .venv/bin/python scripts/validate_sitl.py --connect tcp:127.0.0.1:5760
FROM ubuntu:22.04
ARG COPTER_TAG=Copter-4.6.3
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        git sudo lsb-release tzdata ca-certificates && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ArduPilot/ardupilot.git /ardupilot
WORKDIR /ardupilot
RUN git checkout ${COPTER_TAG} && git submodule update --init --recursive

# SITL build deps (minimal — no ARM toolchain). Installed as root; waf allows
# root, unlike Tools/environment_install/install-prereqs-ubuntu.sh which refuses
# it. empy MUST be 3.3.x (waf is incompatible with empy 4.x).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ccache g++ gawk make wget rsync \
        python3 python3-dev python3-pip python3-setuptools python3-wheel \
        libtool libxml2-dev libxslt1-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir "empy==3.3.4" future pexpect dronecan pymavlink

# Build SITL copter (native board).
RUN ./waf configure --board sitl && ./waf copter

EXPOSE 5760/tcp

# Headless, no MAVProxy -> raw MAVLink on tcp:0.0.0.0:5760 (one client = our script).
CMD ["/bin/bash","-lc","/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter -I0 --frame + --speedup 1 -w --no-rebuild --no-mavproxy --custom-location=42.3898,-71.1476,14,270"]
