FROM python:3.12.13-slim-bookworm@sha256:93ab4b7fa528b25124c97bcc755415e60eb671a86b4dbe0328df2fe2d1c1193d AS python
FROM rust:1.83.0-slim-bookworm@sha256:540c902e99c384163b688bbd8b5b8520e94e7731b27f7bd0eaa56ae1960627ab

ENV CARGO_HOME=/usr/local/cargo \
    LD_LIBRARY_PATH=/usr/local/lib \
    PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUSTUP_HOME=/usr/local/rustup

COPY --from=python /usr/local /usr/local
COPY requirements/container-build-constraints.txt /tmp/container-build-constraints.txt

WORKDIR /build-src
COPY . /build-src

RUN python --version \
    && rustc --version \
    && cargo --version \
    && python -m pip install --constraint /tmp/container-build-constraints.txt maturin==1.13.1 plox==0.0.3 rdkit==2026.3.1 \
    && python -m maturin build --release --out /tmp/grimace-dist -i python3.12 \
    && python -m pip install --no-deps /tmp/grimace-dist/*.whl \
    && rm -rf /tmp/grimace-dist

WORKDIR /build-src
USER 65532:65532

ENTRYPOINT ["python"]
CMD ["-m", "unittest", "discover", "-s", "tests/perf", "-t", ".", "-q"]
