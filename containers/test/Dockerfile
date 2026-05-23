FROM python:3.12.13-slim-bookworm@sha256:93ab4b7fa528b25124c97bcc755415e60eb671a86b4dbe0328df2fe2d1c1193d AS python
FROM rust:1.83.0-slim-bookworm@sha256:540c902e99c384163b688bbd8b5b8520e94e7731b27f7bd0eaa56ae1960627ab

ENV CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUSTUP_HOME=/usr/local/rustup

COPY --from=python /usr/local /usr/local

RUN python --version \
    && rustc --version \
    && cargo --version \
    && python -m pip install --no-cache-dir maturin==1.13.1 rdkit==2026.3.1

WORKDIR /src
COPY . /src

RUN python -m maturin build --release --out /tmp/grimace-dist -i python3.12 \
    && python -m pip install --no-cache-dir /tmp/grimace-dist/*.whl \
    && rm -rf /tmp/grimace-dist

USER 65532:65532

ENTRYPOINT ["python"]
CMD ["-m", "unittest", "tests.run_installed_package_correctness", "-q"]
