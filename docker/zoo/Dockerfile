FROM rust:bookworm AS rust-build

RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/r2l
COPY . .
RUN cargo build --locked --release -p r2l-zoo-evaluator


FROM python:3.11-slim-bookworm AS python-deps

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential swig \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        "gymnasium[classic-control]==1.2.2" \
        "box2d-py==2.3.5" \
        "popgym==1.0.7"


FROM python:3.11-slim-bookworm

COPY --from=python-deps /opt/venv /opt/venv
COPY --from=rust-build /opt/r2l/target/release/r2l-zoo-evaluator /usr/local/bin/
COPY --from=rust-build /opt/r2l/assets/ppo.yaml /opt/r2l/assets/ppo.yaml

ENV PATH=/opt/venv/bin:$PATH

RUN mkdir -p /opt/r2l/crates/r2l-zoo-evaluator /opt/r2l/logs

ENTRYPOINT ["r2l-zoo-evaluator"]
