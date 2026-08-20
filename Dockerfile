FROM rust:bookworm AS builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cargo build --locked --release --package r2l-zoo-evaluator

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"

RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir "gymnasium[box2d]" popgym vizdoom

WORKDIR /app
COPY assets ./assets
COPY --from=builder /app/target/release/r2l-zoo-evaluator /usr/local/bin/

ENTRYPOINT ["r2l-zoo-evaluator"]
