FROM lukemathwalker/cargo-chef:0.1.78-rust-bookworm AS chef

RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/r2l

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS rust-build
COPY --from=planner /opt/r2l/recipe.json recipe.json
RUN cargo chef cook --locked --profile benchmark --package r2l-benchmark-worker --recipe-path recipe.json
COPY . .
RUN cargo build --locked --profile benchmark -p r2l-benchmark-worker


FROM python:3.11-slim-bookworm AS python-deps

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        curl \
        libgl1-mesa-dev \
        libglew-dev \
        libosmesa6-dev \
        patchelf \
        swig \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/mujoco \
    && curl --fail --location \
        https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
        | tar --extract --gzip --directory /opt/mujoco

ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.13.0" \
    && /opt/venv/bin/pip install --no-cache-dir \
        "gymnasium[classic-control,mujoco]==1.2.2" \
        "box2d-py==2.3.5" \
        "gymnasium-robotics[mujoco_py]==1.4.2" \
        "minigrid==3.1.0" \
        "popgym==1.0.7" \
        "pybullet-envs-gymnasium==0.6.0" \
        "PyYAML==6.0.2" \
        "stable-baselines3==2.9.0" \
    && /opt/venv/bin/python -c "import mujoco_py"


FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglfw3 \
        libglew2.2 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-deps /opt/venv /opt/venv
COPY --from=python-deps /opt/mujoco /opt/mujoco
COPY --from=rust-build /opt/r2l/target/benchmark/r2l-benchmark-worker /usr/local/bin/
COPY --from=rust-build /opt/r2l/benchmark/assets/ppo.yaml /opt/r2l/assets/ppo.yaml
COPY --from=rust-build /opt/r2l/benchmark/sb3/ppo.py /opt/r2l/sb3/ppo.py

ENV PATH=/opt/venv/bin:$PATH
ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
ENV LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin

# Third-party Gymnasium environments register themselves when imported. The task
# guard prevents native-library probe subprocesses from repeating these imports.
RUN echo 'import os; os.environ.pop("R2L_TASK", None) and (__import__("gymnasium_robotics"), __import__("minigrid"), __import__("popgym"), __import__("pybullet_envs_gymnasium"))' \
    > /opt/venv/lib/python3.11/site-packages/r2l_benchmark_envs.pth

RUN mkdir -p /opt/r2l/logs

ENTRYPOINT ["r2l-benchmark-worker"]
