# OpenCV dependencies
ARG BUILD_PACKAGES="wget clang libclang-dev libopencv-dev libleptonica-dev libtesseract-dev"

FROM rust:latest as builder

# Install OpenCV dependencies
ARG BUILD_PACKAGES
RUN apt-get update -qq && \
    apt-get install -y ${BUILD_PACKAGES}

# Make a fake Rust app to keep a cached layer of compiled crates
RUN USER=root cargo new app
WORKDIR /usr/src/app
COPY Cargo.toml Cargo.lock ./
# Needs at least a main.rs file with a main function
RUN mkdir src && echo "fn main(){}" > src/main.rs
# Will build all dependent crates in release mode
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/app/target \
    cargo build --release

# Copy the rest
COPY . .
# Build (install) the actual binaries
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/app/target \
    cargo install --path .

# Download tesseract digits training data
RUN wget https://github.com/Shreeshrii/tessdata_shreetest/raw/master/digits.traineddata

# Runtime image
FROM debian:bullseye-slim

# Install OpenCV dependencies
ARG BUILD_PACKAGES
RUN apt-get update -qq && \
    apt-get install -y ${BUILD_PACKAGES}

# Run as "app" user
RUN useradd -ms /bin/bash app

USER app
WORKDIR /app

# Get compiled binaries from builder's cargo install directory
COPY --from=builder /usr/local/cargo/bin/reddit_sudoku_solver /app/reddit_sudoku_solver

# Copy tesseract digits training data
COPY --from=builder /usr/src/app/digits.traineddata /usr/share/tesseract-ocr/4.00/tessdata/digits.traineddata

# No CMD or ENTRYPOINT, see fly.toml with `cmd` override.
