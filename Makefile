BIN := target/release/laras

.PHONY: all build test lint fmt run clean docker models cek-model

all: build

build:
	cargo build --release

test:
	cargo test --all

lint:
	cargo clippy --all-targets -- -D warnings
	ruff check worker.py norm_id.py || true

fmt:
	cargo fmt --all

run: build
	./$(BIN)

models:
	scripts/unduh-model.sh

cek-model:
	scripts/unduh-model.sh --cek

docker:
	docker compose up -d

clean:
	cargo clean
	rm -rf out cache
