build:
	@cargo build --release -p tokenizers-ffi
	@cp target/release/libtokenizers_ffi.a ./libtokenizers.a
	@go build .

build-example-go:
	@docker build -f ./examples/go/Dockerfile . -t tokenizers-example

release-darwin-%: test
	cargo build --release -p tokenizers-ffi --target $*-apple-darwin
	mkdir -p artifacts/darwin-$*
	cp target/$*-apple-darwin/release/libtokenizers_ffi.a artifacts/darwin-$*/libtokenizers.a
	cd artifacts/darwin-$* && \
		tar -czf libtokenizers.darwin-$*.tar.gz libtokenizers.a
	mkdir -p artifacts/all
	cp artifacts/darwin-$*/libtokenizers.darwin-$*.tar.gz artifacts/all/libtokenizers.darwin-$*.tar.gz

release-linux-%: test
	docker buildx build --platform linux/$* --build-arg="DOCKER_TARGETPLATFORM=linux/$*" -f release/Dockerfile . -t tokenizers.linux-$*
	mkdir -p artifacts/linux-$*
	docker run -v $(PWD)/artifacts/linux-$*:/mnt --entrypoint ls tokenizers.linux-$* /workspace/tokenizers/lib/linux
	docker run -v $(PWD)/artifacts/linux-$*:/mnt --entrypoint cp tokenizers.linux-$* /workspace/tokenizers/lib/linux/$*/libtokenizers.a /mnt/libtokenizers.a
	cd artifacts/linux-$* && \
		tar -czf libtokenizers.linux-$*.tar.gz libtokenizers.a
	mkdir -p artifacts/all
	cp artifacts/linux-$*/libtokenizers.linux-$*.tar.gz artifacts/all/libtokenizers.linux-$*.tar.gz

release: release-darwin-aarch64 release-darwin-x86_64 release-linux-arm64 release-linux-x86_64 release-linux-s390x release-linux-ppc64le
	cp artifacts/all/libtokenizers.darwin-aarch64.tar.gz artifacts/all/libtokenizers.darwin-arm64.tar.gz
	cp artifacts/all/libtokenizers.linux-arm64.tar.gz artifacts/all/libtokenizers.linux-aarch64.tar.gz
	cp artifacts/all/libtokenizers.linux-x86_64.tar.gz artifacts/all/libtokenizers.linux-amd64.tar.gz

test: build
	@go test -ldflags="-extldflags '-L./'" -v ./... -count=1

clean:
	rm -rf libtokenizers.a target artifacts

bazel-sync:
	CARGO_BAZEL_REPIN=1 bazel sync --only=crate_index

build-wasm:
	@command -v wasm-bindgen >/dev/null 2>&1 || PATH="$$HOME/.cargo/bin:$$PATH" cargo install wasm-bindgen-cli --version 0.2.100
	cd crates/tokenizers-wasm && PATH="$$HOME/.cargo/bin:$$PATH" RUSTFLAGS='--cfg=getrandom_backend="wasm_js"' wasm-pack build --target web --out-dir ../../examples/web/pkg --mode no-install

build-docs: build-wasm
	@mkdir -p docs/pkg
	@cp examples/web/pkg/* docs/pkg/

serve-web: build-wasm
	cd examples/web && python3 -m http.server 8080

serve-docs: build-docs
	cd docs && python3 -m http.server 8080
