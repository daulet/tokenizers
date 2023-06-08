build:
	@cd lib && cargo build --release
	@cp lib/target/release/libtokenizers.a .
	@go build .

build-example:
	@docker build -f ./example/Dockerfile . -t tokenizers-example

release-darwin-arm64:
	cd lib && cargo build --release --target aarch64-apple-darwin
	mkdir -p artifacts/darwin-arm64
	mv lib/target/aarch64-apple-darwin/release/libtokenizers.a libtokenizers-darwin-arm64.a

release-darwin-x86_64:
	cd lib && cargo build --release --target x86_64-apple-darwin
	mkdir -p artifacts/darwin-x86_64
	mv lib/target/x86_64-apple-darwin/release/libtokenizers.a libtokenizers-darwin-x86_64.a

release-linux-%:
	docker buildx build --platform linux/$* -f release/Dockerfile . -t tokenizers.linux-$*
	mkdir -p artifacts/linux-$*
	docker run -v $(PWD)/release/linux-$*:/mnt --entrypoint cp tokenizers.linux-$* /workspace/libtokenizers.a /mnt/libtokenizers.a
	mv libtokenizers.a libtokenizers-linux-$*.a

release: release-darwin-arm64 release-linux-amd64 release-linux-arm64 release-linux-x86_64

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libtokenizers.a lib/target
