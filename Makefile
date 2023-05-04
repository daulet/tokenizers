build:
	@cd lib && cargo build --release
	@cp lib/target/release/libtokenizers.a .
	@go build .

build-example:
	@docker build -f ./example/Dockerfile . -t tokenizers-example

release-darwin-arm64:
	cd lib && cargo build --release --target aarch64-apple-darwin
	mkdir -p release/darwin-arm64
	cp lib/target/aarch64-apple-darwin/release/libtokenizers.a release/darwin-arm64/libtokenizers.a
	cd release/darwin-arm64 && \
		tar -czf libtokenizers.darwin-arm64.tar.gz libtokenizers.a
	mkdir -p release/artifacts
	cp release/darwin-arm64/libtokenizers.darwin-arm64.tar.gz release/artifacts/libtokenizers.darwin-arm64.tar.gz

release-linux-%:
	docker buildx build --platform linux/$* -f example/Dockerfile . -t tokenizers.linux-$*
	mkdir -p release/linux-$*
	docker run -v $(PWD)/release/linux-$*:/mnt --entrypoint cp tokenizers.linux-$* /workspace/libtokenizers.a /mnt/libtokenizers.a
	cd release/linux-$* && \
		tar -czf libtokenizers.linux-$*.tar.gz libtokenizers.a
	mkdir -p release/artifacts
	cp release/linux-$*/libtokenizers.linux-$*.tar.gz release/artifacts/libtokenizers.linux-$*.tar.gz

release: release-darwin-arm64 release-linux-amd64 release-linux-arm64 release-linux-x86_64

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libtokenizers.a lib/target
