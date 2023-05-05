build:
	@cd lib && cargo build --release
	@cp lib/target/release/libtokenizers.a .
	@go build .

build-example:
	@docker build -f ./example/Dockerfile . -t tokenizers-example

release-darwin-arm64:
	cd lib && cargo build --release --target aarch64-apple-darwin
	mkdir -p artifacts/darwin-arm64
	cp lib/target/aarch64-apple-darwin/release/libtokenizers.a artifacts/darwin-arm64/libtokenizers.a
	cd artifacts/darwin-arm64 && \
		tar -czf libtokenizers.darwin-arm64.tar.gz libtokenizers.a
	mkdir -p artifacts/all
	cp artifacts/darwin-arm64/libtokenizers.darwin-arm64.tar.gz artifacts/all/libtokenizers.darwin-arm64.tar.gz

release-linux-%:
	docker buildx build --platform linux/$* -f example/Dockerfile . -t tokenizers.linux-$*
	mkdir -p artifacts/linux-$*
	docker run -v $(PWD)/release/linux-$*:/mnt --entrypoint cp tokenizers.linux-$* /workspace/libtokenizers.a /mnt/libtokenizers.a
	cd artifacts/linux-$* && \
		tar -czf libtokenizers.linux-$*.tar.gz libtokenizers.a
	mkdir -p artifacts/all
	cp artifacts/linux-$*/libtokenizers.linux-$*.tar.gz artifacts/all/libtokenizers.linux-$*.tar.gz

release: release-darwin-arm64 release-linux-amd64 release-linux-arm64 release-linux-x86_64

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libtokenizers.a lib/target
