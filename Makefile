build:
	@cd lib && cargo build --release
	@cp lib/target/release/libtokenizers.a .
	@go build .

build-example:
	@docker build -f ./example/Dockerfile . -t tokenizers-example

release-darwin-%:
	cd lib && cross build --release --target $*-apple-darwin
	mkdir -p artifacts/darwin-$*
	cp lib/target/$*-apple-darwin/release/libtokenizers.a artifacts/darwin-$*/libtokenizers.a
	cd artifacts/darwin-$* && \
		tar -czf libtokenizers.darwin-$*.tar.gz libtokenizers.a
	mkdir -p artifacts/all
	cp artifacts/darwin-$*/libtokenizers.darwin-$*.tar.gz artifacts/all/libtokenizers.darwin-$*.tar.gz

release-linux-%:
	cd lib && cross build --release --target $*-unknown-linux-gnu
	mkdir -p artifacts/linux-$*
	cp lib/target/$*-unknown-linux-gnu/release/libtokenizers.a artifacts/linux-$*/libtokenizers.a
	cd artifacts/linux-$* && \
		tar -czf libtokenizers.linux-$*.tar.gz libtokenizers.a
	mkdir -p artifacts/all
	cp artifacts/linux-$*/libtokenizers.linux-$*.tar.gz artifacts/all/libtokenizers.linux-$*.tar.gz

release: release-darwin-aarch64 release-darwin-x86_64 release-linux-aarch64 release-linux-x86_64
	cp artifacts/all/libtokenizers.darwin-aarch64.tar.gz artifacts/all/libtokenizers.darwin-arm64.tar.gz
	cp artifacts/all/libtokenizers.linux-aarch64.tar.gz artifacts/all/libtokenizers.linux-arm64.tar.gz
	cp artifacts/all/libtokenizers.linux-x86_64.tar.gz artifacts/all/libtokenizers.linux-amd64.tar.gz

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libtokenizers.a lib/target
