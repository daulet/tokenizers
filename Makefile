build:
	@cd lib && cargo build --release
	@cp lib/target/release/libtokenizers.a .
	@go build .

build-example:
	@docker build -f ./example/Dockerfile . -t tokenizers-example

release-linux-%:
	docker buildx build --platform linux/$* -f example/Dockerfile . -t tokenizers.linux-$*
	mkdir -p release/linux-$*
	docker run -v $(PWD)/release/linux-$*:/mnt --entrypoint cp tokenizers.linux-$* /workspace/libtokenizers.a /mnt/libtokenizers.a
	mkdir -p release/output
	tar -czf release/output/libtokenizers.linux-$*.tar.gz release/linux-$*/libtokenizers.a

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libtokenizers.a lib/target
