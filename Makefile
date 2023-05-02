build:
	@cd lib && cargo build --release
	@cp lib/target/release/libtokenizers.a .
	@go build .

build-example:
	@docker build -f ./example/Dockerfile . -t tokenizers-example

release-linux-%:
	docker buildx build --platform linux/$* -f example/Dockerfile . -t tokenizers.linux-$*
	docker run --name tokenizers.linux-$* tokenizers.linux-$*
	mkdir -p release/linux-$*
	docker cp tokenizers.linux-$*:/workspace/libtokenizers.a release/linux-$*/tokenizers.a
	docker kill tokenizers.linux-$*

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libtokenizers.a lib/target
