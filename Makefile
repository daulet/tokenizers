build:
	@cd lib && cargo build --release
	@cp lib/target/release/libtokenizers.a .
	@go build .


build-example:
	@docker build -f ./example/Dockerfile . -t tokenizers-example

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libtokenizers.a lib/target
