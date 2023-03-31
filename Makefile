build:
	@cd lib/tokenizers && cargo build --release
	@cp lib/tokenizers/target/release/libtokenizers.a lib/
	@go build .

test: build
	@go test -v ./... -count=1

clean:
	rm -rf lib/libtokenizers.a lib/tokenizers/target
