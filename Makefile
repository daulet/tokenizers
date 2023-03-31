build:
	@cd lib/tokenizer && cargo build --release
	@cp lib/tokenizer/target/release/libtokenizer.a lib/
	@go build .

test: build
	@go test -v ./... -count=1

clean:
	rm -rf lib/libtokenizer.a lib/tokenizer/target
