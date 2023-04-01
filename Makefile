build:
	@cd lib && cargo build --release
	@cp lib/target/release/libtokenizers.a .
	@go build .

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libtokenizers.a lib/target
