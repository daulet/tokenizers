# Example

To run the example you need to obtain the built rust library.
```
# On M1+ Mac (note arch)
curl -fsSL https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.darwin-arm64.tar.gz | tar xvz

# change -L argument to where you've placed the library download above
go run -ldflags="-extldflags '-L$(pwd)'" main.go
```
or `make build` from the parent directory and `go run -ldflags="-extldflags '-L..'" main.go`
