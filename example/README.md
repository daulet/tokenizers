# Example

```
# Keep this version in sync with go module release
curl -fsSL https://github.com/daulet/tokenizers/releases/download/v0.9.0/libtokenizers.darwin-aarch64.tar.gz | tar xvz
# change -L argument to where you've placed the library download above
go run -ldflags="-extldflags '-L$(pwd)'" main.go
```