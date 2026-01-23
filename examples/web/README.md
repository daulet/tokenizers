# Tokenizers WASM Web Example

A minimal web application demonstrating the tokenizers library running entirely in the browser via WebAssembly.

## Prerequisites

- [Rust](https://rustup.rs/)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

Install wasm-pack:
```bash
cargo install wasm-pack
```

## Building

From the repository root:

```bash
make build-wasm
```

Or manually:

```bash
cd crates/tokenizers-wasm
wasm-pack build --target web --out-dir ../../examples/web/pkg
```

## Running

Serve the `examples/web` directory with any HTTP server:

```bash
# From repository root
make serve-web

# Or manually
cd examples/web
python3 -m http.server 8080
```

Then open http://localhost:8080 in your browser.

## Usage

1. **Load a tokenizer**: Enter a URL to a HuggingFace `tokenizer.json` file or select a local file
   - Example URL: `https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json`
   - Or download a tokenizer.json and load it from disk

2. **Encode text**: Enter text and click "Encode" to get token IDs
   - Toggle "Add special tokens" to include/exclude [CLS], [SEP], etc.
   - Toggle "Show tokens" to see the string representation of each token
   - Toggle "Show offsets" to see character offsets for each token

3. **Decode token IDs**: Enter comma or space-separated token IDs to decode back to text
   - Toggle "Skip special tokens" to exclude special tokens from output

## Supported Tokenizers

Any HuggingFace tokenizer in JSON format is supported. Common examples:

- `bert-base-uncased`
- `gpt2`
- `roberta-base`
- `distilbert-base-uncased`
- Any model on HuggingFace Hub with a `tokenizer.json` file

## Notes

- All processing happens client-side - no data is sent to any server
- The WASM module is loaded once and cached by the browser
- Large tokenizer files may take a moment to load and parse
