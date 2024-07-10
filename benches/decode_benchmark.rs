use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::time::Instant;
use tokenizers::tokenizer::Tokenizer;

fn decode(tokenizer:&Tokenizer, ids_slice: &[u32], skip_special_tokens: bool) -> String {
    tokenizer.decode(ids_slice, skip_special_tokens).expect("failed to decode input")
}

fn bench_decode_n_times(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_file("./test/data/bert-base-uncased.json").expect("failed to create tokenizer");
    c.bench_function("decode_n_times", 
        |b| b.iter(||
            decode(&tokenizer, black_box(&[2829, 4419, 14523, 2058, 1996, 13971, 3899]), black_box(true))
        )
    );
}

fn bench_decode_n_tokens(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_file("./test/data/bert-base-uncased.json").expect("failed to create tokenizer");
    let max_token_id = tokenizer.get_vocab_size(true);
    let mut rng = rand::thread_rng();

    c.bench_function("decode_n_tokens",
        move |b| { b.iter_custom(|iters| {
            let tokens: Vec<u32> = (0..iters).map(|_| rng.gen_range(0..max_token_id) as u32).collect();

            let start = Instant::now();
            decode(&tokenizer, black_box(&tokens), black_box(true));
            start.elapsed()
        })}
    );
}

criterion_group!(benches, bench_decode_n_times, bench_decode_n_tokens);
criterion_main!(benches);
