
/// Common interface for tokenizers of all kinds. The `Send + Sync` bounds are 
/// required for the tokenizer to be used in parallel contexts.
#[allow(dead_code)]
pub trait Tokenizer: Send + Sync {
    /// Converts a text string into a sequence of tokens.
    fn encode(&self, value: &str) -> Vec<usize>;

    /// Converts a sequence of tokens back into a text string.
    fn decode(&self, tokens: &[usize]) -> String;

    /// Gets the size of the tokenizer's vocabulary.
    fn vocab_size(&self) -> usize;

    /// Gets the token used for padding sequences to a consistent length.
    fn pad_token(&self) -> usize;

    /// Gets the string representation of the padding token.
    /// The default implementation uses `decode` on the padding token.
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
}

/// A tokenizer that uses the BERT cased tokenizer from the Hugging Face tokenizers library.
pub struct BertCasedTokenizer {
    /// The underlying Hugging Face tokenizer.
    tokenizer: tokenizers::Tokenizer,
}

// Default implementation for the BERT cased tokenizer.
// This implementation uses the `bert-base-cased` model from the Hugging Face model hub.
impl Default for BertCasedTokenizer {
    fn default() -> Self {
        let tokenizer = tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        Self { tokenizer }
    }
}

// Implementation of the `Tokenizer` trait for the BERT cased tokenizer.
impl Tokenizer for BertCasedTokenizer {
    fn encode(&self, value: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(value, true).unwrap();
        tokens.get_ids().into_iter().map(|&x| x as usize).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let tokens: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
        self.tokenizer.decode(&tokens, true).unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }
}