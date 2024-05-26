use std::sync::Arc;

use burn::{data::dataloader::batcher::Batcher, nn::attention::generate_padding_mask, tensor::{backend::Backend, Bool, Data, ElementConversion, Int, Tensor}};
use derive_new::new;

use super::{dataset::TextClassificationItem, tokenizer::Tokenizer};

/// A batcher for text classification tasks.
#[derive(Clone, new)]
pub struct TextClassificationBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>,  // Tokenizer for converting text to token IDs.
    device: B::Device,              // Device on which to perform computation (e.g. CPU/GPU/CUDA etc)
    max_seq_length: usize,          // Maximum sequence length for tokenized text
}

/// A structure for a training batch for text classification tasks.
#[derive(Debug, Clone)]
pub struct TextClassificationTrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,          // The input IDs for the batch.
    pub labels: Tensor<B, 1, Int>,          // The labels for the batch.
    pub mask_padding: Tensor<B, 2, Bool>,   // The padding mask for the tokenized text.
}

/// A structure for an inference batch for text classification tasks.
#[derive(Debug, Clone)]
pub struct TextClassificationInferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,          // Tokenized text.
    pub mask_padding: Tensor<B, 2, Bool>,   // The padding mask for the tokenized text.
}

impl<B: Backend> Batcher<TextClassificationItem, TextClassificationTrainingBatch<B>>
    for TextClassificationBatcher<B> 
{
    /// Batches a vector of text classification items into a training batch.
    fn batch(&self, items: Vec<TextClassificationItem>) -> TextClassificationTrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        let mut labels_list = Vec::with_capacity(items.len());

        // Tokenize text and create label tensor for each item
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text));
            labels_list.push(Tensor::from_data(
                Data::from([(item.label as i64).elem()]), 
                &self.device,
            ));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(), 
            tokens_list, 
            Some(self.max_seq_length), 
            &self.device,
        );

        // Create and return training batch
        TextClassificationTrainingBatch {
            tokens: mask.tensor,
            labels: Tensor::cat(labels_list, 0),
            mask_padding: mask.mask,
        }
    }
}

/// Implementation of the `Batcher` trait for text classification inference batches.
impl<B: Backend> Batcher<String, TextClassificationInferenceBatch<B>>
    for TextClassificationBatcher<B>
{
    /// Batches a vector of text strings into an inference batch.
    fn batch(&self, items: Vec<String>) -> TextClassificationInferenceBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        // Tokenize each string
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(), 
            tokens_list, 
            Some(self.max_seq_length), 
            &self.device,
        );

        // Create and return inference batch
        TextClassificationInferenceBatch {
            tokens: mask.tensor.to_device(&self.device),
            mask_padding: mask.mask.to_device(&self.device),
        }
    }
}