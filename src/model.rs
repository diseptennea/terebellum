use burn::{config::Config, module::Module, nn::{loss::CrossEntropyLossConfig, transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput}, Embedding, EmbeddingConfig, Linear, LinearConfig}, tensor::{activation::softmax, backend::{AutodiffBackend, Backend}, Tensor}, train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep}};

use crate::data::{TextClassificationInferenceBatch, TextClassificationTrainingBatch};


/// Configuration for a text classification model.
#[derive(Config)]
pub struct TextClassificationModelConfig {
    transformer: TransformerEncoderConfig,
    num_classes: usize,
    vocab_size: usize,
    max_seq_length: usize,
}

/// A text classification model.
/// This model uses a transformer encoder to process tokenized text and a 
/// linear layer for classification.
#[derive(Module, Debug)]
pub struct TextClassificationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    num_classes: usize,
    max_seq_length: usize,
}

// Define functions for model initialization
impl TextClassificationModelConfig {
    /// Initializes a new text classification model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextClassificationModel<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.num_classes).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

        TextClassificationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            num_classes: self.num_classes,
            max_seq_length: self.max_seq_length,
        }
    }
}

/// The model implementation.
impl<B: Backend> TextClassificationModel<B> {
    /// Forward pass of the model for training.
    pub fn forward(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        // Get batch and sequence length, and the device
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);
        let mask_padding = item.mask_padding.to_device(device);

        // Calculate token and position embeddings, and combine them
        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_tokens + embedding_positions) / 2;

        // Perform transformer encoder forward pass, calculate output and loss
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_padding));
        let output = self.output.forward(encoded);

        let output_classification = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.num_classes]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());

        // Return the output and loss
        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }
    }

    /// Forward pass of the model for inference.
    pub fn infer(&self, item: TextClassificationInferenceBatch<B>) -> Tensor<B, 2> {
        // Get batch and sequence length, and the device
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let mask_padding = item.mask_padding.to_device(device);

        // Calculate token and position embeddings, and combine them
        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_tokens + embedding_positions) / 2;

        // Perform transformer encoder forward pass and calculate output
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_padding));
        let output = self.output.forward(encoded);
        let output = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.num_classes]);

        softmax(output, 1)
    }
}

/// Implementation of the `TrainStep` trait for the text classification model.
impl<B: AutodiffBackend> TrainStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
    for TextClassificationModel<B>
{
    fn step(&self, item: TextClassificationTrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Run forward pass, calculate gradients and update parameters
        let item = self.forward(item);
        let grads = item.loss.backward();

        // Return the output and loss
        TrainOutput::new(self, grads, item)
    }
}

/// Validation function for the text classification model.
impl<B: Backend> ValidStep<TextClassificationTrainingBatch<B>, ClassificationOutput<B>>
    for TextClassificationModel<B>
{
    fn step(&self, item: TextClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        // Run forward pass and return the output
        self.forward(item)
    }
}