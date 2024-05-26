use std::sync::Arc;

use burn::{config::Config, data::dataloader::batcher::Batcher, module::Module, record::{CompactRecorder, Recorder}, tensor::backend::Backend};

use crate::{data::{BertCasedTokenizer, TextClassificationBatcher, Tokenizer}, model::TextClassificationModelConfig, training::TextClassificationExperimentConfig, TextClassificationDataset};


// Define inference function
pub fn infer<B: Backend, D: TextClassificationDataset + 'static>(
    device: B::Device,      // Device on which to perform computation (e.g. CPU/GPU/CUDA etc)
    artifact_dir: &str,     // Directory containing the model artifact
    samples: Vec<String>,   // Text samples for inference
) {
    // Load experiment configuration
    let config = TextClassificationExperimentConfig::load(format!("{}/config.json", artifact_dir).as_str())
        .expect("Config file present");

    // Initialize tokenizer
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    // Get the number of classes and vocabulary size
    let num_classes = D::num_classes();

    // Initialize model
    let batcher = Arc::new(TextClassificationBatcher::<B>::new(
        tokenizer.clone(), 
        device.clone(), 
        config.max_seq_length,
    ));

    // Load pre-trained weights
    println!("Loading model weights...");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model weights");

    // Create model from the loaded weights
    let model = TextClassificationModelConfig::new(
        config.transformer, 
        num_classes, 
        tokenizer.vocab_size(), 
        config.max_seq_length,
    )
    .init(&device)
    .load_record(record); // Load the record into the model

    // Perform inference on the samples
    println!("Performing inference...");
    let item = batcher.batch(samples.clone()); // Batch the samples using the batcher
    let predictions = model.infer(item); // Get the model predictions

    // Print the predictions for each sample
    for (i, text) in samples.into_iter().enumerate() {
        let prediction = predictions.clone().slice([i..i+1]); // Get the prediction for the current sample
        let logits = prediction.to_data(); // Convert the prediction tensor to data
        let class_index = prediction.argmax(1).into_data().convert::<i32>().value[0]; // Get the class index with the highest probability
        let class = D::class_name(class_index as usize); // Get the class name from the class index

        // Print sample text, predicted class, and logits
        println!("\n=== Item {i} ===\n- Text: {text}\n- Logits: {logits}\n- Prediction: {class}\n================");
    }
}