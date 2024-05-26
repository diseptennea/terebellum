use std::sync::Arc;

use burn::{config::Config, data::{dataloader::DataLoaderBuilder, dataset::transform::SamplerDataset}, lr_scheduler::noam::NoamLrSchedulerConfig, module::Module, nn::transformer::TransformerEncoderConfig, optim::AdamWConfig, record::{CompactRecorder, Recorder}, tensor::backend::AutodiffBackend, train::{metric::{AccuracyMetric, LearningRateMetric, LossMetric}, LearnerBuilder}};

use crate::{data::{BertCasedTokenizer, TextClassificationBatcher, Tokenizer}, model::TextClassificationModelConfig, TextClassificationDataset};

/// This module contains the training configuration for the text classification experiment.
#[derive(Config)]
pub struct TextClassificationExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamWConfig,

    #[config(default = 512)]
    pub max_seq_length: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 5)]
    pub num_epochs: usize,
}

// Define train function
pub fn train<B: AutodiffBackend, D: TextClassificationDataset + 'static>(
    devices: Vec<B::Device>,                    // Devices on which to perform computation (e.g. CPU/GPU/CUDA etc)
    dataset_train: D,                           // Training dataset
    dataset_test: D,                            // Test dataset
    config: TextClassificationExperimentConfig, // Experiment configuration
    artifact_dir: &str,                         // Directory to save the model artifact
) {
    // Initialize tokenizer
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    // Initialize batchers for training and testing data
    let batcher_train = TextClassificationBatcher::<B>::new(
        tokenizer.clone(), 
        devices[0].clone(), 
        config.max_seq_length,
    );
    let batcher_test = TextClassificationBatcher::<B::InnerBackend>::new(
        tokenizer.clone(), 
        devices[0].clone(), 
        config.max_seq_length,
    );

    // Initialize model
    let model = TextClassificationModelConfig::new(
        config.transformer.clone(), 
        D::num_classes(), 
        tokenizer.vocab_size(), 
        config.max_seq_length,
    ).init(&devices[0]);

    // Initialize data loaders for training and testing data
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(dataset_train, 35_000));
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(dataset_test, 7_000));

    // Initialize optimizer
    let optimizer = config.optimizer.init();

    // Initialize learning rate scheduler
    let lr_scheduler = NoamLrSchedulerConfig::new(1e-2)
        .with_warmup_steps(1000)
        .with_model_size(config.transformer.d_model)
        .init();

    // Initialize learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(devices)
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optimizer, lr_scheduler);

    // Train the model
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // Save the trained model and configuration
    config.save(format!("{artifact_dir}/config.json")).expect("Should be able to save configuration");
    CompactRecorder::new().record(model_trained.into_record(), format!("{artifact_dir}/model").into())
        .expect("Should be able to save model");
}