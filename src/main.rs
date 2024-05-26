use burn::{backend::{wgpu::{AutoGraphicsApi, WgpuDevice}, Autodiff, Wgpu}, nn::transformer::TransformerEncoderConfig, optim::AdamWConfig, tensor::backend::AutodiffBackend};
use terebellum::{training::TextClassificationExperimentConfig, TurkishOffensiveLanguageDataset};


fn main() {
    launch::<Autodiff<Wgpu<AutoGraphicsApi, f32, i32>>>(vec![WgpuDevice::default()]);
}

pub fn launch<B: AutodiffBackend>(devices: Vec<B::Device>) {
    let config = TextClassificationExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 16, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamWConfig::new().with_weight_decay(5e-5),
    );

    terebellum::training::train::<B, TurkishOffensiveLanguageDataset>(
        devices, 
        TurkishOffensiveLanguageDataset::train(), 
        TurkishOffensiveLanguageDataset::test(), 
        config, 
        "./artifact/turkish-offensive-language-model",
    );
}