use burn::data::dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset};
use serde::{Deserialize, Serialize};


/// A struct to represent a text classification item.
#[derive(Clone, Debug)]
pub struct TextClassificationItem {
    /// The text content of the item.
    pub text: String,

    /// The label of the item (e.g., a category or sentiment score)
    pub label: usize,
}

impl TextClassificationItem {
    /// Creates a new text classification item.
    pub fn new(text: String, label: usize) -> Self {
        Self { text, label }
    }
}

/// A trait for text classification datasets.
pub trait TextClassificationDataset: Dataset<TextClassificationItem> {
    /// Gets the number of unique classes in the dataset.
    fn num_classes() -> usize;

    /// Returns the name of the class with the given label.
    fn class_name(label: usize) -> String;
}

/// Struct for items in the Turkish Offensive Language Detection dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurkishOffensiveLanguageItem {
    /// The text content of the item.
    pub text: String,

    /// The label of the item (0 for non-offensive, 1 for offensive).
    pub label: usize,
}

/// Struct for the Turkish Offensive Language Detection dataset.
pub struct TurkishOffensiveLanguageDataset {
    /// The underlying SQLite dataset.
    dataset: SqliteDataset<TurkishOffensiveLanguageItem>,
}

// Implement the `Dataset` trait for the Turkish Offensive Language Detection dataset.
impl Dataset<TextClassificationItem> for TurkishOffensiveLanguageDataset {
    /// Returns a specific item from the dataset.
    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset
            .get(index)
            // Map the item to a text classification item
            .map(|item| TextClassificationItem::new(item.text, item.label))
    }

    /// Returns the number of items in the dataset.
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

// Implement methods for constructing the Turkish Offensive Language Detection dataset..
impl TurkishOffensiveLanguageDataset {
    /// Returns the training portion of the dataset.
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Returns the testing portion of the dataset.
    pub fn test() -> Self {
        Self::new("test")
    }

    /// Constructs the dataset from a split (either "train" or "test").
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<TurkishOffensiveLanguageItem> = HuggingfaceDatasetLoader::new("Toygar/turkish-offensive-language-detection")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }
}

/// Implement the `TextClassificationDataset` trait for the Turkish Offensive Language Detection dataset.
impl TextClassificationDataset for TurkishOffensiveLanguageDataset {
    /// Gets the number of unique classes in the dataset.
    fn num_classes() -> usize {
        2
    }

    /// Returns the name of the class with the given label.
    fn class_name(label: usize) -> String {
        match label {
            0 => "Non-offensive".to_string(),
            1 => "Offensive".to_string(),
            _ => panic!("Invalid class label: {}", label),
        }.to_string()
    }
}