# Personality Inference Using MBTI

## 1. Original Source

This project is inspired by various research on personality inference and natural language processing (NLP) techniques. Specifically, we utilize the **GPT-2** model from Hugging Face's transformers library to predict Myers-Briggs Type Indicator (MBTI) types based on user posts.

We created our own source based and worked off by loading in GPT-2 as our base model
GPT-2: https://huggingface.co/openai-community/gpt2

---

## 2. Files Modified and Functions

### `main.py`
This file contains the core implementation for training a personality inference model based on GPT-2.

- **`tokenize_function`**: Tokenizes the input text samples using GPT-2's tokenizer.
- **`preprocess_dataset`**: Processes the MBTI dataset by creating input IDs, attention masks, and labels.
- **`MBTIDataset`**: Custom dataset class that prepares the data for use in the model.
- **`WeightedTrainer`**: A subclass of `Trainer` from Hugging Face to include custom loss function with class weights.
- **`compute_metrics`**: Computes the weighted F1 score as the evaluation metric.
- **Training Loop**: Fine-tunes the GPT-2 model for the MBTI classification task using custom loss and weighted sampling.

### `modelTest.py`
This file is used to test the trained model by making predictions on input text.

- **`reverse_type_mapping`**: Maps predicted class indices back to MBTI types.
- **Prediction Loop**: Tokenizes input text, makes predictions using the fine-tuned model, and maps the predictions to the corresponding MBTI type.

---

## 3. Commands to Train and Test the Model

### To train the model:

1. Clone the repository or download the project files.
2. Install dependencies (listed below).
3. Make sure the dataset `mbti_1.csv` is in the same directory as the scripts or modify the path in `main.py`.
4. Run the following command to start training:
$python main.py

### To test the model:
After training, use the trained model checkpoint (./results/checkpoint-5202) in modelTest.py to make predictions on new input text.
You can obtain checkpoint-5202 (our model) in the google drive: https://drive.google.com/drive/folders/1hvVywslyEQ5HaRpgtWHvvyoPMyqtGPZd?usp=sharing

You can modify the sample text in modelTest.py with any text you want to classify.
Run the following command:
$python modelTest.py

## 4. Trained model

Mentioned previously in 'Testing' as well, the google drive to our trained model is the following link below
Link: https://drive.google.com/drive/folders/1hvVywslyEQ5HaRpgtWHvvyoPMyqtGPZd?usp=sharing

## 5. Prompts

No specific prompts were used during testing. However, the input text for classification is expected to be a short piece of text, such as a social media post, that describes a person's personality, behavior, or emotions.
Tested using dataset line prompts or utlizied "seemingly" extroverted input lines like the following:
"What? High school was great. I was able to sleep all during class all day and keep to myself and just do what I want"

## 6. Requirements

Python 3.8+
PyTorch 1.9+
Transformers 4.0+ (Hugging Face library)
Scikit-learn 0.24+
NumPy 1.19+

REQUIRED DEPENDENCIES:
pip install torch transformers scikit-learn numpy pandas