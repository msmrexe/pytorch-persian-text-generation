# Persian Text Generation: RNN vs. Transformer

This project, developed for a graduate Generative Models course, implements and compares two neural network architectures for next-word prediction on a Persian Wikipedia dataset. It includes a from-scratch GRU (RNN) model and a from-scratch Encoder-Decoder Transformer model, providing a deep dive into their implementation and performance for language modeling.

## Features

* **From-Scratch RNN:** A clean implementation of a GRU-based recurrent neural network for sequence prediction.
* **From-Scratch Transformer:** A complete, from-scratch implementation of an Encoder-Decoder Transformer, including Multi-Head Attention, Positional Encoding, and Encoder/Decoder stacks.
* **Modular Pipeline:** The project is structured with separate, documented scripts for preprocessing, training, and generation.
* **Robust Preprocessing:** Uses the `hazm` library for expert Persian text normalization, lemmatization, and tokenization, plus a custom vocabulary builder.
* **CLI Controllable:** All scripts (`preprocess.py`, `train.py`, `generate.py`) are controllable via command-line arguments.
* **Advanced Generation:** Implements Top-K sampling during generation to prevent the common issue of repetitive text loops seen with greedy decoding.

## Core Concepts & Techniques

* **Generative Language Modeling:** The fundamental task of predicting the next token ($w_t$) given a sequence of preceding tokens ($w_1, ..., w_{t-1}$).
* **Recurrent Neural Networks (GRU):** Using Gated Recurrent Units (GRUs) to maintain a hidden state that captures sequential information over time.
* **Transformer Architecture:** Implementing the "Attention Is All You Need" paper from scratch.
    * **Scaled Dot-Product Attention:** The core mechanism for relating different tokens in a sequence.
    * **Multi-Head Attention:** Running the attention mechanism in parallel to capture different types of relationships.
    * **Positional Encoding:** Injecting information about token order into the model.
    * **Encoder-Decoder Stacks:** Using the encoder to build a rich representation of the context (n-gram) and the decoder to generate the output token.
* **Perplexity (PPL):** The primary metric used to evaluate the language model, measuring how well it predicts the test data. A lower PPL is better.
* **Persian NLP:** Tackling the challenges of a morphologically rich language using `hazm` for lemmatization and normalization.

---

## How It Works

This project frames the text generation task as an N-gram prediction problem: given `N` context words, predict the `N+1`-th word.

### 1. Data Preprocessing (`scripts/preprocess.py`)

The raw Persian Wikipedia text is cleaned using a multi-step pipeline:
1.  **Load Data:** The raw text and a list of Persian stop words are loaded.
2.  **Noise Removal:** Regex is used to remove all non-Persian characters and digits.
3.  **Tokenization & Stopword Removal:** The text is tokenized, and common stop words (e.g., "از", "در", "که") are removed.
4.  **Normalization & Lemmatization:** The `hazm` library is used to standardize the text (e.g., unify "ها" and "های") and lemmatize words to their root (e.g., "می‌روم" -> "رفت").
5.  **Vocabulary Building:** A vocabulary `vocab.json` is built from the processed tokens, mapping each unique word (with a minimum frequency) to an integer ID. Special tokens `<pad>` (padding) and `<unk>` (unknown) are added.

### 2. Dataloading (`src/dataset.py`)

The processed text is converted into samples for the models:
1.  **N-Gram Creation:** The script slides a window of size `N+1` over the entire text.
2.  **Context/Target Split:** For each window, the first `N` tokens become the `context` (input) and the `N+1`-th token becomes the `target` (label).
3.  **Indexing:** All tokens are converted to their integer IDs from the vocabulary.
4.  **Dataloaders:** The (context, target) pairs are loaded into `TensorDataset` and split into `train`, `validation`, and `test` dataloaders for efficient batching.

### 3. Model Architectures & Comparison

Both models are trained on the same n-gram prediction task, allowing for a direct comparison.

#### Model 1: RNN (GRU) (`src/models/rnn.py`)

* **Architecture:**
    1.  **Embedding Layer:** Converts input token IDs into dense vectors.
    2.  **GRU Layer:** A multi-layer GRU processes the sequence of `N` embedded tokens. It updates its hidden state at each step, capturing a summary of the context.
    3.  **Output Layer:** The final hidden state from the last token is passed through a linear layer to produce logits over the entire vocabulary.
* **How it Works:** The GRU's strength is its simplicity and effectiveness at capturing sequential dependencies. The hidden state acts as the model's "memory" of the context. However, for very long contexts (much larger than N=2 or 3), it can struggle to remember information from the beginning (vanishing gradient problem).

#### Model 2: Transformer (From-Scratch) (`src/models/transformer.py`)

* **Architecture:** This model is implemented as an **Encoder-Decoder**.
    1.  **Embedding & Positional Encoding:** Both context and target tokens are embedded and combined with positional encodings.
    2.  **Encoder:** The `N` context tokens are fed into the Encoder stack. The encoder layers use **self-attention** to build a rich, contextualized representation of the input n-gram. The final output is the `memory`.
    3.  **Decoder:** The *single* target token is fed into the Decoder. The decoder first uses **masked self-attention** (to prevent it from seeing "future" tokens, though here it's just one token). Then, it uses **cross-attention** to look at the encoder's `memory`, allowing it to decide which parts of the input context are most important for predicting the next word.
    4.  **Output Layer:** The decoder's output is passed through a linear layer to get the final vocabulary logits.
* **How it Works:** The Transformer does not rely on sequential hidden states. Its power comes from **attention**, which allows it to directly model the relationship between any two tokens in the context, regardless of distance. For this n-gram task, it learns to "attend" to the most relevant context words to make its prediction.

#### Analysis & Comparison

| Feature | RNN (GRU) | Transformer |
| :--- | :--- | :--- |
| **Core Idea** | Sequential hidden state (memory) | Parallel attention mechanism |
| **Context Handling** | Compresses context into a fixed-size state. Can "forget" early tokens. | Attends to all context tokens simultaneously. Better at long-range dependencies. |
| **Training** | Sequential, cannot be parallelized *within* a sequence. Generally faster per epoch for small models. | Highly parallelizable *across* tokens. More complex and often slower per epoch, but scales better. |
| **Complexity** | Simpler to implement and understand. Fewer hyperparameters. | Highly complex, with many from-scratch components (Attention, FFN, Layers). |
| **Task Suitability** | Very well-suited for this simple n-gram task. | Overkill for a small `N=2` context, but its architecture is the state-of-the-art for larger-scale language modeling. |
| **Repetition Issue** | As noted in the original notebook, greedy `argmax` decoding causes repetition. | This is also true for Transformers. The problem isn't the model, but the *decoding strategy*. |

**Solution to Repetition:** The `generate.py` script solves this by implementing **Top-K Sampling**. Instead of just picking the *most likely* word (greedy), it:
1.  Gets the logits for all words.
2.  Selects the `K` words with the highest logits (e.g., `K=10`).
3.  Redistributes the probability mass among only these `K` words (via softmax).
4.  *Samples* from this new, smaller distribution.
This introduces randomness, allowing the model to escape repetitive loops and produce more diverse, natural-sounding text.

---

## Project Structure

```
pytorch-persian-text-generation/
├── .gitignore              # Ignores Python cache, data, logs, and model files
├── LICENSE                 # MIT License file
├── README.md               # This file
├── requirements.txt        # Project dependencies
├── scripts/
│   ├── download_data.sh    # Script to download data from Kaggle
│   ├── preprocess.py       # Data cleaning and vocabulary building script
│   ├── train.py            # Main script to train either model
│   └── generate.py         # Main script to generate text with a trained model
├── src/                    # All Python source code
│   ├── __init__.py
│   ├── dataset.py          # Dataloader and n-gram creation logic
│   ├── utils.py            # Helpers for logging, plots, and file I/O
│   └── models/
│       ├── __init__.py
│       ├── rnn.py          # RNN (GRU) model definition
│       └── transformer.py  # From-scratch Transformer model definition
└── run_project.ipynb       # Jupyter Notebook to run the full pipeline

````

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-persian-text-generation.git
    cd pytorch-persian-text-generation
    ```

2.  **Setup and Download Data:**
    * First, install all required Python packages.
        ```bash
        pip install -r requirements.txt
        ```
    * **Kaggle API:** This project requires the Kaggle API. Ensure you have `kaggle.json` in your `~/.kaggle/` directory.
    * Run the download script. This will create `data/` and `logs/` folders and download the datasets.
        ```bash
        bash scripts/download_data.sh
        ```

3.  **Preprocess the Data:**
    * Run the preprocessing script to clean the text and build the vocabulary.
        ```bash
        python scripts/preprocess.py
        ```
    * This will create `data/processed/processed_text.txt` and `data/processed/vocab.json`.

4.  **Train a Model:**
    * **To train the RNN (GRU) model:**
        ```bash
        python scripts/train.py \
            --model_type rnn \
            --n_gram 3 \
            --batch_size 128 \
            --epochs 10 \
            --embed_size 128 \
            --hidden_size 256 \
            --num_layers 2
        ```
    * **To train the Transformer model:**
        ```bash
        python scripts/train.py \
            --model_type transformer \
            --n_gram 2 \
            --batch_size 128 \
            --epochs 10 \
            --embed_size 128 \
            --num_heads 4 \
            --num_layers 2 \
            --d_ff 256
        ```
    * Models are saved to `outputs/models/` and plots to `outputs/plots/`.

5.  **Generate Text:**
    * Use a trained model to generate new text.
    * **Example with the RNN:**
        ```bash
        python scripts/generate.py \
            --model_type rnn \
            --model_path 'outputs/models/rnn_best.pth' \
            --seed_text 'تاریخ ایران بسیار' \
            --n_gram 3 \
            --top_k 10 \
            --embed_size 128 \
            --hidden_size 256 \
            --num_layers 2
        ```
    * **Example with the Transformer:**
        ```bash
        python scripts/generate.py \
            --model_type transformer \
            --model_path 'outputs/models/transformer_best.pth' \
            --seed_text 'تاریخ ایران بسیار' \
            --n_gram 2 \
            --top_k 10 \
            --embed_size 128 \
            --num_heads 4 \
            --num_layers 2 \
            --d_ff 256
        ```

6.  **Run with the Notebook:**
    * Alternatively, you can run the entire pipeline step-by-step by opening and running the cells in `run_project.ipynb`.

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
