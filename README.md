# Research Paper Recommendation System

This project is a machine learning-based system that recommends relevant research papers to users based on input queries or paper abstracts. The model uses the Sentence Transformer library to generate semantic embeddings and predict subject areas.

## Features

- Recommend research papers based on input text
- Predict subject area from research paper abstracts
- Uses Sentence Transformer for semantic similarity

## Files

- `app.py`: Main application script
- `Research Paper recommendation and subject area prediction using sentence transformer.ipynb`: Jupyter notebook with the full implementation
- `.google-cookie`: Helper file (ignore if not required)

## Dataset

The dataset used in this project can be downloaded from Kaggle:
👉 [Semantic Scholar Papers (Arxiv)](https://www.kaggle.com/datasets/Cornell-University/arxiv)

Due to GitHub’s 25MB upload limit, the dataset is **not included** in this repository. Please download it directly from Kaggle and place it in the appropriate folder before running the code.

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the app:
    ```bash
    python app.py
    ```

## Requirements

- Python 3.x
- pandas
- numpy
- sentence-transformers
- scikit-learn

## License

This project is open-source and available under the [MIT License](LICENSE).
