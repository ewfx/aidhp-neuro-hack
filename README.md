# Install the dependencies
pip install torch pandas numpy transformers sentence-transformers scikit-learn rank_bm25

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 \
transformers>=4.25.1 \
sentence-transformers \
scikit-learn \
pandas \
numpy \
rank_bm25 \
accelerate \
bitsandbytes

# Prepare the data
py .\customer_data_generator.py

# To get the recommendations 
py .\recommender.py
