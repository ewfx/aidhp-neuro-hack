import torch
import pandas as pd
import numpy as np
from transformers import GPTJForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import json
import os
from rank_bm25 import BM25Okapi
import warnings
warnings.filterwarnings("ignore")

class AdaptiveBankingRecommender:
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_data()
        self.initialize_models()
        self.setup_systems()
        
    def load_data(self):
        """Load all data files"""
        self.customers = pd.read_csv(f'{self.data_path}/customers.csv')
        self.transactions = pd.read_csv(f'{self.data_path}/transactions.csv')
        self.sentiments = pd.read_csv(f'{self.data_path}/sentiment.csv')
        
        # Convert dates
        self.transactions['purchase_date'] = pd.to_datetime(self.transactions['purchase_date'])
        self.sentiments['timestamp'] = pd.to_datetime(self.sentiments['timestamp'])
        
        # Product knowledge base
        self.products = {
            "Credit Cards": [
                {"id": "CC-1001", "name": "Travel Rewards", "features": "No foreign fees, 3x travel points"},
                {"id": "CC-2002", "name": "Cash Back", "features": "2% unlimited cash back"},
                {"id": "CC-3003", "name": "Premium", "features": "Concierge, lounge access"}
            ],
            "Loans": [
                {"id": "LN-4001", "name": "Personal Loan", "features": "Fixed 6.99% APR"},
                {"id": "LN-4002", "name": "Business Loan", "features": "Up to $500k"}
            ],
            "Investments": [
                {"id": "INV-5001", "name": "Robo-Advisor", "features": "Automated investing"},
                {"id": "INV-5002", "name": "Wealth Management", "features": "Personal advisor"}
            ]
        }
    
    def initialize_models(self):
        """Initialize ML models"""
        # GPT-J for explanations
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.gptj = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Behavior clustering
        self.kmeans = KMeans(n_clusters=5)
        self.fit_behavior_clusters()
    
    def setup_systems(self):
        """Setup retrieval systems"""
        # Product retrieval
        product_docs = []
        for category, items in self.products.items():
            for product in items:
                doc = f"{product['name']} {product['features']} {category}"
                product_docs.append((product['id'], doc))
        
        self.product_retriever = BM25Okapi([doc.split() for _, doc in product_docs])
        self.product_corpus = product_docs
        
        # Customer embeddings
        self.customer_embeddings = {
            cust_id: self.embedder.encode(self.get_customer_context(cust_id))
            for cust_id in self.customers['customer_id']
        }
    
    def fit_behavior_clusters(self):
        """Cluster customers by spending behavior"""
        features = []
        for cust_id in self.customers['customer_id']:
            profile = self.get_customer_profile(cust_id)
            features.append([
                profile['avg_amount'],
                profile['luxury_ratio'],
                profile['international_ratio']
            ])
        
        self.kmeans.fit(features)
    
    def get_customer_profile(self, customer_id):
        """Get comprehensive customer profile"""
        cust_data = self.customers[self.customers['customer_id'] == customer_id].iloc[0]
        transactions = self.transactions[self.transactions['customer_id'] == customer_id]
        sentiments = self.sentiments[self.sentiments['customer_id'] == customer_id]
        
        # Recent transactions (last 90 days)
        recent = transactions[transactions['purchase_date'] > datetime.now() - timedelta(days=90)]
        
        return {
            'industry': cust_data['industry'],
            'needs': cust_data['financial_needs'],
            'preferences': cust_data['preferences'],
            'total_spent': transactions['amount'].sum(),
            'avg_amount': transactions['amount'].mean(),
            'main_category': transactions['category'].mode()[0],
            'luxury_ratio': len(transactions[transactions['category'] == 'Luxury']) / len(transactions),
            'international_ratio': len(transactions[transactions['category'] == 'Travel']) / len(transactions),
            'recent_spending_change': recent['amount'].mean() / transactions['amount'].mean() if len(recent) > 0 else 1,
            'sentiment': sentiments['sentiment_id'].mean() if len(sentiments) > 0 else 3
        }
    
    def get_customer_context(self, customer_id):
        """Generate text context for embeddings"""
        profile = self.get_customer_profile(customer_id)
        return (
            f"Industry: {profile['industry']}. Needs: {profile['needs']}. "
            f"Prefers: {profile['preferences']}. Spends mostly on: {profile['main_category']}. "
            f"Recent spending trend: {'up' if profile['recent_spending_change'] > 1.2 else 'down' if profile['recent_spending_change'] < 0.8 else 'stable'}"
        )
    
    def detect_behavior_change(self, customer_id):
        """Detect significant behavior changes"""
        profile = self.get_customer_profile(customer_id)
        changes = []
        
        if profile['recent_spending_change'] > 1.5:
            changes.append('spending_increase')
        if profile['luxury_ratio'] > 0.3:
            changes.append('luxury_interest')
        if profile['international_ratio'] > 0.2:
            changes.append('international_interest')
        if profile['sentiment'] < 2.5:
            changes.append('dissatisfied')
            
        return changes
    
    def retrieve_products(self, query, top_k=5):
        """Retrieve relevant products"""
        tokenized_query = query.split()
        scores = self.product_retriever.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.product_corpus[i][0] for i in top_indices]
    
    def generate_explanation(self, customer_id, product_id):
        """Generate personalized explanation with GPT-J"""
        profile = self.get_customer_profile(customer_id)
        product = next((p for cat in self.products.values() for p in cat if p['id'] == product_id), None)
        
        if not product:
            return "We recommend this product based on your profile."
        
        prompt = f"""Customer Profile:
- Industry: {profile['industry']}
- Financial Needs: {profile['needs']}
- Preferences: {profile['preferences']}
- Spending Pattern: {profile['main_category']}
- Recent Behavior: {self.detect_behavior_change(customer_id)}

Product:
- Name: {product['name']}
- Features: {product['features']}

Generate a concise, natural language explanation (1-2 sentences) why this product fits the customer's needs:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.gptj.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation.replace(prompt, "").strip()
    
    def get_recommendations(self, customer_id, n=5):
        """Generate personalized recommendations"""
        if customer_id not in self.customers['customer_id'].values:
            return []
        
        profile = self.get_customer_profile(customer_id)
        behavior_changes = self.detect_behavior_change(customer_id)
        
        # Base query
        query = f"{profile['needs']} {profile['preferences']} {profile['main_category']}"
        
        # Augment query based on behavior changes
        if 'luxury_interest' in behavior_changes:
            query += " premium luxury"
        if 'international_interest' in behavior_changes:
            query += " international travel"
        if 'spending_increase' in behavior_changes:
            query += " high limit"
        
        # Retrieve products
        product_ids = self.retrieve_products(query)
        
        # Generate recommendations
        recommendations = []
        for product_id in product_ids:
            explanation = self.generate_explanation(customer_id, product_id)
            product = next((p for cat in self.products.values() for p in cat if p['id'] == product_id))
            
            recommendations.append({
                'product_id': product_id,
                'name': product['name'],
                'category': next(cat for cat, items in self.products.items() if product in items),
                'explanation': explanation,
                'match_reason': behavior_changes if behavior_changes else ['general_fit']
            })
        
        return recommendations[:n]
    
    def update_with_new_data(self, new_transactions=None, new_sentiments=None):
        """Update model with new data"""
        if new_transactions is not None:
            new_transactions['purchase_date'] = pd.to_datetime(new_transactions['purchase_date'])
            self.transactions = pd.concat([self.transactions, new_transactions])
        
        if new_sentiments is not None:
            new_sentiments['timestamp'] = pd.to_datetime(new_sentiments['timestamp'])
            self.sentiments = pd.concat([self.sentiments, new_sentiments])
        
        # Retrain behavior clusters
        self.fit_behavior_clusters()
        
        # Update customer embeddings
        for cust_id in new_transactions['customer_id'].unique():
            self.customer_embeddings[cust_id] = self.embedder.encode(self.get_customer_context(cust_id))

# Example Usage
if __name__ == "__main__":
    # First generate data if not exists
    if not os.path.exists('data/customers.csv'):
        print("Generating sample data...")
        import customer_data_generator  # Runs the data generation script
    
    # Initialize recommender
    recommender = AdaptiveBankingRecommender()
    
    # Example: Get recommendations for customer 42
    customer_id = 42
    print(f"\nRecommendations for customer {customer_id}:")
    recs = recommender.get_recommendations(customer_id)
    
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec['name']} ({rec['category']})")
        print(f"   Reason: {', '.join(rec['match_reason'])}")
        print(f"   Explanation: {rec['explanation']}")
    
    # Example: Simulate new data and update
    print("\nSimulating new luxury purchase...")
    new_transaction = pd.DataFrame([{
        'customer_id': customer_id,
        'product_id': 'CC-3003',
        'transaction_type': 'Purchase',
        'category': 'Luxury',
        'amount': 4500,
        'purchase_date': datetime.now().strftime('%Y-%m-%d'),
        'payment_mode': 'Credit Card'
    }])
    
    recommender.update_with_new_data(new_transactions=new_transaction)
    
    print("\nUpdated recommendations:")
    updated_recs = recommender.get_recommendations(customer_id)
    for i, rec in enumerate(updated_recs, 1):
        print(f"{i}. {rec['name']} - {rec['explanation']}")
        
 
        
