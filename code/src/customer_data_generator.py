import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()

# Configuration
NUM_CUSTOMERS = 100
TRANSACTIONS_PER_CUSTOMER = 20
POSTS_PER_CUSTOMER = 5

def generate_customer_data():
    industries = ['Technology', 'Retail', 'Healthcare', 'Finance', 'Manufacturing']
    needs = ['Wealth Management', 'Business Loans', 'Payment Solutions', 'Investment Banking']
    preferences = ['Digital Banking', 'Personal Advisor', 'Low Fees', 'High Credit Limits']
    
    customers = []
    for i in range(1, NUM_CUSTOMERS + 1):
        customers.append({
            'customer_id': i,
            'industry': random.choice(industries),
            'financial_needs': random.choice(needs),
            'preferences': ', '.join(random.sample(preferences, random.randint(1, 3))),
            'revenue': random.randint(500000, 10000000),
            'no_of_employees': random.randint(10, 500)
        })
    return pd.DataFrame(customers)

def generate_transaction_data(customer_ids):
    products = {
        'Credit Cards': ['CC-1001', 'CC-2002', 'CC-3003'],
        'Loans': ['LN-4001', 'LN-4002'],
        'Accounts': ['ACCT-5001', 'ACCT-5002']
    }
    categories = ['Travel', 'Retail', 'Dining', 'Entertainment', 'Luxury']
    payment_modes = ['Credit Card', 'Debit Card', 'Digital Wallet', 'Bank Transfer']
    
    transactions = []
    for cust_id in customer_ids:
        # Create spending pattern (premium vs budget)
        is_premium = random.random() < 0.3  # 30% premium customers
        
        for _ in range(TRANSACTIONS_PER_CUSTOMER):
            # Determine transaction characteristics
            if is_premium:
                amount = random.randint(200, 5000)
                if random.random() < 0.4:  # 40% luxury purchases
                    category = 'Luxury'
                    amount *= 1.5
                else:
                    category = random.choice([c for c in categories if c != 'Luxury'])
            else:
                category = random.choice([c for c in categories if c != 'Luxury'])
                amount = random.randint(10, 500)
            
            product_type = random.choice(list(products.keys()))
            product_id = random.choice(products[product_type])
            
            days_ago = random.randint(1, 365)
            purchase_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            transactions.append({
                'customer_id': cust_id,
                'product_id': product_id,
                'transaction_type': random.choice(['Purchase', 'Payment', 'Withdrawal']),
                'category': category,
                'amount': amount,
                'purchase_date': purchase_date,
                'payment_mode': random.choice(payment_modes)
            })
    return pd.DataFrame(transactions)

def generate_sentiment_data(customer_ids):
    platforms = ['Twitter', 'Facebook', 'Instagram', 'LinkedIn']
    intents = ['Feedback', 'Complaint', 'Inquiry', 'Praise', 'Recommendation']
    
    sentiments = []
    for cust_id in customer_ids:
        base_sentiment = random.randint(1, 5)  # 1-5 scale
        
        for i in range(POSTS_PER_CUSTOMER):
            days_ago = random.randint(1, 90)
            timestamp = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%dT%H:%M:%S')
            
            # Generate financial-themed content
            topics = [
                "online banking experience",
                "credit card benefits", 
                "loan application",
                "investment advice",
                "customer service"
            ]
            topic = random.choice(topics)
            content = f"My {topic} was {'excellent' if base_sentiment > 3 else 'poor'}. "
            content += fake.sentence()
            
            # Add some financial keywords
            keywords = ["fees", "interest", "rate", "approval", "service", "app"]
            content += " " + random.choice(keywords) + "."
            
            sentiments.append({
                'customer_id': cust_id,
                'post_id': f"POST-{cust_id}-{i}",
                'platform': random.choice(platforms),
                'content': content,
                'timestamp': timestamp,
                'sentiment_id': min(5, max(1, base_sentiment + random.randint(-1, 1))),
                'intent': random.choice(intents)
            })
    return pd.DataFrame(sentiments)

# Generate all data
customers_df = generate_customer_data()
transactions_df = generate_transaction_data(customers_df['customer_id'])
sentiment_df = generate_sentiment_data(customers_df['customer_id'])

# Save to CSV
os.makedirs('data', exist_ok=True)
customers_df.to_csv('data/customers.csv', index=False)
transactions_df.to_csv('data/transactions.csv', index=False)
sentiment_df.to_csv('data/sentiment.csv', index=False)

print(f"Generated {NUM_CUSTOMERS} customers with {TRANSACTIONS_PER_CUSTOMER} transactions and {POSTS_PER_CUSTOMER} posts each")