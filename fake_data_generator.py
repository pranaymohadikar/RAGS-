# import pandas as pd
# import numpy as np
# from faker import Faker

# # Initialize faker for realistic text generation
# fake = Faker()

# # Define message templates for each sentiment
# positive_templates = [
#     "I'm really happy with {product}, it works perfectly!",
#     "Great service from {company}, very professional.",
#     "The {product} exceeded my expectations, thank you!",
#     "Excellent customer support, my issue was resolved quickly.",
#     "I love the new {product}, it's exactly what I needed.",
#     "Fast delivery and high quality product, will buy again!",
#     "The team at {company} was very helpful and friendly.",
#     "Perfect experience with {product}, no complaints at all.",
#     "5 stars for {company}, amazing service as always.",
#     "I recommend {product} to everyone, it's fantastic!"
# ]

# neutral_templates = [
#     "I received my order of {product}, it's okay.",
#     "The service was average, nothing special.",
#     "I have mixed feelings about {product}.",
#     "It works as described, but could be better.",
#     "Standard experience with {company}.",
#     "The {product} is fine, but not exceptional.",
#     "My experience was neither good nor bad.",
#     "The delivery was on time, product is acceptable.",
#     "I expected more from {company}, but it's not bad.",
#     "The {product} meets basic requirements."
# ]

# negative_templates = [
#     "I'm very disappointed with {product}, it broke immediately.",
#     "Terrible service from {company}, will never buy again.",
#     "The {product} is defective and doesn't work properly.",
#     "Awful customer support, they didn't help at all.",
#     "My order arrived late and damaged, very unhappy.",
#     "The quality of {product} is much worse than expected.",
#     "I regret buying from {company}, waste of money.",
#     "The {product} is nothing like advertised.",
#     "Extremely frustrated with the service at {company}.",
#     "Worst experience ever with {product}, avoid at all costs."
# ]

# # Generate sample data
# data = []
# for _ in range(100):
#     company = fake.company()
#     product = fake.word(ext_word_list=['phone', 'laptop', 'TV', 'headphones', 'monitor', 'tablet'])
    
#     # Randomly select sentiment (weighted distribution)
#     sentiment = np.random.choice(
#         ['Positive', 'Neutral', 'Negative'],
#         p=[0.4, 0.3, 0.3]  # 40% positive, 30% neutral, 30% negative
#     )
    
#     if sentiment == 'Positive':
#         template = np.random.choice(positive_templates)
#     elif sentiment == 'Neutral':
#         template = np.random.choice(neutral_templates)
#     else:
#         template = np.random.choice(negative_templates)
    
#     message = template.format(company=company, product=product)
#     data.append({'message_id': fake.uuid4(), 'message': message, 'sentiment': sentiment})

# # Create DataFrame
# df = pd.DataFrame(data)

# # Save to CSV
# df.to_csv('customer_service_messages.csv', index=False)
# print("Sample dataset created with 100 rows:")
# print(df.head())

# # Show distribution
# print("\nSentiment distribution:")
# print(df['sentiment'].value_counts())




import pandas as pd

# Sample Data
data = {
    'Review_Message': [
        "The airport was very clean and easy to navigate. The staff was helpful and friendly.",
        "The security lines were extremely long and there was no clear direction. Very frustrating experience.",
        "Great shopping options and the lounges were very comfortable. A positive experience overall.",
        "The flight delays were not communicated properly. I had to wait for hours with no information.",
        "Security checks were smooth, but the airport could use more food options in the terminal.",
        "I missed my flight due to poor signage and lack of assistance from the staff. Terrible service.",
        "The airport is very well-maintained and the staff is professional. I always enjoy flying from here.",
        "Very crowded and chaotic. Not enough seating areas for passengers. Needs better organization.",
        "My bags were delayed but the staff provided updates and apologized, which made it less stressful.",
        "The terminal was outdated and uncomfortable. The WiFi was slow, and the food was overpriced."
    ],
    'Sentiment': [
        'Positive',  # Positive feedback
        'Negative',  # Negative feedback
        'Positive',  # Positive feedback
        'Negative',  # Negative feedback
        'Neutral',   # Neutral feedback
        'Negative',  # Negative feedback
        'Positive',  # Positive feedback
        'Negative',  # Negative feedback
        'Neutral',   # Neutral feedback
        'Negative'   # Negative feedback
    ]
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("airport_reviews.csv", index=False)

print("CSV file 'airport_reviews.csv' has been generated.")
