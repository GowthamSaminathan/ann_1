from transformers import pipeline, set_seed
import concurrent.futures

# Initialize the generator pipeline and set the seed
generator = pipeline('text-generation', model='openai-gpt')
set_seed(42)

# Define a function to generate text for a given query
def generate_text(query):
    return generator(query[:-1], max_length=30, num_return_sequences=5)

# Process user queries using multiple threads
while True:
    query = input("Query > ")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(generate_text, query)
        results = future.result()
    print("========================================\n\n")
    for r in results:
        print(r)
    print("==========================================\n\n")
