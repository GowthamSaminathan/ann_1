from transformers import pipeline, set_seed
import concurrent.futures

# Initialize the generator pipeline and set the seed
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
set_seed(42)

# Define a function to generate text for a given query
def generate_response(query):
    return generator(query, max_length=50, do_sample=True, temperature=0.7)

# Define a function to format and print the chatbot response
def print_response(response):
    print("ChatGPT: " + response['generated_text'].strip())

# Chat with the chatbot using multiple threads
while True:
    query = input("You: ")
    if query.lower() == 'bye':
        print_response(generate_response(query))
        break
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(generate_response, query)
        response = future.result()[0]
    print_response(response)
