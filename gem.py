#v.0.0.0.6 GEMATRON

# run on a jupypter notebook to have a better experience
# install list: !pip install  transformers torch secrets matplotlib numpy langchain playwright 

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.nn.functional import softmax
import secrets
import matplotlib.pyplot as plt
import numpy as np
import unittest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from playwright.async_api import async_playwright
from langchain.llms import OpenAI
from bs4 import BeautifulSoup
import urllib.parse
import asyncio
import hashlib
import urllib
import torch
import re
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "gpt2-medium"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

# Load pre-trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2) # Assuming binary sentiment






RULE = {(1, 1, 1): 0, (1, 1, 0): 0, (1, 0, 1): 0, (1, 0, 0): 1, (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): 0}

def rule_30(current_generation, generations=7):
    key_binary = ""
    # We'll save all generations in a list for the visualization
    all_generations = [current_generation]
    for _ in range(generations):
        next_generation = []
        for i in range(len(current_generation)):
            left = current_generation[i - 1] if i > 0 else 0
            center = current_generation[i]
            right = current_generation[(i + 1) % len(current_generation)]
            next_generation.append(RULE[(left, center, right)])
        key_binary += ''.join(map(str, next_generation))
        all_generations.append(next_generation) # Save the generation
        current_generation = next_generation

    # Visualization of the automaton
    plt.imshow(np.array(all_generations), cmap="binary", interpolation="none")
    plt.show()

    key_hex = hex(int(key_binary[:28], 2))[2:].zfill(7)
    return key_hex


# define search function



# 
async def fetch_search_results(query, max_results=20):
    content = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        search_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&PC=U316&FORM=CHROMN"
        await page.goto(search_url)
        await page.wait_for_selector(".b_algo")

        # Loop through the search results
        for i in range(max_results):
            link_element = await page.query_selector(f".b_algo:nth-child({i + 1}) h2 a")

            # If there's no link found, continue to next iteration
            if not link_element:
                continue

            link = await link_element.get_attribute("href")
            await page.goto(link)

            # Get the content of the <main> element, and if not found, get the content of the <body> element
            main_content_element = await page.query_selector("main") or await page.query_selector("body")
            content = await main_content_element.inner_text() if main_content_element else ""

            if content:
                break

        await browser.close()
        return content if content else "No suitable content found. The application will not proceed."










    
def generate_gpt2_response(seed_text, max_length=512):
    input_ids = gpt2_tokenizer.encode(seed_text, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=1.5)
    generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Check if the generated text is the same as the seed text
    if generated_text.strip() == seed_text.strip():
        return "Generated text is the same as seed. Try again."
    return generated_text

#gematria script translation

def simple_gematria_with_total(word):
    word = word.upper()
    
    # Define the gematria mapping for English gematria
    gematria_mapping = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
        'J': 10, 'K': 20, 'L': 30, 'M': 40, 'N': 50, 'O': 60, 'P': 70, 'Q': 80, 'R': 90,
        'S': 100, 'T': 200, 'U': 300, 'V': 400, 'W': 500, 'X': 600, 'Y': 700, 'Z': 800
    }
    
    results, total = [], 0
    for letter in word:
        if letter in gematria_mapping:
            results.append((letter, gematria_mapping[letter]))
            total += gematria_mapping[letter]
    return results, total

async def gematria_lookup(number):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        lookup_url = f"https://www.gematrix.org/?word={number}"
        await page.goto(lookup_url)

        # Extract content from the page
        content_element = await page.query_selector("body")
        if content_element:
            full_content = await content_element.inner_text()
            target_text = "Gematria Words And Phrases Of"
            target_index = full_content.find(target_text)
            if target_index != -1:
                content = full_content[target_index + len(target_text):]
            else:
                content = "Target text not found"
        else:
            content = "Body element not found"

        await browser.close()
        return content
    
    

    
    


async def main():
    # Initialize the tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 1. Get an input by the user, as text.
    text = input("Enter your text: ").strip()
    print(f"Got input: {text}")

    # 2. Tokenize the input using the GPT2 tokenizer from Transformers
    print("Tokenizing input...")
    inputs = gpt2_tokenizer.encode(text, return_tensors="pt", max_length=50, truncation=True)
    max_length = 50
    print(f"Tokenized input: {inputs}")

    # 3. Do the web search with the terms.
    search_results = await fetch_search_results(text)
    print(f"Web search results: {search_results}")

    # 4. Get the data from the search’s tokenise the data.
    print("Tokenizing first search result...")
    search_result_text = search_results if search_results else ""
    search_result_tokenized = gpt2_tokenizer.encode(search_result_text, return_tensors="pt")

    # 5. Pass the tokenized data to the cellular automata as seed.
    print("Creating seed for cellular automata...")
    hash_object = hashlib.md5(search_result_text.encode())
    hex_dig = hash_object.hexdigest()
    seed_value = int(hex_dig, 16) % 256
    pattern = [int(b) for b in format(seed_value, '08b')] * 8

    # 6. Get the result from automata.
    print("Running cellular automata...")
    key = rule_30(pattern)
    print(f"7-Character Key: {key}")

    # 7. Turn both data into a seed back into as many words as the seed allows using the language model.
    print("7. Generating GPT-2 response...")
    combined_seed = f"{key} {gpt2_tokenizer.decode(search_result_tokenized[0])}"[:max_length]
    response_from_gpt2 = generate_gpt2_response(combined_seed, max_length)
    print(f"GPT-2 Output: {response_from_gpt2}")

    # 8. Turn the words into gematria number.
    print("Calculating gematria...")
    gematria_values, total_value = simple_gematria_with_total(response_from_gpt2)
    if gematria_values:
        gematria_formula = ", ".join([f"{pair[0]}: {pair[1]}" for pair in gematria_values])
        print(f"Gematria: {gematria_formula} | Total: {total_value}")

        # 9. Lookup the gematria number on gematrix.org
        print("Looking up gematria value on gematrix.org...")
        gematrix_result = await gematria_lookup(total_value)
        print(f"Gematrix.org Result: {gematrix_result}")
    else:
        print(" Gematria: No valid letters found.")
        
     # Directory path
    dir_path = './tarot'
    images = os.listdir(dir_path)
    
    
    # use total_value as tarot draw seed
    secretsGenerator = secrets.SystemRandom()
    secretsGenerator.seed(total_value)

    # Utilize CSPRNG
    # rng = random.SystemRandom()
    # selected_card = rng.choice(images)
    selected_card = secretsGenerator.choice(images)


    # Open and display the selected image
    image_path = os.path.join(dir_path, selected_card)
    image = PILImage.open(image_path)
    display(image)

    # Output the card name
    card_name = os.path.splitext(selected_card)[0]
    print('Card Name:', card_name)
    
    #sentiment score 
    #sia = SentimentIntensityAnalyzer()
    #sentiment_result = sia.polarity_scores(gematrix_result)
    #sentiment_score = sentiment_result['compound']
    #print(f"Gematria Sentiment Score: {sentiment_score}")



    # Tokenize the input
    inputs = tokenizer(gematrix_result, return_tensors="pt", max_length=512, truncation=True)

    # Predict the sentiment
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)

    # Get the sentiment score; the result will depend on how the model was fine-tuned
    sentiment_score = probs[0][1].item() - probs[0][0].item()

    print(f"Sentiment Score: {sentiment_score}")



await main()





