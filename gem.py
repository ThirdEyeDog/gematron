from transformers import GPT2LMHeadModel, GPT2Tokenizer
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import urllib.parse
import asyncio
import hashlib
import urllib
import torch
import re
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "gpt2-medium"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

#v.0.0.0.3
# current

import time

def rule_30(current_generation, generations=7):
    key_binary = ""
    for _ in range(generations):
        next_generation = []
        # ASCII visualization of the current generation
        print(''.join('#' if cell == 1 else '.' for cell in current_generation))
        time.sleep(0.3) # Delay for animation effect
        for i in range(len(current_generation)):
            left = current_generation[i - 1] if i > 0 else 0
            center = current_generation[i]
            right = current_generation[(i + 1) % len(current_generation)]
            next_cell = (left, center, right)
            rule = {(1, 1, 1): 0, (1, 1, 0): 0, (1, 0, 1): 0, (1, 0, 0): 1, (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): 0}
            next_generation.append(rule[next_cell])
        key_binary += ''.join(map(str, next_generation))
        current_generation = next_generation
    key_hex = hex(int(key_binary[:28], 2))[2:].zfill(7) # Using only the first 28 bits to create a 7-character key
    return key_hex



# define search function

async def fetch_search_results(query):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        search_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
        await page.goto(search_url)
        await page.wait_for_selector(".b_algo")

        # Get the link of the first search result
        first_link_element = await page.query_selector(".b_algo h2 a")
        
        # If there's no link found, we simply return an empty string
        if not first_link_element:
            await browser.close()
            return ""

        first_link = await first_link_element.get_attribute("href")
        
        # Navigate to the first link
        await page.goto(first_link)

        # Extract content from <main> element
        main_content_element = await page.query_selector("main")
        main_content = await main_content_element.inner_text() if main_content_element else ""

        # Split the content to lines and get the middle part
        lines = main_content.splitlines()
        mid_idx = len(lines) // 2
        middle_content = "\n".join(lines[mid_idx-5:mid_idx+5]) if len(lines) > 10 else main_content

        # Combine title and middle content
        title = await page.title()
        content = "\n" + middle_content

        # Remove extra whitespaces
        content = "\n".join([line.strip() for line in content.splitlines() if line.strip()])

        # Use regex to remove unwanted patterns
        patterns_to_remove = [
            r"https?://\S+",  # URLs
            r"\S+@\S+",  # Emails
            r"\d{5}-\d{4}|\d{5}|\d{9}",  # ZIP codes
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # Phone numbers
            r"\d{1,5}\s\w.\s(\b\w*\b\s){1,2}\w*\.",  # Address
        ]

        for pattern in patterns_to_remove:
            content = re.sub(pattern, "", content)

        await browser.close()
        
        return content
    
def generate_gpt2_response(seed_text, max_length=50):
    input_ids = gpt2_tokenizer.encode(seed_text, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=1.2)
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
    print(f"1. Got input: {text}")

    # 2. Tokenize the input using the GPT2 tokenizer from Transformers
    print("2. Tokenizing input...")
    inputs = gpt2_tokenizer.encode(text, return_tensors="pt", max_length=50, truncation=True)
    max_length = 50
    print(f"Tokenized input: {inputs}")

    # 3. Do the web search with the terms.
    search_results = await fetch_search_results(text)
    print(f"3. Web search results: {search_results}")

    # 4. Get the data from the search’s tokenise the data.
    print("4. Tokenizing first search result...")
    search_result_text = search_results if search_results else ""
    search_result_tokenized = gpt2_tokenizer.encode(search_result_text, return_tensors="pt")

    # 5. Pass the tokenized data to the cellular automata as seed.
    print("5. Creating seed for cellular automata...")
    hash_object = hashlib.md5(search_result_text.encode())
    hex_dig = hash_object.hexdigest()
    seed_value = int(hex_dig, 16) % 256
    pattern = [int(b) for b in format(seed_value, '08b')] * 8

    # 6. Get the result from automata.
    print("6. Running cellular automata...")
    key = rule_30(pattern)
    print(f"6. 7-Character Key: {key}")

    # 7. Turn both data into a seed back into as many words as the seed allows using the language model.
    print("7. Generating GPT-2 response...")
    combined_seed = f"{key} {gpt2_tokenizer.decode(search_result_tokenized[0])}"[:max_length]
    response_from_gpt2 = generate_gpt2_response(combined_seed, max_length)
    print(f"7. GPT-2 Output: {response_from_gpt2}")

    # 8. Turn the words into gematria number.
    print("8. Calculating gematria...")
    gematria_values, total_value = simple_gematria_with_total(response_from_gpt2)
    if gematria_values:
        gematria_formula = ", ".join([f"{pair[0]}: {pair[1]}" for pair in gematria_values])
        print(f"8. Gematria: {gematria_formula} | Total: {total_value}")

        # 9. Lookup the gematria number on gematrix.org
        print("9. Looking up gematria value on gematrix.org...")
        gematrix_result = await gematria_lookup(total_value)
        print(f"9. Gematrix.org Result: {gematrix_result}")
    else:
        print("8. Gematria: No valid letters found.")

await main()

