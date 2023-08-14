# GEMATRON aka gem.py
import re
from bs4 import BeautifulSoup
import re
import urllib
from playwright.async_api import async_playwright


def rule_30(current_generation):
    next_generation = []
    for i in range(len(current_generation)):
        left = current_generation[i - 1] if i > 0 else 0
        center = current_generation[i]
        right = current_generation[i + 1] if i < len(current_generation) - 1 else 0
        next_generation.append(0 if left == right else 1)
    return next_generation



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

        # Get the page's title
        title = await page.title()

        # Get the page's content
        content = await page.content()

        # Parse and clean content using BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        clean_text = soup.get_text()

        # Combine title and cleaned content
        content = title + "\n" + clean_text

        # Remove extra whitespaces
        content = "\n".join([line.strip() for line in content.splitlines() if line.strip()])

        # Use regex to remove unwanted patterns
        patterns_to_remove = [
            r"https?://\S+",  # URLs
            r"\S+@\S+",  # Emails
            r"\d{5}-\d{4}|\d{5}|\d{9}",  # ZIP codes
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # Phone numbers
            # This is a simple address regex, which might catch some addresses but not all
            r"\d{1,5}\s\w.\s(\b\w*\b\s){1,2}\w*\.",  
        ]

        for pattern in patterns_to_remove:
            content = re.sub(pattern, "", content)

        await browser.close()
        
        return content




def generate_gpt2_response(seed_text, max_length=500):
    input_ids = gpt2_tokenizer.encode(seed_text, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.8)
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)


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




async def main():
    # 1. Get an input by the user, as text.
    text = input("Enter your text: ")
    print(f"1. Got input: {text}")

    # 2. Tokenize the input using the GPT2.
    print("2. Tokenizing input...")
    inputs = gpt2_tokenizer.encode(text, return_tensors="pt")
    max_length = len(inputs[0])

    # 3. Do the web search with the terms.
    search_results = await fetch_search_results(text)
    print(f"3. Web search results: {search_results}")

    # 4. Get the data from the search’s tokenise the data.
    print("4. Tokenizing first search result...")
    search_result_text = search_results if search_results else ""
    search_result_tokenized = gpt2_tokenizer.encode(search_result_text, return_tensors="pt")

    # 5. Pass the tokenized data to the cellular automata as seed.
    hash_object = hashlib.md5(search_result_text.encode())
    hex_dig = hash_object.hexdigest()
    seed_value = int(hex_dig, 16) % 256
    pattern = [int(b) for b in format(seed_value, '08b')]

    # 6. Get the result from automata.
    for _ in range(len(pattern)):
        pattern = rule_30(pattern)
    ca_response = pattern[len(pattern) // 2]
    print(f"6. Cellular Automata Result: {ca_response}")

    # 7. Turn both data into a seed back into as many words as the seed allows using the language model.
    combined_seed = f"{ca_response} {gpt2_tokenizer.decode(search_result_tokenized[0])}"[:max_length]
    response_from_gpt2 = generate_gpt2_response(combined_seed, max_length)
    print(f"7. GPT-2 Output: {response_from_gpt2}")

    # 8. Turn the words into gematria number.
    gematria_values, total_value = simple_gematria_with_total(response_from_gpt2)
    if gematria_values:
        gematria_formula = ", ".join([f"{pair[0]}: {pair[1]}" for pair in gematria_values])
        print(f"8. Gematria: {gematria_formula} | Total: {total_value}")
    else:
        print("8. Gematria: No valid letters found.")
        
        combined_seed = f"{ca_response} {gpt2_tokenizer.decode(inputs[0])}"[:max_length]
    print(f"9. Combined Seed: {combined_seed}")  # Added this line to output the combined seed.

    response_from_gpt2 = generate_gpt2_response(combined_seed, max_length)
    print(f"10. GPT-2 Output: {response_from_gpt2}")

await main()

# WORKING
