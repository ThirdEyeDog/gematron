# GEMATRON aka gem.py

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
        search_results_elements = await page.query_selector_all(".b_algo")
        search_results = [await element.text_content() for element in search_results_elements]
        await browser.close()
        return search_results

def generate_gpt2_response(seed_text, max_length=500):
    input_ids = gpt2_tokenizer.encode(seed_text, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.8)
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

def simple_gematria_with_total(word):
    word = word.upper()
    gematria_mapping = {chr(i): i - 64 for i in range(65, 91)}
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
    search_result_text = search_results[0] if search_results else ""
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

