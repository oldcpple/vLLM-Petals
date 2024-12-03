import json
import random

def select_prompts(cnt):
    with open('prompt_dataset.json', 'r') as file:
        data = json.load(file)
    
    selected_instructions = random.sample([item['instruction'] for item in data], min(cnt, len(data)))
    
    with open('prompts.txt', 'w') as outfile:
        for instruction in selected_instructions:
            outfile.write(instruction + '\n')

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python get_prompt.py <count>")
    else:
        cnt = int(sys.argv[1])
        select_prompts(cnt)
