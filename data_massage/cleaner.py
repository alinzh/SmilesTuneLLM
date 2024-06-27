from transformers import AutoTokenizer

def remove_trash_tokens(tokens: list, file_path: str) -> list:
    """Delete unnecessary tokens from file"""
    with open(file_path) as f:
        lines = [line.rstrip() for line in f]

    for i, line in enumerate(lines):
        new_line = ''
        for token in tokens:
            if new_line == '':
                new_line = line.replace(token, '')
            else:
                new_line = new_line.replace(token, '')
        lines[i] = new_line

    with open(file_path.replace('.txt', '_clean.txt'), 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

    return lines


if __name__ == "__main__":
    path = '/home/user/PycharmProjects/SmilesTuneMistral/language_models/llama3-8/result_llama_tune_v3.txt'

    tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='/home/user/PycharmProjects/SmilesTuneMistral/language_models/llama-smiles-tokenizer-large'
    )

    EOS_TOKEN = tokenizer.eos_token 
    BOS_TOKEN = tokenizer.bos_token
    remove_trash_tokens(['<|end_of_text|>O', EOS_TOKEN, BOS_TOKEN], path)

