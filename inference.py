from generation import LlaSMolGeneration

from huggingface_hub import login
login('TOKEN')

generator = LlaSMolGeneration('osunlp/LlaSMol-Mistral-7B', device='cuda')
query = ('Give me a molecule that satisfies the conditions outlined in the description: '
         'The molecule is a member of the class of tripyrroles that is a red-coloured pigment')
out = generator.generate(query)

generator.model.save_pretrained("./saved_model")
generator.tokenizer.save_pretrained("./saved_model")

print(out)