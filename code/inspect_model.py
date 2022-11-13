# This is a script for inspecting spans
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from span_model import IMDBSpan
from jsd import JSD
from glove import Glove  # TODO
from tokenizer import Tokenizer  # TODO


span_model_checkpoint = "baseline_glove_cnn_spans.pt"

# TODO: From Save
NUM_SPANS = 4
NUM_EMBED_DIMS = 100
THETA = 0.5

# TODO: avoid
glove = Glove()
glove_fname = f'glove.6B.{NUM_EMBED_DIMS}d.txt'
glove_path = os.path.join('models', 'glove', glove_fname)
glove.load_embeddings(glove_path, NUM_EMBED_DIMS)
tokenizer = Tokenizer(glove_embeds=glove)
M = tokenizer.build_nn_embedding_matrix()


model = IMDBSpan(embedding_matrix=M, num_spans=NUM_SPANS)  # TODO: update to not have to pass M

checkpoint = torch.load(span_model_checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])


text = "It started in the jungle. An alien warrior came to hunt the soldiers. I thought it was a really good movie."
text = "Dark Angel is a futuristic sci-fi series, set in post-apocalyptic Seattle, centering on Max (Jessica Alba), a genetically enhanced young woman, on the run from her creators.<br /><br />The Dark Angel universe is absorbing, (not as much as say Buffy, but absorbing nonetheless) with an interesting and believable set of characters. Certainly, it is not for everyone, but those who give it time will find themselves watching one of the most enjoyable series out there. Dark Angel is criminally overlooked, and under-rated, and was unfortunatly canceled after only 2 series. Which was a great shame, as this had the potential to become a great series, although its 42 episodes are only 10 shy of long running BBC sci-fi comedy Red Dwarf. As it is Dark Angel remains unfinished, so seek it out, and if you want more, lobby Fox to make another series."
tokens, token_ids, mask = tokenizer.tokenize(text, truncate=512, padding=512)

xs_tensor = torch.as_tensor([token_ids])  # B, 512
masks_tensor = torch.as_tensor([mask]).float()  # B, 512

model.eval()
y_pred, r = model(xs_tensor, masks_tensor)
jsds = JSD(r, THETA)

print(y_pred, jsds)
print(r.shape)


fig = plt.figure(figsize=(16, 12))
for j in range(NUM_SPANS):
    one_r = r[0, :, j].detach().cpu().numpy()
    print(one_r.shape)
    plt.plot(one_r)

plt.savefig('all_r.png')


print(tokens)
print('------------------------------------')

for j in range(NUM_SPANS):
    one_r = r[0, :, j].detach().cpu().numpy()
    inds = np.where(one_r > 0.01)
    print(np.array(tokens)[inds])
