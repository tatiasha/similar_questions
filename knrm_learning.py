import torch
import json
import os
from src.knrm import Solution_KNRM


if __name__ == "__main__":
    obj = Solution_KNRM(glue_qqp_dir=os.environ['GLUE_QQP_DIR'], glove_vectors_path=os.environ['EMB_PATH_GLOVE'])
    obj.train(n_epochs=20)

    with open(os.environ['VOCAB_PATH'], 'w') as wfile:
        json.dump(obj.vocab, wfile)
    torch.save(obj.model.mlp, os.environ['MLP_PATH'])
    torch.save(obj.model.embeddings.state_dict(), os.environ['EMB_PATH_KNRM'])

