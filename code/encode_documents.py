import argparse
import importlib
import os
from itertools import batched

import ir_datasets
import numpy as np
import pandas as pd
from tqdm import tqdm


def main(args):
    dataset = ir_datasets.load(args.collection)
    encoder = getattr(
        importlib.import_module("ir_models.dense"), args.encoder.capitalize()
    )()

    dids = []
    passages = []
    print("Loading dataset")
    for i, (did, text) in enumerate(dataset.docs_iter()):
        dids.append(did)
        passages.append(text)

    print("Dataset loaded!")
    id_map = pd.DataFrame({"doc_id": dids, "offset": np.arange(len(dids), dtype=int)})
    save_dir = os.path.join(args.output, "msmarco-passages", args.encoder)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, args.encoder)
    id_map.to_csv(f"{filename}_map.csv", index=False)

    encodings = []
    batch_size = 256
    batches = batched(passages, batch_size)
    for ps in tqdm(batches, total=(len(passages) + batch_size - 1) // batch_size):
        encodings.append(encoder.encode_documents(ps))

    encodings = np.vstack(encodings)
    fp = np.memmap(f"{filename}.dat", dtype="float32", mode="w+", shape=encodings.shape)
    fp[:] = encodings[:]
    fp.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection", "-c", required=True, help="The collection to use"
    )
    parser.add_argument(
        "--encoder",
        "-e",
        required=True,
        help="The model to use to encode the documents",
    )
    parser.add_argument("--output", "-o", default="../data/memmap")
    main(parser.parse_args())
