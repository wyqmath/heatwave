#!/usr/bin/env python3

import os
import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import faiss
import numexpr
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config/default.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"Failed to load configuration file {config_path}: {e}")
        return get_default_config()

def get_default_config():
    return {
        "node_grouping": {
            "similarity_threshold": 0.2,
            "batch_size": 1000,
            "max_workers": 30,
            "search_k": 500,
            "ivf_nprobe": 50,
            "search_batch_size": 1000,
            "min_group_size": 2,
            "max_groups_display": -1
        }
    }

def setup_cpu_environment(config):
    os.environ["FAISS_NO_GPU"] = "1"
    device = "cpu"
    
    logger.info("✅ Using CPU mode for node grouping")
    logger.info(f"   CPU cores: {os.cpu_count()}")
    
    return device

def validate_entities(entities):
    valid_entities = []
    invalid_count = 0
    
    for e in entities:
        if isinstance(e, str) and e.strip():
            valid_entities.append(e.strip())
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid entities, filtered")
    
    return valid_entities

def encode_entities_batch(model, entities, batch_size):
    all_vectors = []
    num_batches = (len(entities) + batch_size - 1) // batch_size
    
    logger.info(f"Encoding {len(entities)} entities in batches, batch size: {batch_size}")
    
    for i in tqdm(range(0, len(entities), batch_size), desc="Batch Encoding"):
        batch_entities = entities[i:i+batch_size]
        batch_texts = [e.lower().strip() for e in batch_entities]
        
        batch_vectors = model.encode(batch_texts, show_progress_bar=False)
        all_vectors.append(batch_vectors)
    
    vectors = np.vstack(all_vectors)
    vectors_float32 = vectors.astype('float32')
    
    logger.info(f"✅ Vector encoding completed: {vectors_float32.shape}")
    return vectors_float32

def create_faiss_index(vectors, config):
    dimension = vectors.shape[1]
    
    nlist = min(int(np.sqrt(len(vectors))), 1000)
    logger.info(f"Creating IVF index, number of cluster centers: {nlist}")
    
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    faiss.normalize_L2(vectors)
    
    logger.info("Training IVF index...")
    if not index.is_trained:
        train_size = min(len(vectors), 50000)
        train_vectors = vectors[:train_size]
        index.train(train_vectors)
        logger.info(f"✅ IVF index training completed, using {train_size} vectors")
    
    logger.info(f"Adding {len(vectors)} vectors to index...")
    index.add(vectors)
    
    node_config = config.get("node_grouping", {})
    nprobe = node_config.get("ivf_nprobe", 50)
    nprobe = min(nprobe, index.nlist)
    index.nprobe = nprobe
    logger.info(f"Setting IVF search parameters: nprobe={nprobe}")
    
    return index

def search_similar_vectors(index, vectors, config):
    node_config = config.get("node_grouping", {})
    search_k = node_config.get("search_k", 500)
    k = min(len(vectors), search_k)
    search_batch_size = node_config.get("search_batch_size", 1000)
    
    logger.info(f"Searching for the {k} most similar vectors, batch size: {search_batch_size}")
    
    all_distances = []
    all_indices = []
    
    for i in tqdm(range(0, len(vectors), search_batch_size), desc="FAISS Search"):
        batch_vectors = vectors[i:i+search_batch_size]
        batch_distances, batch_indices = index.search(batch_vectors, k)
        all_distances.append(batch_distances)
        all_indices.append(batch_indices)

    distances = np.vstack(all_distances)
    indices = np.vstack(all_indices)
    logger.info(f"✅ FAISS search completed: {distances.shape}")

    return distances, indices

def group_similar_entities(entities, distances, indices, config):
    node_config = config.get("node_grouping", {})
    SIMILARITY_THRESHOLD = node_config.get("similarity_threshold", 0.2)
    min_group_size = node_config.get("min_group_size", 2)

    logger.info(f"Starting grouping, similarity threshold: {SIMILARITY_THRESHOLD}")

    cosine_distances = 1.0 - distances

    groups = []
    processed = set()

    for i, entity in enumerate(tqdm(entities, desc="Grouping Progress")):
        if entity not in processed:
            similar_indices = []
            for j, dist in zip(indices[i], cosine_distances[i]):
                if dist < SIMILARITY_THRESHOLD and j != i and j < len(entities):
                    similar_indices.append(j)

            group = [entity] + [entities[j] for j in similar_indices if entities[j] not in processed]

            if len(group) >= min_group_size:
                groups.append(group)
                processed.update(group)
            else:
                processed.add(entity)

    groups.sort(key=len, reverse=True)

    logger.info(f"✅ Grouping completed, total {len(groups)} groups")
    return groups

def write_groups_to_file(groups, output_file, config):
    node_config = config.get("node_grouping", )
    max_groups_display = node_config.get("max_groups_display", -1)

    if max_groups_display > 0 and len(groups) > max_groups_display:
        groups_to_write = groups[:max_groups_display]
        logger.info(f"Writing only top {max_groups_display} largest groups")
    else:
        groups_to_write = groups
        logger.info("Writing all groups")

    logger.info(f"Writing output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("grouped_nodes\n")
        for group in tqdm(groups_to_write, desc="Writing Progress"):
            f.write(','.join(group) + '\n')

    return groups_to_write

def main():
    logger.info("🚀 Starting node similarity grouping...")
    logger.info("=" * 60)

    config = load_config()

    device = setup_cpu_environment(config)

    input_file = "combination_nodes.csv"
    output_file = "output_relation_nodes.csv"

    if not Path(input_file).exists():
        logger.error(f"❌ Input file does not exist: {input_file}")
        return 1

    logger.info(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    entities = df['Node'].tolist()
    logger.info(f"Read {len(entities)} nodes")

    entities = validate_entities(entities)
    if not entities:
        logger.error("❌ No valid entities to process")
        return 1

    logger.info(f"Valid entities count: {len(entities)}")

    logger.info("Loading sentence embedding model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info("✅ Model loaded to CPU")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        return 1

    node_config = config.get("node_grouping", {})
    max_workers = node_config.get("max_workers", 30)
    faiss.omp_set_num_threads(max_workers)
    numexpr.set_num_threads(max_workers)
    logger.info(f"Parallel threads set to: {max_workers}")

    batch_size = node_config.get("batch_size", 1000)
    vectors = encode_entities_batch(model, entities, batch_size)

    index = create_faiss_index(vectors, config)

    distances, indices = search_similar_vectors(index, vectors, config)

    groups = group_similar_entities(entities, distances, indices, config)

    groups_written = write_groups_to_file(groups, output_file, config)

    logger.info("=" * 60)
    logger.info("✅ Node grouping completed!")
    logger.info(f"   Processed entities: {len(entities)}")
    logger.info(f"   Generated groups: {len(groups)}")
    logger.info(f"   Written groups: {len(groups_written)}")
    if groups:
        logger.info(f"   Max group size: {max(len(g) for g in groups)}")
        logger.info(f"   Avg group size: {sum(len(g) for g in groups) / len(groups):.1f}")
    logger.info(f"   Output file: {output_file}")
    logger.info("=" * 60)
    logger.info("🎉 Node grouping script executed successfully!")

    return 0

if __name__ == "__main__":
    exit(main())