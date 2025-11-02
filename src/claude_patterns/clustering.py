"""Cluster similar messages using sentence transformers and hierarchical clustering."""

import os

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from claude_patterns.output import print_info, print_verbose


def compute_embeddings(messages: List[Dict[str, Any]], model_name: str):
    """Compute sentence embeddings. Returns (model, embeddings)."""
    if not messages:
        print(
            "Error: Cannot compute embeddings for empty message list", file=sys.stderr
        )
        sys.exit(1)

    print_info("  Initializing embedding model...")
    import time
    import torch

    init_start = time.time()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.", file=sys.stderr)
        print("Install with: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)
    init_time = time.time() - init_start
    print_verbose(f"  Initialization completed in {init_time:.1f}s")

    if torch.cuda.is_available():
        device = "cuda"
        print_verbose("  Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print_verbose("  Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print_verbose("  Using CPU")

    print_verbose(f"  Loading model: {model_name}...")
    load_start = time.time()
    model = SentenceTransformer(model_name, device=device)
    load_time = time.time() - load_start
    print_verbose(f"  Model loaded in {load_time:.1f}s")

    texts = [msg["message"] for msg in messages]
    print_info(f"  Computing embeddings for {len(texts)} messages...")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print_verbose(f"  Computed embeddings: shape {embeddings.shape}")

    return model, embeddings


def cluster_messages(
    messages: List[Dict[str, Any]], embeddings: np.ndarray, threshold: float
) -> Tuple[List[List[int]], np.ndarray]:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances

    if len(embeddings) == 0:
        print("Error: Cannot cluster empty embeddings array", file=sys.stderr)
        sys.exit(1)

    print_verbose(f"  Clustering with distance threshold {threshold}...")

    distance_matrix = cosine_distances(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,  # type: ignore[arg-type]
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )

    labels = clustering.fit_predict(distance_matrix)

    clusters_dict: Dict[Any, List[int]] = {}
    for idx, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(idx)

    clusters = list(clusters_dict.values())
    clusters.sort(key=len, reverse=True)

    print_info(f"  Found {len(clusters)} initial clusters")

    return clusters, labels


def load_existing_commands(commands_dir: Path) -> List[str]:
    """Load existing slash command prompts, extracting content after frontmatter."""
    if not commands_dir.exists():
        return []

    commands = []

    for md_file in commands_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    prompt = parts[2].strip()
                else:
                    prompt = content.strip()
            else:
                prompt = content.strip()

            if prompt:
                commands.append(prompt)

        except Exception as e:
            print(f"Warning: Could not read {md_file.name}: {e}")
            continue

    return commands


def filter_duplicate_clusters(
    clusters_data: List[Dict[str, Any]],
    model,
    commands_dir: Path,
    similarity_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Filter clusters similar to existing commands using cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    if not clusters_data:
        print_verbose("  No clusters to filter - skipping duplicate filtering")
        return clusters_data

    existing_commands = load_existing_commands(commands_dir)

    if not existing_commands:
        print_verbose(
            "  No existing slash commands found - skipping duplicate filtering"
        )
        return clusters_data

    print_verbose(
        f"  Filtering duplicates against {len(existing_commands)} existing commands..."
    )
    print_verbose(f"  Similarity threshold: {similarity_threshold}")

    command_embeddings = model.encode(
        existing_commands, show_progress_bar=False, convert_to_numpy=True
    )

    representatives = [cluster["representative"] for cluster in clusters_data]
    cluster_embeddings = model.encode(
        representatives, show_progress_bar=False, convert_to_numpy=True
    )

    similarity_matrix = cosine_similarity(cluster_embeddings, command_embeddings)
    max_similarities = similarity_matrix.max(axis=1)

    filtered_clusters = []
    filtered_count = 0

    for i, cluster in enumerate(clusters_data):
        max_sim = max_similarities[i]

        if max_sim >= similarity_threshold:
            filtered_count += 1
            print_verbose(
                f"    Filtered cluster {cluster['cluster_id']} "
                f"(similarity: {max_sim:.3f}): {cluster['representative'][:80]}..."
            )
        else:
            filtered_clusters.append(cluster)

    print_verbose(f"  Filtered out {filtered_count} duplicate clusters")
    print_verbose(f"  Remaining clusters: {len(filtered_clusters)}")

    return filtered_clusters


def prepare_clusters(
    clusters: List[List[int]],
    messages: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    model,
    min_size: int = 2,
    commands_dir: Optional[Path] = None,
    similarity_threshold: float = 0.85,
    min_absolute: int = 5,
    min_percentage: float = 0.03,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """Prepare clusters with filtering: min size, word count, duplicates, and impact threshold."""
    from sklearn.metrics.pairwise import cosine_similarity

    if commands_dir is None:
        commands_dir = Path.cwd() / ".claude" / "commands"

    initial_cluster_count = len(clusters)

    output = []
    cluster_messages_map = {}

    for cluster_id, message_indices in enumerate(clusters):
        cluster_size = len(message_indices)

        if cluster_size < min_size:
            continue

        cluster_embeddings = embeddings[message_indices]
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        similarities = cosine_similarity(cluster_embeddings, centroid).flatten()
        medoid_idx_in_cluster = similarities.argmax()
        medoid_idx = message_indices[medoid_idx_in_cluster]
        representative = messages[medoid_idx]["message"]

        word_count = len(representative.split())
        if word_count < 5:  # noqa: PLR2004
            continue

        output.append(
            {
                "cluster_id": cluster_id,
                "size": cluster_size,
                "representative": representative,
            }
        )

        cluster_messages_map[cluster_id] = [messages[idx] for idx in message_indices]

    output.sort(key=lambda x: x["size"], reverse=True)
    after_size_filter = len(output)

    output = filter_duplicate_clusters(
        output, model, commands_dir, similarity_threshold
    )
    after_duplicate_filter = len(output)

    total_messages = sum(c["size"] for c in output)
    threshold = max(min_absolute, math.ceil(total_messages * min_percentage))
    impact_clusters = [c for c in output if c["size"] >= threshold]

    old_to_new_id = {}
    final_cluster_messages = {}

    for idx, cluster in enumerate(impact_clusters):
        old_id = cluster["cluster_id"]
        old_to_new_id[old_id] = idx
        cluster["cluster_id"] = idx
        final_cluster_messages[idx] = cluster_messages_map[old_id]

    filtered_messages = sum(c["size"] for c in impact_clusters)

    print_info("\n  Cluster filtering pipeline:")
    print_info(f"    {initial_cluster_count} initial clusters")

    size_filtered = initial_cluster_count - after_size_filter
    if size_filtered > 0:
        print_info(
            f"    → {after_size_filter} after size filtering (-{size_filtered} too small or short)"
        )

    duplicate_filtered = after_size_filter - after_duplicate_filter
    if duplicate_filtered > 0:
        print_info(
            f"    → {after_duplicate_filter} after duplicate filtering (-{duplicate_filtered} similar to existing)"
        )

    impact_filtered = after_duplicate_filter - len(impact_clusters)
    if impact_filtered > 0:
        print_info(
            f"    → {len(impact_clusters)} after impact filtering (-{impact_filtered} low-impact, threshold: {threshold} messages)"
        )

    if total_messages > 0:
        coverage_pct = filtered_messages / total_messages * 100
        print_info(
            f"\n  Coverage: {filtered_messages}/{total_messages} messages ({coverage_pct:.1f}%)"
        )

    return impact_clusters, final_cluster_messages
