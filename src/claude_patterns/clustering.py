"""Semantic similarity computation and clustering of user messages.

This module provides functionality to cluster similar user messages using
sentence transformers and hierarchical clustering.
"""

import os

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np


def load_messages(json_file: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load user messages from JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        messages = json.load(f)

    if limit:
        messages = messages[:limit]

    print(f"Loaded {len(messages)} messages")
    return messages


def compute_embeddings(messages: List[Dict[str, Any]], model_name: str):
    """Compute sentence embeddings using sentence-transformers.

    Returns:
        Tuple of (model, embeddings) - model is needed for duplicate filtering
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.", file=sys.stderr)
        print("Install with: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    # Guard: Validate non-empty input
    if not messages:
        print(
            "Error: Cannot compute embeddings for empty message list", file=sys.stderr
        )
        sys.exit(1)

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    texts = [msg["message"] for msg in messages]
    print(f"Computing embeddings for {len(texts)} messages...")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"Computed embeddings: shape {embeddings.shape}")

    return model, embeddings


def cluster_messages(
    messages: List[Dict[str, Any]], embeddings: np.ndarray, threshold: float
) -> Tuple[List[List[int]], np.ndarray]:
    """Cluster messages using hierarchical clustering."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances

    # Guard: Validate non-empty input
    if len(embeddings) == 0:
        print("Error: Cannot cluster empty embeddings array", file=sys.stderr)
        sys.exit(1)

    print(f"Clustering with distance threshold {threshold}...")

    # Compute distance matrix (1 - cosine similarity)
    distance_matrix = cosine_distances(embeddings)

    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,  # type: ignore[arg-type]
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )

    labels = clustering.fit_predict(distance_matrix)

    # Group message indices by cluster
    clusters_dict: Dict[Any, List[int]] = {}
    for idx, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(idx)

    # Convert to list of clusters, sorted by size (largest first)
    clusters = list(clusters_dict.values())
    clusters.sort(key=len, reverse=True)

    print(f"Found {len(clusters)} clusters")

    return clusters, labels


def load_existing_commands(commands_dir: Path) -> List[str]:
    """Load existing slash command prompts from .claude/commands/ directory.

    Parses markdown files and extracts the prompt content (after frontmatter).

    Args:
        commands_dir: Path to .claude/commands/ directory

    Returns:
        List of command prompt strings
    """
    if not commands_dir.exists():
        return []

    commands = []

    for md_file in commands_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")

            # Skip frontmatter if present (between --- markers)
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    # Extract content after frontmatter
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
    """Filter out clusters that are similar to existing slash commands.

    Computes cosine similarity between cluster representatives and existing
    slash command prompts. Removes clusters with similarity above threshold.

    Args:
        clusters_data: List of cluster dictionaries with 'representative' field
        model: SentenceTransformer model for computing embeddings
        commands_dir: Path to .claude/commands/ directory
        similarity_threshold: Minimum similarity to consider a duplicate (default: 0.85)

    Returns:
        Filtered list of clusters
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Guard: Return early if no clusters to process
    if not clusters_data:
        print("No clusters to filter - skipping duplicate filtering")
        return clusters_data

    # Load existing commands
    existing_commands = load_existing_commands(commands_dir)

    if not existing_commands:
        print("No existing slash commands found - skipping duplicate filtering")
        return clusters_data

    print(
        f"\nFiltering duplicates against {len(existing_commands)} existing commands..."
    )
    print(f"Similarity threshold: {similarity_threshold}")

    # Compute embeddings for existing commands
    command_embeddings = model.encode(
        existing_commands, show_progress_bar=False, convert_to_numpy=True
    )

    # Compute embeddings for cluster representatives
    representatives = [cluster["representative"] for cluster in clusters_data]
    cluster_embeddings = model.encode(
        representatives, show_progress_bar=False, convert_to_numpy=True
    )

    # Compute similarity matrix: [n_clusters x n_commands]
    similarity_matrix = cosine_similarity(cluster_embeddings, command_embeddings)

    # Find max similarity for each cluster
    max_similarities = similarity_matrix.max(axis=1)

    # Filter clusters
    filtered_clusters = []
    filtered_count = 0

    for i, cluster in enumerate(clusters_data):
        max_sim = max_similarities[i]

        if max_sim >= similarity_threshold:
            filtered_count += 1
            print(
                f"  Filtered cluster {cluster['cluster_id']} "
                f"(similarity: {max_sim:.3f}): {cluster['representative'][:80]}..."
            )
        else:
            filtered_clusters.append(cluster)

    print(f"Filtered out {filtered_count} duplicate clusters")
    print(f"Remaining clusters: {len(filtered_clusters)}")

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
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """Prepare clustered messages as in-memory data structures.

    Applies filtering:
    - Minimum cluster size (default: 2, removes small clusters)
    - Minimum 5 words in representative message
    - Duplicate detection: filters clusters similar to existing slash commands
    - Pareto principle: keeps top clusters accounting for 80% of total messages

    Args:
        clusters: List of message index lists per cluster
        messages: Original messages
        embeddings: Precomputed embeddings
        labels: Cluster labels
        model: SentenceTransformer model for duplicate detection
        min_size: Minimum cluster size (default: 2)
        commands_dir: Directory with existing slash commands (default: .claude/commands)
        similarity_threshold: Threshold for duplicate detection (default: 0.85)

    Returns:
        Tuple of (cluster metadata list, cluster messages dict)
        - cluster metadata: List[Dict] with cluster_id, size, representative
        - cluster messages: Dict[cluster_id, List[message_dicts]]
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Default commands directory
    if commands_dir is None:
        commands_dir = Path.cwd() / ".claude" / "commands"

    output = []
    cluster_messages_map = {}  # Map to store full messages for each cluster

    for cluster_id, message_indices in enumerate(clusters):
        cluster_size = len(message_indices)

        # Filter: Skip clusters smaller than minimum size
        if cluster_size < min_size:
            continue

        # Get embeddings for this cluster
        cluster_embeddings = embeddings[message_indices]

        # Compute centroid (mean embedding)
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)

        # Find medoid: message with highest similarity to centroid
        similarities = cosine_similarity(cluster_embeddings, centroid).flatten()
        medoid_idx_in_cluster = similarities.argmax()
        medoid_idx = message_indices[medoid_idx_in_cluster]

        # Get the representative message (medoid)
        representative = messages[medoid_idx]["message"]

        # Filter: Skip clusters with short representatives (< 5 words)
        word_count = len(representative.split())
        if word_count < 5:  # noqa: PLR2004
            continue

        # Store cluster info
        output.append(
            {
                "cluster_id": cluster_id,
                "size": cluster_size,
                "representative": representative,
            }
        )

        # Store all messages for this cluster
        cluster_messages_map[cluster_id] = [messages[idx] for idx in message_indices]

    # Sort by size descending for Pareto analysis
    output.sort(key=lambda x: x["size"], reverse=True)

    # Filter duplicates against existing slash commands
    output = filter_duplicate_clusters(
        output, model, commands_dir, similarity_threshold
    )

    # Apply Pareto principle: keep clusters accounting for 80% of messages
    total_messages = sum(c["size"] for c in output)
    pareto_threshold = total_messages * 0.8
    cumulative_messages = 0
    pareto_clusters = []
    pareto_cluster_ids = []

    for cluster in output:
        cumulative_messages += cluster["size"]
        pareto_clusters.append(cluster)
        pareto_cluster_ids.append(cluster["cluster_id"])
        if cumulative_messages >= pareto_threshold:
            break

    # Re-assign cluster IDs sequentially after filtering
    old_to_new_id = {}
    final_cluster_messages = {}

    for idx, cluster in enumerate(pareto_clusters):
        old_id = cluster["cluster_id"]
        old_to_new_id[old_id] = idx
        cluster["cluster_id"] = idx
        # Map messages with new cluster ID
        final_cluster_messages[idx] = cluster_messages_map[old_id]

    print(f"Prepared {len(pareto_clusters)} clusters")
    print(f"  Filtered out: {len(output) - len(pareto_clusters)} low-volume clusters")
    if total_messages > 0:
        print(
            f"  Coverage: {cumulative_messages}/{total_messages} messages ({cumulative_messages / total_messages * 100:.1f}%)"
        )

    return pareto_clusters, final_cluster_messages


def save_clusters(
    clusters: List[List[int]],
    messages: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    model,
    output_dir: Optional[Path] = None,
    min_size: int = 2,
    commands_dir: Optional[Path] = None,
    similarity_threshold: float = 0.85,
) -> Tuple[List[Dict[str, Any]], Path]:
    """Save clustered messages to timestamped folder with cluster files.

    Creates a folder with timestamp containing:
    - clusters.json: Summary with representatives
    - cluster_N.json: Full messages for each cluster

    Applies filtering:
    - Minimum cluster size (default: 2, removes small clusters)
    - Minimum 5 words in representative message
    - Duplicate detection: filters clusters similar to existing slash commands
    - Pareto principle: keeps top clusters accounting for 80% of total messages

    Args:
        clusters: List of message index lists per cluster
        messages: Original messages
        embeddings: Precomputed embeddings
        labels: Cluster labels
        model: SentenceTransformer model for duplicate detection
        output_dir: Output directory (default: auto-generated with timestamp)
        min_size: Minimum cluster size (default: 2)
        commands_dir: Directory with existing slash commands (default: .claude/commands)
        similarity_threshold: Threshold for duplicate detection (default: 0.85)

    Returns:
        Tuple of (filtered cluster data, output directory path)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Default commands directory
    if commands_dir is None:
        commands_dir = Path.cwd() / ".claude" / "commands"

    # Create timestamped output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"clusters_{timestamp}")

    output_dir.mkdir(exist_ok=True)
    print(f"Creating output directory: {output_dir}")

    output = []
    cluster_messages_map = {}  # Map to store full messages for each cluster

    for cluster_id, message_indices in enumerate(clusters):
        cluster_size = len(message_indices)

        # Filter: Skip clusters smaller than minimum size
        if cluster_size < min_size:
            continue

        # Get embeddings for this cluster
        cluster_embeddings = embeddings[message_indices]

        # Compute centroid (mean embedding)
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)

        # Find medoid: message with highest similarity to centroid
        similarities = cosine_similarity(cluster_embeddings, centroid).flatten()
        medoid_idx_in_cluster = similarities.argmax()
        medoid_idx = message_indices[medoid_idx_in_cluster]

        # Get the representative message (medoid)
        representative = messages[medoid_idx]["message"]

        # Filter: Skip clusters with short representatives (< 5 words)
        word_count = len(representative.split())
        if word_count < 5:  # noqa: PLR2004
            continue

        # Store cluster info
        output.append(
            {
                "cluster_id": cluster_id,
                "size": cluster_size,
                "representative": representative,
            }
        )

        # Store all messages for this cluster
        cluster_messages_map[cluster_id] = [messages[idx] for idx in message_indices]

    # Sort by size descending for Pareto analysis
    output.sort(key=lambda x: x["size"], reverse=True)

    # Filter duplicates against existing slash commands
    output = filter_duplicate_clusters(
        output, model, commands_dir, similarity_threshold
    )

    # Apply Pareto principle: keep clusters accounting for 80% of messages
    total_messages = sum(c["size"] for c in output)
    pareto_threshold = total_messages * 0.8
    cumulative_messages = 0
    pareto_clusters = []
    pareto_cluster_ids = []

    for cluster in output:
        cumulative_messages += cluster["size"]
        pareto_clusters.append(cluster)
        pareto_cluster_ids.append(cluster["cluster_id"])
        if cumulative_messages >= pareto_threshold:
            break

    # Re-assign cluster IDs sequentially after filtering
    old_to_new_id = {}
    for idx, cluster in enumerate(pareto_clusters):
        old_id = cluster["cluster_id"]
        old_to_new_id[old_id] = idx
        cluster["cluster_id"] = idx

    # Save clusters summary
    clusters_file = output_dir / "clusters.json"
    with open(clusters_file, "w", encoding="utf-8") as f:
        json.dump(pareto_clusters, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(pareto_clusters)} clusters to {clusters_file}")
    print(f"  Filtered out: {len(output) - len(pareto_clusters)} low-volume clusters")
    if total_messages > 0:
        print(
            f"  Coverage: {cumulative_messages}/{total_messages} messages ({cumulative_messages / total_messages * 100:.1f}%)"
        )

    # Save individual cluster files with all messages
    for old_id in pareto_cluster_ids:
        new_id = old_to_new_id[old_id]
        cluster_file = output_dir / f"cluster_{new_id}.json"
        cluster_messages = cluster_messages_map[old_id]

        with open(cluster_file, "w", encoding="utf-8") as f:
            json.dump(cluster_messages, f, indent=2, ensure_ascii=False)

        print(f"  Saved cluster_{new_id}.json ({len(cluster_messages)} messages)")

    return pareto_clusters, output_dir


def print_summary(clusters_data: List[Dict[str, Any]], show_top: int = 10):
    """Print summary of clusters."""
    print(f"\n{'=' * 80}")
    print("CLUSTER SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total clusters: {len(clusters_data)}")
    print(f"Total messages: {sum(c['size'] for c in clusters_data)}")
    print(f"\nTop {show_top} clusters by size:\n")

    for i, cluster in enumerate(clusters_data[:show_top], 1):
        print(f"{i}. Cluster {cluster['cluster_id']} ({cluster['size']} messages)")
        print(f"   Representative: {cluster['representative']}")
        print()


def main():
    """CLI entry point for clustering."""
    parser = argparse.ArgumentParser(
        description="Compute semantic similarity and cluster user messages"
    )
    parser.add_argument(
        "messages_file", type=Path, help="Input JSON file with messages"
    )
    parser.add_argument("--limit", type=int, help="Process only first N messages")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Distance threshold for clustering (default: 0.7, lower = stricter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for clusters (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="Minimum cluster size (default: 2)",
    )

    args = parser.parse_args()

    if not args.messages_file.exists():
        print(f"Error: File '{args.messages_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Load messages
    messages = load_messages(args.messages_file, args.limit)

    if not messages:
        print("Error: No messages to process", file=sys.stderr)
        sys.exit(1)

    # Compute embeddings
    model, embeddings = compute_embeddings(messages, args.model)

    # Cluster messages
    clusters, labels = cluster_messages(messages, embeddings, args.threshold)

    # Save results (includes duplicate filtering against existing commands)
    clusters_data, output_dir = save_clusters(
        clusters, messages, embeddings, labels, model, args.output_dir, args.min_size
    )

    # Print summary
    print_summary(clusters_data)

    print(f"\nDone! View clusters in {output_dir}")
