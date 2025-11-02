"""Semantic similarity computation and clustering of user messages.

This module provides functionality to cluster similar user messages using
sentence transformers and hierarchical clustering.
"""

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
    """Compute sentence embeddings using sentence-transformers.

    Returns:
        Tuple of (model, embeddings) - model is needed for duplicate filtering
    """
    # Guard: Validate non-empty input
    if not messages:
        print(
            "Error: Cannot compute embeddings for empty message list", file=sys.stderr
        )
        sys.exit(1)

    # Import sentence_transformers (can be slow on first load)
    print_info("  Initializing embedding model...")
    import time

    init_start = time.time()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.", file=sys.stderr)
        print("Install with: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)
    init_time = time.time() - init_start
    print_verbose(f"  Initialization completed in {init_time:.1f}s")

    print_verbose(f"  Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    texts = [msg["message"] for msg in messages]
    print_info(f"  Computing embeddings for {len(texts)} messages...")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print_verbose(f"  Computed embeddings: shape {embeddings.shape}")

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

    print_verbose(f"  Clustering with distance threshold {threshold}...")

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

    print_info(f"  Found {len(clusters)} initial clusters")

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
        print_verbose("  No clusters to filter - skipping duplicate filtering")
        return clusters_data

    # Load existing commands
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
    """Prepare clustered messages as in-memory data structures.

    Applies filtering:
    - Minimum cluster size (default: 2, removes small clusters)
    - Minimum 5 words in representative message
    - Duplicate detection: filters clusters similar to existing slash commands
    - Impact threshold: keeps clusters with ≥ min_absolute messages AND ≥ min_percentage of total

    Args:
        clusters: List of message index lists per cluster
        messages: Original messages
        embeddings: Precomputed embeddings
        labels: Cluster labels
        model: SentenceTransformer model for duplicate detection
        min_size: Minimum cluster size (default: 2)
        commands_dir: Directory with existing slash commands (default: .claude/commands)
        similarity_threshold: Threshold for duplicate detection (default: 0.85)
        min_absolute: Minimum absolute message count per cluster (default: 5)
        min_percentage: Minimum percentage of total messages per cluster (default: 0.03)

    Returns:
        Tuple of (cluster metadata list, cluster messages dict)
        - cluster metadata: List[Dict] with cluster_id, size, representative
        - cluster messages: Dict[cluster_id, List[message_dicts]]
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Default commands directory
    if commands_dir is None:
        commands_dir = Path.cwd() / ".claude" / "commands"

    # Track filtering stages for pipeline visualization
    initial_cluster_count = len(clusters)

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

    # Track size after first filtering (min_size and word_count)
    after_size_filter = len(output)

    # Filter duplicates against existing slash commands
    output = filter_duplicate_clusters(
        output, model, commands_dir, similarity_threshold
    )
    after_duplicate_filter = len(output)

    # Apply impact threshold: keep clusters with ≥ min_absolute AND ≥ min_percentage
    total_messages = sum(c["size"] for c in output)
    threshold = max(min_absolute, math.ceil(total_messages * min_percentage))

    impact_clusters = [c for c in output if c["size"] >= threshold]

    # Re-assign cluster IDs sequentially after filtering
    old_to_new_id = {}
    final_cluster_messages = {}

    for idx, cluster in enumerate(impact_clusters):
        old_id = cluster["cluster_id"]
        old_to_new_id[old_id] = idx
        cluster["cluster_id"] = idx
        # Map messages with new cluster ID
        final_cluster_messages[idx] = cluster_messages_map[old_id]

    # Calculate coverage of filtered clusters
    filtered_messages = sum(c["size"] for c in impact_clusters)

    # Show filtering pipeline funnel
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
