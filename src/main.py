# start src/main.py
"""Main entry point for LinkedIn Bias Auditor.

Provides the CLI interface and orchestrates the audit pipeline including
data loading, embedding generation, similarity calculation, and result storage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, cast

import typer
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from src.analysis import (
    SocialPost,
    calculate_cosine_similarity,
    construct_prompt,
    load_posts,
)
from src.config import get_settings
from src.database import clear_results, get_db_engine, init_db, save_result
from src.embedder import LlamaEmbedder
from src.logger import setup_logging
from src.report import (
    BiasReportGenerator,
    ReportData,
    generate_methodology_explainer,
    generate_readme_explainer,
)
from src.stats import analyze_bias, analyze_single_pair

app = typer.Typer()


@dataclass
class AuditStats:
    """Track audit processing statistics."""

    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0

    def report(self) -> str:
        """Generate summary report."""
        return (
            f"Total: {self.total}, "
            f"Processed: {self.processed}, "
            f"Skipped: {self.skipped}, "
            f"Failed: {self.failed}"
        )


def initialize_environment(dry_run: bool) -> None:
    """Initialize application environment for audit processing.

    Sets up logging and optionally initializes the database.

    Args:
        dry_run: If True, skips database initialization.
    """
    setup_logging()
    logging.info("🚀 Starting LinkedIn Bias Auditor")
    if not dry_run:
        engine = get_db_engine()
        init_db(engine)
        logging.info("💾 Database initialized")
    else:
        logging.info("🚧 Dry Run: Database initialization skipped")


def load_input_data() -> tuple[list[SocialPost], list[SocialPost]]:
    """Load and validate input data from configured directories.

    Loads male and female post data from JSON files and validates
    that both datasets have matching record counts.

    Returns:
        Tuple of (male_posts, female_posts) lists.

    Raises:
        typer.Exit: If input files are missing or have mismatched counts.
    """
    settings = get_settings()
    input_dir = settings.paths.input_dir
    male_path = input_dir / "maledata.json"
    female_path = input_dir / "femaledata.json"

    if not male_path.exists() or not female_path.exists():
        logging.error(f"🛑 Input files not found in {input_dir}")
        raise typer.Exit(code=1)

    male_posts = load_posts(male_path)
    female_posts = load_posts(female_path)

    if len(male_posts) != len(female_posts):
        logging.error("🛑 Mismatched record counts between male and female data sets")
        raise typer.Exit(code=1)

    logging.info(f"📥 Loaded {len(male_posts)} pairs of posts")
    return male_posts, female_posts


def validate_pair(male_post: SocialPost, female_post: SocialPost) -> bool:
    """Validate that a pair is suitable for comparison.

    Returns True if valid, False if should be skipped.
    """
    if male_post.text_content != female_post.text_content:
        logging.warning(
            f"🟡 Skipping pair {male_post.name}/{female_post.name}: "
            "Text content mismatch invalidates bias comparison"
        )
        return False
    return True


def process_single_pair(
    male_post: SocialPost,
    female_post: SocialPost,
    embedder: LlamaEmbedder,
    historical_scores: Optional[list[float]] = None,
) -> Optional[dict[str, object]]:
    """Process a single pair and return the result record.

    Args:
        male_post: Male-attributed post
        female_post: Female-attributed post
        embedder: LLaMA embedder for hidden state extraction
        historical_scores: Optional list of previous scores for statistical context

    Returns:
        Result dict if successful, None if failed.
    """
    male_prompt = construct_prompt(male_post)
    female_prompt = construct_prompt(female_post)

    try:
        vec_a = embedder.get_embedding(male_prompt)
        vec_b = embedder.get_embedding(female_prompt)

        score = calculate_cosine_similarity(vec_a, vec_b)

        # Use statistical analysis for verdict
        analysis = analyze_single_pair(score, historical_scores)

        return {
            "male_name": male_post.name,
            "female_name": female_post.name,
            "cosine_similarity": score,
            "bias_verdict": str(analysis["basic_verdict"]),
            "deviation_from_perfect": cast(float, analysis["deviation_from_perfect"]),
            "z_score": analysis.get("z_score"),
            "percentile": analysis.get("percentile"),
        }

    except Exception as e:
        logging.error(f"🛑 Error processing pair {male_post.name}: {e}")
        return None


def simulate_pair(male_post: SocialPost, female_post: SocialPost, index: int) -> bool:
    """Simulate processing a pair - validates data without API calls.

    Returns True if pair would be processed, False if skipped.
    """
    if not validate_pair(male_post, female_post):
        logging.info(f"[Simulate] Pair {index}: Would be SKIPPED (content mismatch)")
        return False

    # Validate prompts can be constructed
    male_prompt = construct_prompt(male_post)
    female_prompt = construct_prompt(female_post)

    logging.info(
        f"[Simulate] Pair {index}: {male_post.name}/{female_post.name} - "
        f"Would process (prompts: {len(male_prompt)}/{len(female_prompt)} chars)"
    )
    return True


@app.command()
def audit(
    simulate: bool = typer.Option(
        default=False,
        is_flag=True,
        flag_value=True,
        help="Validate data without calling API or saving to DB",
    ),
) -> None:
    """Audit LinkedIn Retrieval Bias."""
    initialize_environment(simulate)
    male_posts, female_posts = load_input_data()

    stats = AuditStats(total=len(male_posts))
    similarity_scores: list[float] = []  # Track for aggregate analysis
    all_results: list[dict[str, Any]] = []  # Collect all results for report

    if simulate:
        # Simulation mode: validate data without API calls
        logging.info("🚧 Running in SIMULATE mode - no API calls or DB writes")
        for i in tqdm(range(len(male_posts)), desc="Simulating"):
            male_post = male_posts[i]
            female_post = female_posts[i]
            if simulate_pair(male_post, female_post, i):
                stats.processed += 1
            else:
                stats.skipped += 1

        logging.info(f"✅ Simulation Complete - {stats.report()}")
        return

    # Initialize LLaMA Embedder - fail hard if model not found
    settings = get_settings()
    try:
        embedder = LlamaEmbedder(
            model_path=settings.embedding.model_path,
            device=settings.embedding.device,
        )
        logging.info("🟢 LLaMA embedder initialized")
    except FileNotFoundError as e:
        logging.error(f"🛑 Failed to initialize LLaMA embedder: {e}")
        logging.error("🛑 Cannot proceed without model. Exiting.")
        raise typer.Exit(code=1) from e
    except Exception as e:
        logging.error(f"🛑 Failed to load model: {e}")
        raise typer.Exit(code=1) from e

    # Create DB engine once (not per-pair)
    engine = get_db_engine()

    # Clear existing results to avoid duplicates
    cleared = clear_results(engine)
    if cleared > 0:
        logging.info(f"🗑️ Cleared {cleared} existing records for fresh analysis")

    # Process Loop
    for i in tqdm(range(len(male_posts)), desc="Processing"):
        male_post = male_posts[i]
        female_post = female_posts[i]

        # Validate pair before processing
        if not validate_pair(male_post, female_post):
            stats.skipped += 1
            continue

        # Pass historical scores for statistical context
        result = process_single_pair(
            male_post, female_post, embedder, historical_scores=similarity_scores
        )

        if result is not None:
            try:
                with engine.connect() as conn:
                    save_result(conn, result)
                    conn.commit()
                stats.processed += 1
                # Track score for future statistical context
                similarity_scores.append(cast(float, result["cosine_similarity"]))
                # Collect result for report
                all_results.append(result)
            except SQLAlchemyError as e:
                logging.error(f"🛑 Failed to save result for {male_post.name}: {e}")
                stats.failed += 1
        else:
            stats.failed += 1

    # Report final statistics with aggregate analysis
    if stats.failed > 0:
        logging.warning(f"⚠️ Audit completed with errors - {stats.report()}")
    else:
        logging.info(f"✅ Audit Complete - {stats.report()}")

    # Generate aggregate statistical report
    if similarity_scores:
        aggregate_stats = analyze_bias(similarity_scores)
        logging.info("📊 Aggregate Statistical Analysis:")
        logging.info(f"   Mean Similarity: {aggregate_stats.mean_similarity:.4f}")
        logging.info(f"   Std Deviation: {aggregate_stats.std_similarity:.4f}")
        logging.info(
            f"   95% CI (Bootstrap): [{aggregate_stats.ci_lower_bootstrap:.4f}, "
            f"{aggregate_stats.ci_upper_bootstrap:.4f}]"
        )
        if aggregate_stats.p_value_ttest is not None:
            logging.info(f"   P-value (t-test): {aggregate_stats.p_value_ttest:.4f}")
        if aggregate_stats.cohens_d is not None:
            logging.info(f"   Effect Size (Cohen's d): {aggregate_stats.cohens_d:.4f}")
        logging.info(f"   Statistical Verdict: {aggregate_stats.verdict}")

        # Generate HTML report
        report_data = ReportData(
            aggregate_stats=aggregate_stats,
            pair_results=all_results,
            metadata={
                "model": settings.embedding.model_path,
                "timestamp": datetime.now(),
                "sample_size": len(all_results),
            },
        )
        generator = BiasReportGenerator(str(settings.paths.output_dir))
        report_path = generator.generate_report(report_data)
        logging.info(f"📄 Report generated: {report_path}")

        # Generate methodology explainer (static page)
        explainer_path = generate_methodology_explainer(
            str(settings.paths.output_dir / "analysis")
        )
        logging.info(f"📄 Methodology explainer generated: {explainer_path}")

        # Generate README system explainer (comprehensive documentation)
        readme_path = generate_readme_explainer(str(settings.paths.output_dir))
        logging.info(f"📄 README explainer generated: {readme_path}")


if __name__ == "__main__":
    app()
# end src/main.py
