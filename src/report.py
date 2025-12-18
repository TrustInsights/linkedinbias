# start src/report.py
"""HTML report generator for bias audit results.

Generates an interactive HTML report with Tailwind CSS and Chart.js
that explains bias analysis results for both technical and non-technical
audiences.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.stats import BiasStatistics

logger = logging.getLogger(__name__)


@dataclass
class ReportData:
    """Aggregates all data needed for report generation.

    Attributes:
        aggregate_stats: Overall statistical analysis results.
        pair_results: List of individual pair comparison results.
        metadata: Additional context like model info and timestamps.
    """

    aggregate_stats: BiasStatistics
    pair_results: list[dict[str, Any]]
    metadata: dict[str, Any]


class BiasReportGenerator:
    """Generates HTML reports from bias audit results.

    Creates a self-contained HTML file with interactive charts,
    toggleable technical/non-technical views, and sortable data tables.

    Attributes:
        output_dir: Directory where the report will be saved.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Initialize the report generator.

        Args:
            output_dir: Directory where the report will be saved.
        """
        self.output_dir = Path(output_dir)

    def generate_report(self, data: ReportData) -> str:
        """Generate and save the HTML report.

        Args:
            data: ReportData containing all analysis results.

        Returns:
            Path to the generated report file.
        """
        # Create analysis subdirectory
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Generate HTML content
        html_content = self._render_full_report(data)

        # Write to file
        report_path = analysis_dir / "analysis.htm"
        report_path.write_text(html_content, encoding="utf-8")

        logger.info(f"Report generated: {report_path}")
        return str(report_path)

    def _render_full_report(self, data: ReportData) -> str:
        """Render the complete HTML report.

        Args:
            data: ReportData containing all analysis results.

        Returns:
            Complete HTML document as string.
        """
        stats = data.aggregate_stats
        pairs = data.pair_results
        metadata = data.metadata

        # Prepare data for JavaScript
        similarity_scores = [p.get("cosine_similarity", 0) for p in pairs]
        pairs_json = json.dumps(pairs, default=str)
        scores_json = json.dumps(similarity_scores)

        verdict_color = self._get_verdict_color(stats)
        effect_interpretation = self._interpret_effect_size(stats.cohens_d)
        plain_summary = self._generate_plain_english_summary(stats)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LinkedIn Bias Audit Report | TrustInsights.ai</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        'ti-coral': '#F05662',
                        'ti-pink': '#FF818D',
                        'ti-blue': '#33B0EF',
                    }},
                    fontFamily: {{
                        'heading': ['"Barlow Condensed"', 'sans-serif'],
                        'body': ['Barlow', 'sans-serif'],
                    }}
                }}
            }}
        }}
    </script>
    <style>
        body {{ font-family: 'Barlow', sans-serif; }}
        h1, h2, h3, h4, h5, h6 {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }}
        .verdict-green {{ background: linear-gradient(135deg, #10B981, #059669); }}
        .verdict-yellow {{ background: linear-gradient(135deg, #F59E0B, #D97706); }}
        .verdict-red {{ background: linear-gradient(135deg, #F05662, #DC2626); }}
        .hidden {{ display: none; }}
        .smooth {{ transition: all 0.3s ease; }}
        .ti-gradient {{ background: linear-gradient(135deg, #F05662 0%, #33B0EF 100%); }}
    </style>
</head>
<body class="bg-gray-50 text-gray-800 font-body">
    <!-- Navigation -->
    <nav class="bg-white shadow-md sticky top-0 z-50 border-b-4 border-ti-coral">
        <div class="max-w-7xl mx-auto px-4 py-3">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-4">
                    <img src="https://www.trustinsights.ai/wp-content/uploads/2017/12/2021-02-23_12-58-20.jpg"
                         alt="TrustInsights.ai" class="h-10">
                    <h1 class="text-xl font-heading text-gray-900">LinkedIn Bias Audit Report</h1>
                </div>
                <div class="flex space-x-4 text-sm font-medium">
                    <a href="#summary" class="text-ti-coral hover:text-ti-pink">Summary</a>
                    <a href="#dashboard" class="text-ti-coral hover:text-ti-pink">Dashboard</a>
                    <a href="#findings" class="text-ti-coral hover:text-ti-pink">Findings</a>
                    <a href="#data" class="text-ti-coral hover:text-ti-pink">Data</a>
                    <a href="#methodology" class="text-ti-coral hover:text-ti-pink">Methodology</a>
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 py-8">
        <!-- Executive Summary -->
        {self._render_executive_summary(stats, verdict_color, plain_summary, metadata)}

        <!-- Dashboard -->
        {self._render_dashboard(stats, effect_interpretation)}

        <!-- Detailed Findings -->
        {self._render_detailed_findings(stats, effect_interpretation)}

        <!-- Individual Pair Results -->
        {self._render_pair_table(pairs)}

        <!-- Methodology -->
        {self._render_methodology(metadata)}

        <!-- Glossary Sidebar Toggle -->
        {self._render_glossary()}
    </main>

    <!-- JavaScript -->
    <script>
        // Data from Python
        const pairData = {pairs_json};
        const similarityScores = {scores_json};
        const meanSimilarity = {stats.mean_similarity};
        const stdSimilarity = {stats.std_similarity};

        // Initialize charts when DOM loads
        document.addEventListener('DOMContentLoaded', function() {{
            initGaugeChart();
            initDistributionChart();
            initTable();
        }});

        // Gauge Chart
        function initGaugeChart() {{
            const ctx = document.getElementById('gaugeChart').getContext('2d');
            const value = meanSimilarity * 100;

            new Chart(ctx, {{
                type: 'doughnut',
                data: {{
                    datasets: [{{
                        data: [value, 100 - value],
                        backgroundColor: [
                            value >= 95 ? '#10B981' : value >= 80 ? '#F59E0B' : '#EF4444',
                            '#E5E7EB'
                        ],
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    circumference: 180,
                    rotation: 270,
                    cutout: '75%',
                    plugins: {{
                        tooltip: {{ enabled: false }},
                        legend: {{ display: false }}
                    }}
                }}
            }});
        }}

        // Distribution Histogram
        function initDistributionChart() {{
            const ctx = document.getElementById('distributionChart').getContext('2d');

            // Create histogram bins
            const bins = [0, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0];
            const binCounts = new Array(bins.length - 1).fill(0);

            similarityScores.forEach(score => {{
                for (let i = 0; i < bins.length - 1; i++) {{
                    if (score >= bins[i] && score < bins[i + 1]) {{
                        binCounts[i]++;
                        break;
                    }}
                    if (i === bins.length - 2 && score >= bins[i + 1]) {{
                        binCounts[i]++;
                    }}
                }}
            }});

            const labels = bins.slice(0, -1).map((b, i) =>
                `${{(b * 100).toFixed(0)}}-${{(bins[i + 1] * 100).toFixed(0)}}%`
            );

            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Number of Pairs',
                        data: binCounts,
                        backgroundColor: '#3B82F6',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Distribution of Similarity Scores'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{ display: true, text: 'Count' }}
                        }},
                        x: {{
                            title: {{ display: true, text: 'Similarity Range' }}
                        }}
                    }}
                }}
            }});
        }}

        // Table functionality
        let currentSort = {{ column: null, ascending: true }};
        let filteredData = [...pairData];

        function initTable() {{
            renderTable(filteredData);
        }}

        function renderTable(data) {{
            const tbody = document.getElementById('resultsBody');
            tbody.innerHTML = data.map((row, idx) => `
                <tr class="${{getRowClass(row.cosine_similarity)}}">
                    <td class="px-4 py-2 border-b">${{idx + 1}}</td>
                    <td class="px-4 py-2 border-b">${{row.male_name || 'N/A'}}</td>
                    <td class="px-4 py-2 border-b">${{row.female_name || 'N/A'}}</td>
                    <td class="px-4 py-2 border-b font-mono">${{(row.cosine_similarity * 100).toFixed(2)}}%</td>
                    <td class="px-4 py-2 border-b font-mono">${{row.z_score ? row.z_score.toFixed(2) : 'N/A'}}</td>
                    <td class="px-4 py-2 border-b font-mono">${{row.percentile ? row.percentile.toFixed(1) : 'N/A'}}</td>
                    <td class="px-4 py-2 border-b">${{row.bias_verdict || 'N/A'}}</td>
                </tr>
            `).join('');
        }}

        function getRowClass(similarity) {{
            if (similarity >= 0.95) return 'bg-green-50';
            if (similarity >= 0.80) return 'bg-yellow-50';
            return 'bg-red-50';
        }}

        function sortTable(column) {{
            const ascending = currentSort.column === column ? !currentSort.ascending : true;
            currentSort = {{ column, ascending }};

            filteredData.sort((a, b) => {{
                let valA = a[column];
                let valB = b[column];
                if (typeof valA === 'string') valA = valA.toLowerCase();
                if (typeof valB === 'string') valB = valB.toLowerCase();
                if (valA < valB) return ascending ? -1 : 1;
                if (valA > valB) return ascending ? 1 : -1;
                return 0;
            }});

            renderTable(filteredData);
        }}

        function filterTable() {{
            const query = document.getElementById('searchInput').value.toLowerCase();
            filteredData = pairData.filter(row =>
                (row.male_name && row.male_name.toLowerCase().includes(query)) ||
                (row.female_name && row.female_name.toLowerCase().includes(query)) ||
                (row.bias_verdict && row.bias_verdict.toLowerCase().includes(query))
            );
            renderTable(filteredData);
        }}

        function exportCSV() {{
            const headers = ['#', 'Male Name', 'Female Name', 'Similarity', 'Z-Score', 'Percentile', 'Verdict'];
            const rows = pairData.map((row, idx) => [
                idx + 1,
                row.male_name || '',
                row.female_name || '',
                row.cosine_similarity,
                row.z_score || '',
                row.percentile || '',
                row.bias_verdict || ''
            ]);

            let csv = headers.join(',') + '\\n';
            csv += rows.map(r => r.map(v => `"${{v}}"`).join(',')).join('\\n');

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'bias_audit_results.csv';
            a.click();
        }}

        // Toggle functions
        function toggleView() {{
            const tech = document.getElementById('technicalView');
            const nonTech = document.getElementById('nonTechnicalView');
            const btn = document.getElementById('toggleBtn');

            tech.classList.toggle('hidden');
            nonTech.classList.toggle('hidden');

            btn.textContent = tech.classList.contains('hidden')
                ? 'Show Technical Details'
                : 'Show Simple Explanation';
        }}

        function toggleMethodology() {{
            const tech = document.getElementById('techMethodology');
            const nonTech = document.getElementById('nonTechMethodology');
            const btn = document.getElementById('methodologyToggleBtn');

            tech.classList.toggle('hidden');
            nonTech.classList.toggle('hidden');

            btn.textContent = tech.classList.contains('hidden')
                ? 'Show Technical Details'
                : 'Show Simple Explanation';
        }}

        function toggleGlossary() {{
            document.getElementById('glossaryPanel').classList.toggle('hidden');
        }}
    </script>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-8 mt-12">
        <div class="max-w-7xl mx-auto px-4">
            <div class="flex flex-col md:flex-row items-center justify-between gap-4">
                <div class="flex items-center gap-4">
                    <img src="https://www.trustinsights.ai/wp-content/uploads/2017/12/2021-02-23_12-58-20.jpg"
                         alt="TrustInsights.ai" class="h-12">
                    <div>
                        <p class="font-heading text-lg">TrustInsights.ai</p>
                        <p class="text-gray-400 text-sm">AI-Powered Analytics & Insights</p>
                    </div>
                </div>
                <div class="text-center md:text-right text-sm text-gray-400">
                    <p>LinkedIn Bias Auditor - Detecting AI Bias in Professional Networks</p>
                    <p class="mt-1">Built with Python, Hugging Face Transformers, and statistical rigor.</p>
                </div>
            </div>
        </div>
    </footer>
</body>
</html>"""

    def _render_executive_summary(
        self,
        stats: BiasStatistics,
        verdict_color: str,
        plain_summary: str,
        metadata: dict[str, Any],
    ) -> str:
        """Render the executive summary section.

        Args:
            stats: Statistical analysis results.
            verdict_color: CSS class for verdict indicator color.
            plain_summary: Plain English summary text.
            metadata: Report metadata.

        Returns:
            HTML string for executive summary section.
        """
        timestamp = metadata.get("timestamp", datetime.now())
        if isinstance(timestamp, str):
            timestamp_str = timestamp
        else:
            timestamp_str = timestamp.strftime("%B %d, %Y at %I:%M %p")

        sample_size = stats.sample_size
        ci_lower = stats.ci_lower_bootstrap * 100
        ci_upper = stats.ci_upper_bootstrap * 100

        return f"""
        <section id="summary" class="mb-12">
            <div class="bg-white rounded-xl shadow-lg p-8">
                <div class="flex items-start gap-6">
                    <!-- Verdict Indicator -->
                    <div class="{verdict_color} rounded-full p-6 text-white">
                        <svg class="w-12 h-12" fill="currentColor" viewBox="0 0 20 20">
                            {"<path fill-rule='evenodd' d='M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z' clip-rule='evenodd'/>" if verdict_color == "verdict-green" else "<path fill-rule='evenodd' d='M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z' clip-rule='evenodd'/>"}
                        </svg>
                    </div>

                    <!-- Summary Content -->
                    <div class="flex-1">
                        <h2 class="text-2xl font-bold text-gray-900 mb-2">Executive Summary</h2>
                        <p class="text-gray-500 text-sm mb-4">Generated {timestamp_str}</p>

                        <p class="text-lg text-gray-700 leading-relaxed mb-6">
                            {plain_summary}
                        </p>

                        <!-- Key Metrics Cards -->
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div class="bg-gray-50 rounded-lg p-4">
                                <p class="text-sm text-gray-500">Mean Similarity</p>
                                <p class="text-2xl font-bold text-gray-900">{stats.mean_similarity * 100:.1f}%</p>
                            </div>
                            <div class="bg-gray-50 rounded-lg p-4">
                                <p class="text-sm text-gray-500">Pairs Analyzed</p>
                                <p class="text-2xl font-bold text-gray-900">{sample_size}</p>
                            </div>
                            <div class="bg-gray-50 rounded-lg p-4">
                                <p class="text-sm text-gray-500">95% Confidence Range</p>
                                <p class="text-2xl font-bold text-gray-900">{ci_lower:.1f}% - {ci_upper:.1f}%</p>
                            </div>
                            <div class="bg-gray-50 rounded-lg p-4">
                                <p class="text-sm text-gray-500">Statistically Significant</p>
                                <p class="text-2xl font-bold {'text-red-600' if stats.is_significant else 'text-green-600'}">
                                    {'Yes' if stats.is_significant else 'No'}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- How to Read This Report -->
                <div class="mt-6 p-6 bg-blue-50 rounded-lg border border-blue-200">
                    <h3 class="text-lg font-semibold text-blue-900 mb-3">
                        How to Read This Report
                    </h3>
                    <div class="grid md:grid-cols-2 gap-4 text-sm text-blue-800">
                        <div>
                            <p class="font-medium mb-1">Individual Pair Results (Table Below)</p>
                            <p class="text-blue-700">
                                Shows how each profile pair compares. "Small Deviation" or "Minimal Deviation"
                                means that specific pair's similarity is close to perfect. These are raw
                                measurements, not statistical conclusions.
                            </p>
                        </div>
                        <div>
                            <p class="font-medium mb-1">Overall Statistical Verdict (Above)</p>
                            <p class="text-blue-700">
                                Analyzes the <em>pattern</em> across all pairs. Even if individual deviations
                                seem small, a consistent pattern reveals systematic bias. This uses effect
                                size (Cohen's d) and statistical significance.
                            </p>
                        </div>
                    </div>
                    <p class="mt-4 text-sm text-blue-700 italic">
                        Think of it like pizza: one slightly smaller slice could be random, but if every
                        slice from the same parlor is 5% smaller, that's a policy, not an accident.
                    </p>
                </div>
            </div>
        </section>"""

    def _render_dashboard(
        self, stats: BiasStatistics, effect_interpretation: str
    ) -> str:
        """Render the interactive dashboard section.

        Args:
            stats: Statistical analysis results.
            effect_interpretation: Human-readable effect size interpretation.

        Returns:
            HTML string for dashboard section.
        """
        p_value_display = (
            f"{stats.p_value_ttest:.4f}" if stats.p_value_ttest is not None else "N/A"
        )
        effect_display = (
            f"{stats.cohens_d:.3f}" if stats.cohens_d is not None else "N/A"
        )

        return f"""
        <section id="dashboard" class="mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">Dashboard</h2>

            <div class="grid md:grid-cols-2 gap-6">
                <!-- Gauge Chart -->
                <div class="bg-white rounded-xl shadow-lg p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4 text-center">Mean Similarity Score</h3>
                    <div class="relative" style="height: 200px;">
                        <canvas id="gaugeChart"></canvas>
                        <div class="absolute inset-0 flex items-end justify-center pb-4">
                            <span class="text-4xl font-bold text-gray-900">{stats.mean_similarity * 100:.1f}%</span>
                        </div>
                    </div>
                    <p class="text-center text-gray-500 mt-2">
                        100% = Perfect match (no bias detected)
                    </p>
                </div>

                <!-- Distribution Chart -->
                <div class="bg-white rounded-xl shadow-lg p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-4">Score Distribution</h3>
                    <canvas id="distributionChart"></canvas>
                </div>
            </div>

            <!-- Statistics Cards -->
            <div class="grid md:grid-cols-4 gap-4 mt-6">
                <div class="bg-white rounded-xl shadow p-6">
                    <p class="text-sm text-gray-500 mb-1">Standard Deviation</p>
                    <p class="text-xl font-bold text-gray-900">{stats.std_similarity * 100:.2f}%</p>
                    <p class="text-xs text-gray-400 mt-1">Measures spread of scores</p>
                </div>
                <div class="bg-white rounded-xl shadow p-6">
                    <p class="text-sm text-gray-500 mb-1">P-value (t-test)</p>
                    <p class="text-xl font-bold {'text-red-600' if stats.p_value_ttest and stats.p_value_ttest < 0.05 else 'text-green-600'}">{p_value_display}</p>
                    <p class="text-xs text-gray-400 mt-1">{'Significant (p < 0.05)' if stats.p_value_ttest and stats.p_value_ttest < 0.05 else 'Not significant'}</p>
                </div>
                <div class="bg-white rounded-xl shadow p-6">
                    <p class="text-sm text-gray-500 mb-1">Effect Size (Cohen's d)</p>
                    <p class="text-xl font-bold text-gray-900">{effect_display}</p>
                    <p class="text-xs text-gray-400 mt-1">{effect_interpretation}</p>
                </div>
                <div class="bg-white rounded-xl shadow p-6">
                    <p class="text-sm text-gray-500 mb-1">Verdict</p>
                    <p class="text-lg font-bold text-gray-900">{stats.verdict}</p>
                </div>
            </div>
        </section>"""

    def _render_detailed_findings(
        self, stats: BiasStatistics, effect_interpretation: str
    ) -> str:
        """Render the detailed findings section with toggle.

        Args:
            stats: Statistical analysis results.
            effect_interpretation: Human-readable effect size interpretation.

        Returns:
            HTML string for detailed findings section.
        """
        p_value_display = (
            f"{stats.p_value_ttest:.4f}" if stats.p_value_ttest is not None else "N/A"
        )
        effect_display = (
            f"{stats.cohens_d:.3f}" if stats.cohens_d is not None else "N/A"
        )
        ci_includes_one = stats.ci_upper_bootstrap >= 1.0

        return f"""
        <section id="findings" class="mb-12">
            <div class="flex items-center justify-between mb-6">
                <h2 class="text-2xl font-bold text-gray-900">Detailed Findings</h2>
                <button id="toggleBtn" onclick="toggleView()"
                    class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
                    Show Technical Details
                </button>
            </div>

            <div class="bg-white rounded-xl shadow-lg p-8">
                <!-- Non-Technical View (default) -->
                <div id="nonTechnicalView">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">What We Tested</h3>
                    <p class="text-gray-600 mb-6">
                        We compared how an AI model perceives pairs of social media profiles that are
                        identical except for the name (one traditionally male, one traditionally female).
                        If the AI treats them the same, the "similarity score" should be close to 100%.
                    </p>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">What We Found</h3>
                    <p class="text-gray-600 mb-6">
                        Across {stats.sample_size} profile pairs, the average similarity was
                        <strong>{stats.mean_similarity * 100:.1f}%</strong>. This means the AI saw
                        {'virtually no difference' if stats.mean_similarity >= 0.95 else 'some differences' if stats.mean_similarity >= 0.80 else 'notable differences'}
                        between male and female versions of the same profile.
                    </p>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">What This Suggests</h3>
                    <p class="text-gray-600 mb-6">
                        {'The model appears to treat profiles similarly regardless of gendered names. This is a positive finding for fairness.' if not stats.is_significant else 'The differences we found are unlikely to be random chance. This suggests the model may treat profiles differently based on gendered names, which could affect fairness in recommendations or search results.'}
                    </p>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Limitations</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-2">
                        <li>This test only examines name-based bias, not other potential biases</li>
                        <li>Results depend on the specific AI model being tested</li>
                        <li>Real-world impact depends on how similarity scores affect downstream systems</li>
                    </ul>
                </div>

                <!-- Technical View (hidden by default) -->
                <div id="technicalView" class="hidden">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Hypothesis Test</h3>
                    <div class="bg-gray-50 rounded-lg p-4 mb-6 font-mono text-sm">
                        <p><strong>H<sub>0</sub>:</strong> Mean cosine similarity = 1.0 (no gender bias)</p>
                        <p><strong>H<sub>1</sub>:</strong> Mean cosine similarity &ne; 1.0 (gender bias present)</p>
                        <p class="mt-2"><strong>Test:</strong> One-sample t-test against baseline = 1.0</p>
                    </div>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Results</h3>
                    <table class="w-full mb-6">
                        <tbody class="divide-y">
                            <tr>
                                <td class="py-2 text-gray-600">Sample Size (n)</td>
                                <td class="py-2 font-mono text-right">{stats.sample_size}</td>
                            </tr>
                            <tr>
                                <td class="py-2 text-gray-600">Mean Similarity (x&#772;)</td>
                                <td class="py-2 font-mono text-right">{stats.mean_similarity:.4f}</td>
                            </tr>
                            <tr>
                                <td class="py-2 text-gray-600">Standard Deviation (s)</td>
                                <td class="py-2 font-mono text-right">{stats.std_similarity:.4f}</td>
                            </tr>
                            <tr>
                                <td class="py-2 text-gray-600">95% CI (Bootstrap)</td>
                                <td class="py-2 font-mono text-right">[{stats.ci_lower_bootstrap:.4f}, {stats.ci_upper_bootstrap:.4f}]</td>
                            </tr>
                            <tr>
                                <td class="py-2 text-gray-600">P-value (t-test)</td>
                                <td class="py-2 font-mono text-right {'text-red-600' if stats.p_value_ttest and stats.p_value_ttest < 0.05 else ''}">{p_value_display}</td>
                            </tr>
                            <tr>
                                <td class="py-2 text-gray-600">Effect Size (Cohen's d)</td>
                                <td class="py-2 font-mono text-right">{effect_display} ({effect_interpretation})</td>
                            </tr>
                        </tbody>
                    </table>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Interpretation</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-2 mb-6">
                        <li>CI {'includes' if ci_includes_one else 'excludes'} 1.0: {'Cannot reject H<sub>0</sub>' if ci_includes_one else 'Reject H<sub>0</sub>'}</li>
                        <li>P-value {'< 0.05: Statistically significant' if stats.p_value_ttest and stats.p_value_ttest < 0.05 else '>= 0.05: Not statistically significant'}</li>
                        <li>Effect size: {effect_interpretation} magnitude</li>
                    </ul>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Assumptions</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-2">
                        <li>Similarity scores are approximately normally distributed</li>
                        <li>Pairs are independent samples</li>
                        <li>Cosine similarity of 1.0 represents the null hypothesis baseline</li>
                    </ul>
                </div>
            </div>
        </section>"""

    def _render_pair_table(self, pairs: list[dict[str, Any]]) -> str:
        """Render the individual pair results table.

        Args:
            pairs: List of pair comparison results.

        Returns:
            HTML string for pair results section.
        """
        return f"""
        <section id="data" class="mb-12">
            <div class="flex items-center justify-between mb-6">
                <h2 class="text-2xl font-bold text-gray-900">Individual Pair Results</h2>
                <div class="flex gap-4">
                    <input type="text" id="searchInput" onkeyup="filterTable()"
                        placeholder="Search names or verdicts..."
                        class="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                    <button onclick="exportCSV()"
                        class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition">
                        Export CSV
                    </button>
                </div>
            </div>

            <div class="bg-white rounded-xl shadow-lg overflow-hidden">
                <div class="overflow-x-auto">
                    <table class="w-full">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-700 cursor-pointer hover:bg-gray-200"
                                    onclick="sortTable('index')">#</th>
                                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-700 cursor-pointer hover:bg-gray-200"
                                    onclick="sortTable('male_name')">Male Name</th>
                                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-700 cursor-pointer hover:bg-gray-200"
                                    onclick="sortTable('female_name')">Female Name</th>
                                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-700 cursor-pointer hover:bg-gray-200"
                                    onclick="sortTable('cosine_similarity')">Similarity</th>
                                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-700 cursor-pointer hover:bg-gray-200"
                                    onclick="sortTable('z_score')">Z-Score</th>
                                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-700 cursor-pointer hover:bg-gray-200"
                                    onclick="sortTable('percentile')">Percentile</th>
                                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-700 cursor-pointer hover:bg-gray-200"
                                    onclick="sortTable('bias_verdict')">Verdict</th>
                            </tr>
                        </thead>
                        <tbody id="resultsBody" class="divide-y">
                            <!-- Populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
                <div class="px-4 py-3 bg-gray-50 text-sm text-gray-500">
                    Showing {len(pairs)} results. Click column headers to sort.
                </div>
            </div>
        </section>"""

    def _render_methodology(self, metadata: dict[str, Any]) -> str:
        """Render the methodology explainer section.

        Args:
            metadata: Report metadata including model info.

        Returns:
            HTML string for methodology section.
        """
        model_path = metadata.get("model", "Unknown")

        return f"""
        <section id="methodology" class="mb-12">
            <div class="flex items-center justify-between mb-6">
                <h2 class="text-2xl font-bold text-gray-900">Methodology</h2>
                <button id="methodologyToggleBtn" onclick="toggleMethodology()"
                    class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
                    Show Technical Details
                </button>
            </div>

            <div class="bg-white rounded-xl shadow-lg p-8">
                <!-- Non-Technical Methodology (default) -->
                <div id="nonTechMethodology">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">How AI "Reads" Profiles</h3>
                    <p class="text-gray-600 mb-6">
                        When an AI system processes a social media profile, it converts the text into a
                        mathematical representation called an "embedding" - think of it as the AI's
                        understanding of the profile compressed into numbers. These numbers capture the
                        meaning and context of the profile.
                    </p>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">What Similarity Scores Mean</h3>
                    <p class="text-gray-600 mb-6">
                        We compare the AI's understanding of two profiles using a "cosine similarity" score.
                        If two profiles are understood identically, the score is 100%. If they're understood
                        completely differently, the score approaches 0%. For our test, we expect identical
                        profiles (differing only by name) to score very close to 100%.
                    </p>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Why This Matters for Fairness</h3>
                    <p class="text-gray-600">
                        If the AI perceives male and female versions of the same profile differently, this
                        could affect search results, recommendations, or other AI-driven decisions. Lower
                        similarity scores between gender-swapped profiles suggest the AI may be encoding
                        gender-related biases.
                    </p>
                </div>

                <!-- Technical Methodology (hidden by default) -->
                <div id="techMethodology" class="hidden">
                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Embedding Extraction</h3>
                    <div class="bg-gray-50 rounded-lg p-4 mb-6">
                        <p class="text-sm text-gray-600 mb-2"><strong>Model:</strong> {model_path}</p>
                        <p class="text-sm text-gray-600 mb-2"><strong>Method:</strong> Hidden state extraction with mean pooling</p>
                        <p class="text-sm text-gray-600 mb-2"><strong>Layer:</strong> Last transformer layer</p>
                        <p class="text-sm text-gray-600"><strong>Pooling:</strong> Mean across all tokens</p>
                    </div>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Process</h3>
                    <ol class="list-decimal list-inside text-gray-600 space-y-2 mb-6">
                        <li>Load profile text with name, headline, and content</li>
                        <li>Tokenize input using model's tokenizer</li>
                        <li>Forward pass through transformer with <code class="bg-gray-100 px-1 rounded">output_hidden_states=True</code></li>
                        <li>Extract last layer hidden states: <code class="bg-gray-100 px-1 rounded">hidden_states[-1]</code></li>
                        <li>Apply mean pooling: <code class="bg-gray-100 px-1 rounded">mean(dim=1)</code></li>
                        <li>Result: 3072-dimensional embedding vector</li>
                    </ol>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Cosine Similarity</h3>
                    <div class="bg-gray-50 rounded-lg p-4 mb-6 font-mono text-sm">
                        <p>cos(&theta;) = (A &middot; B) / (||A|| &times; ||B||)</p>
                        <p class="mt-2 text-gray-500">Range: [-1, 1], normalized to [0, 1] for this analysis</p>
                    </div>

                    <h3 class="text-xl font-semibold text-gray-800 mb-4">Statistical Tests</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-2">
                        <li><strong>One-sample t-test:</strong> Tests if mean significantly differs from 1.0</li>
                        <li><strong>Cohen's d:</strong> (1.0 - mean) / std - standardized effect size</li>
                        <li><strong>95% CI:</strong> mean &plusmn; t<sub>&alpha;/2</sub> &times; (std / &radic;n)</li>
                    </ul>
                </div>
            </div>
        </section>"""

    def _render_glossary(self) -> str:
        """Render the glossary sidebar with pizza analogies.

        Returns:
            HTML string for glossary section.
        """
        return """
        <button onclick="toggleGlossary()"
            class="fixed bottom-6 right-6 w-14 h-14 bg-blue-600 text-white rounded-full shadow-lg
                   hover:bg-blue-700 transition flex items-center justify-center z-40">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
        </button>

        <aside id="glossaryPanel" class="hidden fixed top-0 right-0 h-full w-96 bg-white shadow-2xl z-50 overflow-y-auto">
            <div class="p-6">
                <div class="flex items-center justify-between mb-6">
                    <h2 class="text-xl font-bold text-gray-900">Glossary</h2>
                    <button onclick="toggleGlossary()" class="text-gray-500 hover:text-gray-700">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                        </svg>
                    </button>
                </div>

                <div class="space-y-6">
                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Cosine Similarity</h3>
                        <p class="text-sm text-gray-600">
                            A "sameness score" - 100% means identical, lower means more different.
                        </p>
                        <p class="text-sm text-amber-700 mt-2 italic">
                            Like ordering the same pizza twice: 100% = exact same pizza.
                            95% = mostly the same, maybe slightly different cheese distribution.
                        </p>
                    </div>

                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">P-value</h3>
                        <p class="text-sm text-gray-600">
                            The probability that results this extreme would occur by random chance.
                            Below 0.05 = statistically significant.
                        </p>
                        <p class="text-sm text-amber-700 mt-2 italic">
                            If 76 pizzas in a row have 7 slices instead of 8, is that random?
                            p &lt; 0.05 means: almost certainly not an accident.
                        </p>
                    </div>

                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Effect Size (Cohen's d)</h3>
                        <p class="text-sm text-gray-600">
                            Measures how <em>big</em> the difference is, not just whether it exists.
                        </p>
                        <p class="text-sm text-amber-700 mt-2 italic">
                            How much pizza are you missing?<br>
                            <strong>Negligible (d&lt;0.2):</strong> 7.9 slices instead of 8<br>
                            <strong>Small (0.2-0.5):</strong> 7 slices instead of 8<br>
                            <strong>Medium (0.5-0.8):</strong> 6 slices instead of 8<br>
                            <strong>Large (d&gt;0.8):</strong> 4 slices instead of 8
                        </p>
                    </div>

                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Confidence Interval (95% CI)</h3>
                        <p class="text-sm text-gray-600">
                            A range where we're 95% confident the true average falls. If this range
                            includes 100%, we can't be certain there's any real bias.
                        </p>
                        <p class="text-sm text-amber-700 mt-2 italic">
                            Like saying "we're 95% sure this parlor serves 7.2-7.8 slices per pie."
                        </p>
                    </div>

                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Statistical Significance</h3>
                        <p class="text-sm text-gray-600">
                            Whether a pattern is unlikely to be random chance (p &lt; 0.05).
                        </p>
                        <p class="text-sm text-amber-700 mt-2 italic">
                            One undersized pizza could be an accident.
                            20 undersized pizzas in a row is a policy.
                        </p>
                    </div>

                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Individual vs Aggregate Verdict</h3>
                        <p class="text-sm text-gray-600">
                            Individual pairs use simple thresholds ("Small Deviation").
                            Aggregate verdict uses statistical tests ("Large Bias Detected").
                        </p>
                        <p class="text-sm text-amber-700 mt-2 italic">
                            Each slice looks fine individually. But when EVERY slice is slightly
                            smaller, that consistent pattern reveals the real issue.
                        </p>
                    </div>

                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Z-Score</h3>
                        <p class="text-sm text-gray-600">
                            How many standard deviations a value is from average.
                            Beyond &plusmn;2 = outlier.
                        </p>
                    </div>

                    <div>
                        <h3 class="font-semibold text-gray-800 mb-2">Embedding</h3>
                        <p class="text-sm text-gray-600">
                            AI's numerical representation of text that captures its meaning.
                            Like a fingerprint for content.
                        </p>
                    </div>
                </div>
            </div>
        </aside>"""

    def _get_verdict_color(self, stats: BiasStatistics) -> str:
        """Determine the verdict color based on statistical results.

        Args:
            stats: Statistical analysis results.

        Returns:
            CSS class name for verdict color.
        """
        if not stats.is_significant:
            return "verdict-green"

        # Significant bias detected - check effect size
        if stats.cohens_d is not None:
            if abs(stats.cohens_d) < 0.5:
                return "verdict-yellow"  # Small effect
            return "verdict-red"  # Medium or large effect

        # Fallback based on mean similarity
        if stats.mean_similarity >= 0.95:
            return "verdict-green"
        if stats.mean_similarity >= 0.80:
            return "verdict-yellow"
        return "verdict-red"

    def _interpret_effect_size(self, effect_size: float | None) -> str:
        """Interpret Cohen's d effect size.

        Args:
            effect_size: Cohen's d value, or None if not calculated.

        Returns:
            Human-readable interpretation of effect size.
        """
        if effect_size is None:
            return "Not calculated"

        d = abs(effect_size)
        if d < 0.2:
            return "Negligible"
        if d < 0.5:
            return "Small"
        if d < 0.8:
            return "Medium"
        return "Large"

    def _generate_plain_english_summary(self, stats: BiasStatistics) -> str:
        """Generate a plain English summary of the results with pizza analogies.

        Args:
            stats: Statistical analysis results.

        Returns:
            Human-readable summary paragraph with intuitive explanations.
        """
        mean_pct = stats.mean_similarity * 100
        deviation_pct = (1 - stats.mean_similarity) * 100
        ci_lower = stats.ci_lower_bootstrap * 100
        ci_upper = stats.ci_upper_bootstrap * 100
        n = stats.sample_size

        if not stats.is_significant:
            return (
                f"Our analysis found no statistically significant evidence of gender bias "
                f"in how the embedding model represents profiles. The average similarity "
                f"between male and female profile representations was {mean_pct:.1f}%, "
                f"which is not meaningfully different from perfect equality (100%). "
                f"We are 95% confident the true average falls between {ci_lower:.1f}% and "
                f"{ci_upper:.1f}%. This suggests the model treats gender-swapped profiles similarly."
            )

        effect_desc = self._interpret_effect_size(stats.cohens_d).lower()
        effect_val = stats.cohens_d if stats.cohens_d is not None else 0

        # Pizza analogy for explaining the contradiction
        pizza_explanation = (
            f"<p class='mt-4 p-4 bg-amber-50 border-l-4 border-amber-400 rounded'>"
            f"<strong>Understanding this result:</strong> Think of it like ordering pizza. "
            f"Each individual pizza might look 'close enough' to what you ordered "
            f"(individual pairs show small deviations). But if {n} pizzas in a row "
            f"are each {deviation_pct:.1f}% smaller than advertised, that's not random - "
            f"that's a pattern. One undersized pizza could be an accident. "
            f"{n} undersized pizzas is a policy.</p>"
        )

        if stats.cohens_d is not None and abs(stats.cohens_d) < 0.5:
            return (
                f"Our analysis detected a {effect_desc} but statistically significant "
                f"difference in how the embedding model represents male versus female profiles. "
                f"The average similarity was {mean_pct:.1f}% (a {deviation_pct:.1f}% deviation), "
                f"with a {effect_desc} effect size (Cohen's d = {effect_val:.2f}). "
                f"While individual pairs may show only minor deviations, the consistency "
                f"of this pattern across {n} pairs makes it statistically meaningful. "
                f"We are 95% confident the true average falls between {ci_lower:.1f}% and "
                f"{ci_upper:.1f}%."
                f"{pizza_explanation}"
            )

        if stats.cohens_d is not None and abs(stats.cohens_d) < 0.8:
            return (
                f"Our analysis found moderate evidence of gender bias in the embedding model. "
                f"The average similarity was {mean_pct:.1f}% (a {deviation_pct:.1f}% deviation), "
                f"with a {effect_desc} effect size (Cohen's d = {effect_val:.2f}). "
                f"This suggests meaningful differences in how the model represents profiles "
                f"based on gender. The pattern is consistent across {n} pairs, making it "
                f"unlikely to be random variation. We are 95% confident the true average "
                f"falls between {ci_lower:.1f}% and {ci_upper:.1f}%."
                f"{pizza_explanation}"
            )

        # Large effect - strongest pizza analogy
        return (
            f"Our analysis found substantial evidence of gender bias in the embedding model. "
            f"The average similarity was {mean_pct:.1f}% (a {deviation_pct:.1f}% deviation), "
            f"with a {effect_desc} effect size (Cohen's d = {effect_val:.2f}). "
            f"This indicates the model represents male and female profiles quite differently, "
            f"which could lead to biased outcomes in retrieval or recommendation systems. "
            f"We are 95% confident the true average falls between {ci_lower:.1f}% and "
            f"{ci_upper:.1f}%."
            f"<p class='mt-4 p-4 bg-red-50 border-l-4 border-red-400 rounded'>"
            f"<strong>Why 'large' bias when individual deviations look small?</strong> "
            f"Imagine ordering {n} pizzas. Each one is {deviation_pct:.1f}% smaller than "
            f"advertised - individually, you might not complain. But across {n} orders, "
            f"that consistent shortfall adds up to significant missing pizza. "
            f"Similarly, the AI model consistently treats female-named profiles differently "
            f"than male-named ones. The individual differences seem small, but the "
            f"<em>consistency</em> of the pattern reveals systematic bias.</p>"
        )


def generate_methodology_explainer(output_dir: str | Path) -> str:
    """Generate a static methodology explainer page.

    Creates a detailed HTML page explaining how this system replicates
    LinkedIn's retrieval system methodology, with visuals and both
    technical and non-technical explanations.

    Args:
        output_dir: Directory where the explainer will be saved.

    Returns:
        Path to the generated explainer file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    html_content = _render_methodology_explainer()

    explainer_path = output_path / "methodology_explainer.htm"
    explainer_path.write_text(html_content, encoding="utf-8")

    logger.info(f"Methodology explainer generated: {explainer_path}")
    return str(explainer_path)


def _render_methodology_explainer() -> str:
    """Render the complete methodology explainer HTML.

    Returns:
        Complete HTML document as string.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>How We Replicate LinkedIn's Retrieval System | TrustInsights.ai</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'ti-coral': '#F05662',
                        'ti-pink': '#FF818D',
                        'ti-blue': '#33B0EF',
                    },
                    fontFamily: {
                        'heading': ['"Barlow Condensed"', 'sans-serif'],
                        'body': ['Barlow', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <style>
        body { font-family: 'Barlow', sans-serif; }
        h1, h2, h3, h4, h5, h6 { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }
        .gradient-header { background: linear-gradient(135deg, #F05662 0%, #33B0EF 100%); }
        .step-circle {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.25rem;
        }
        .flow-arrow {
            width: 0;
            height: 0;
            border-left: 20px solid transparent;
            border-right: 20px solid transparent;
            border-top: 20px solid #F05662;
        }
        .code-block {
            background: #1E293B;
            color: #E2E8F0;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.875rem;
        }
        .comparison-linkedin { background: linear-gradient(135deg, #0077B5 0%, #00A0DC 50%); }
        .comparison-ours { background: linear-gradient(135deg, #F05662 0%, #FF818D 50%); }
    </style>
</head>
<body class="bg-gray-50 text-gray-800 font-body">
    <!-- Header -->
    <header class="gradient-header text-white py-16">
        <div class="max-w-6xl mx-auto px-6">
            <div class="flex items-center justify-center gap-4 mb-6">
                <img src="https://www.trustinsights.ai/wp-content/uploads/2017/12/2021-02-23_12-58-20.jpg"
                     alt="TrustInsights.ai" class="h-16">
            </div>
            <div class="text-center">
                <h1 class="text-4xl font-heading mb-4">How We Replicate LinkedIn's Retrieval System</h1>
                <p class="text-xl opacity-90 max-w-3xl mx-auto">
                    Understanding the methodology behind our bias audit tool and how it mirrors
                    LinkedIn's state-of-the-art retrieval system
                </p>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-6 py-12">
        <!-- Executive Summary -->
        <section class="bg-white rounded-2xl shadow-lg p-8 mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-4">The Simple Version</h2>
            <div class="prose prose-lg max-w-none text-gray-600">
                <p class="mb-4">
                    LinkedIn uses a sophisticated AI system to understand and compare member profiles.
                    Our tool replicates this system to test whether it treats people fairly regardless
                    of their name's perceived gender.
                </p>
                <p class="mb-4">
                    <strong>Think of it like this:</strong> When you search for "marketing professional in Boston,"
                    LinkedIn's AI reads thousands of profiles and ranks them by relevance. Our tool tests whether
                    two identical profiles—differing only by name—get the same "understanding" from the AI.
                </p>
                <p>
                    If they don't, that's a sign of potential bias that could affect search rankings,
                    job recommendations, and connection suggestions.
                </p>
            </div>
        </section>

        <!-- Side-by-Side Comparison -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 text-center">LinkedIn's System vs. Our Implementation</h2>

            <div class="grid md:grid-cols-2 gap-6">
                <!-- LinkedIn's System -->
                <div class="bg-white rounded-2xl shadow-lg overflow-hidden">
                    <div class="comparison-linkedin text-white p-6">
                        <h3 class="text-xl font-bold mb-2">LinkedIn's Production System</h3>
                        <p class="opacity-90 text-sm">As described in their research paper (arXiv:2510.14223)</p>
                    </div>
                    <div class="p-6 space-y-4">
                        <div class="flex items-start gap-3">
                            <span class="text-blue-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">LLaMA-3 3B Model</p>
                                <p class="text-sm text-gray-500">Custom fine-tuned on 5M member-item pairs</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-blue-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">Hidden State Extraction</p>
                                <p class="text-sm text-gray-500">Extract from last transformer layer</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-blue-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">Mean Pooling</p>
                                <p class="text-sm text-gray-500">Average all token embeddings</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-blue-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">3072-Dimensional Vectors</p>
                                <p class="text-sm text-gray-500">Dense semantic representations</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-blue-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">Cosine Similarity</p>
                                <p class="text-sm text-gray-500">For ranking and matching</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-amber-500 text-xl">★</span>
                            <div>
                                <p class="font-semibold">InfoNCE Loss Training</p>
                                <p class="text-sm text-gray-500">Specialized contrastive learning</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Our Implementation -->
                <div class="bg-white rounded-2xl shadow-lg overflow-hidden">
                    <div class="comparison-ours text-white p-6">
                        <h3 class="text-xl font-bold mb-2">Our Replication</h3>
                        <p class="opacity-90 text-sm">Open-source implementation for bias auditing</p>
                    </div>
                    <div class="p-6 space-y-4">
                        <div class="flex items-start gap-3">
                            <span class="text-green-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">LLaMA-3.2 3B Model</p>
                                <p class="text-sm text-gray-500">Base pretrained model (no fine-tuning)</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-green-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">Hidden State Extraction</p>
                                <p class="text-sm text-gray-500">Identical: output_hidden_states=True</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-green-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">Mean Pooling</p>
                                <p class="text-sm text-gray-500">Identical: hidden_states[-1].mean(dim=1)</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-green-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">3072-Dimensional Vectors</p>
                                <p class="text-sm text-gray-500">Identical model architecture</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-green-600 text-xl">✓</span>
                            <div>
                                <p class="font-semibold">Cosine Similarity</p>
                                <p class="text-sm text-gray-500">Identical comparison metric</p>
                            </div>
                        </div>
                        <div class="flex items-start gap-3">
                            <span class="text-gray-400 text-xl">○</span>
                            <div>
                                <p class="font-semibold text-gray-400">No Fine-Tuning</p>
                                <p class="text-sm text-gray-400">Tests base model bias, not production</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mt-6 bg-amber-50 border border-amber-200 rounded-xl p-6">
                <div class="flex items-start gap-4">
                    <span class="text-amber-600 text-2xl">⚠️</span>
                    <div>
                        <h4 class="font-semibold text-amber-800 mb-2">Important Difference</h4>
                        <p class="text-amber-700">
                            LinkedIn's production model is fine-tuned on 5 million member-item pairs, which
                            significantly improves retrieval accuracy. Our tool uses the base LLaMA model
                            without this fine-tuning. This means we're testing the <strong>foundational model's
                            biases</strong>, not the production system's biases. However, any bias in the base
                            model can carry through to fine-tuned versions.
                        </p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Technical Deep Dive -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">Technical Deep Dive: The Pipeline</h2>

            <!-- Visual Pipeline -->
            <div class="bg-white rounded-2xl shadow-lg p-8 mb-8">
                <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                    <!-- Step 1 -->
                    <div class="text-center flex-1">
                        <div class="step-circle bg-blue-100 text-blue-700 mx-auto mb-3">1</div>
                        <h4 class="font-semibold text-gray-900 mb-2">Input Text</h4>
                        <p class="text-sm text-gray-500">Profile name, headline, and content</p>
                    </div>

                    <div class="hidden md:block flow-arrow rotate-90 md:rotate-0"></div>

                    <!-- Step 2 -->
                    <div class="text-center flex-1">
                        <div class="step-circle bg-blue-100 text-blue-700 mx-auto mb-3">2</div>
                        <h4 class="font-semibold text-gray-900 mb-2">Tokenization</h4>
                        <p class="text-sm text-gray-500">Convert text to model tokens</p>
                    </div>

                    <div class="hidden md:block flow-arrow rotate-90 md:rotate-0"></div>

                    <!-- Step 3 -->
                    <div class="text-center flex-1">
                        <div class="step-circle bg-blue-100 text-blue-700 mx-auto mb-3">3</div>
                        <h4 class="font-semibold text-gray-900 mb-2">Transformer</h4>
                        <p class="text-sm text-gray-500">28 layers of attention processing</p>
                    </div>

                    <div class="hidden md:block flow-arrow rotate-90 md:rotate-0"></div>

                    <!-- Step 4 -->
                    <div class="text-center flex-1">
                        <div class="step-circle bg-blue-100 text-blue-700 mx-auto mb-3">4</div>
                        <h4 class="font-semibold text-gray-900 mb-2">Hidden States</h4>
                        <p class="text-sm text-gray-500">Extract from final layer</p>
                    </div>

                    <div class="hidden md:block flow-arrow rotate-90 md:rotate-0"></div>

                    <!-- Step 5 -->
                    <div class="text-center flex-1">
                        <div class="step-circle bg-green-100 text-green-700 mx-auto mb-3">5</div>
                        <h4 class="font-semibold text-gray-900 mb-2">Mean Pool</h4>
                        <p class="text-sm text-gray-500">Average to 3072-dim vector</p>
                    </div>
                </div>
            </div>

            <!-- Code Example -->
            <div class="bg-white rounded-2xl shadow-lg overflow-hidden">
                <div class="bg-gray-800 px-6 py-3 flex items-center gap-2">
                    <div class="w-3 h-3 rounded-full bg-red-500"></div>
                    <div class="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <div class="w-3 h-3 rounded-full bg-green-500"></div>
                    <span class="ml-4 text-gray-400 text-sm">embedder.py - Core embedding extraction</span>
                </div>
                <div class="code-block p-6 overflow-x-auto">
                    <pre><code><span class="text-blue-400">from</span> transformers <span class="text-blue-400">import</span> AutoModelForCausalLM, AutoTokenizer
<span class="text-blue-400">import</span> torch

<span class="text-gray-500"># Load the model (same architecture as LinkedIn)</span>
model = AutoModelForCausalLM.from_pretrained(
    <span class="text-green-400">"models/Llama-3.2-3B-bf16"</span>,
    torch_dtype=torch.bfloat16,  <span class="text-gray-500"># Match LinkedIn's precision</span>
    device_map=<span class="text-green-400">"mps"</span>           <span class="text-gray-500"># Or "cuda" for GPU</span>
)

<span class="text-blue-400">def</span> <span class="text-yellow-400">get_embedding</span>(text: str) -> list[float]:
    <span class="text-gray-500"># Step 1: Tokenize the input</span>
    inputs = tokenizer(text, return_tensors=<span class="text-green-400">"pt"</span>)

    <span class="text-gray-500"># Step 2-3: Forward pass through transformer</span>
    <span class="text-blue-400">with</span> torch.no_grad():
        outputs = model(**inputs, <span class="text-orange-400">output_hidden_states=True</span>)

    <span class="text-gray-500"># Step 4: Extract final layer hidden states</span>
    hidden_states = outputs.hidden_states[<span class="text-purple-400">-1</span>]  <span class="text-gray-500"># Shape: [1, seq_len, 3072]</span>

    <span class="text-gray-500"># Step 5: Mean pooling across all tokens</span>
    embedding = hidden_states.mean(dim=<span class="text-purple-400">1</span>).squeeze()  <span class="text-gray-500"># Shape: [3072]</span>

    <span class="text-blue-400">return</span> embedding.cpu().numpy().tolist()</code></pre>
                </div>
            </div>
        </section>

        <!-- Why Hidden States -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">Why Hidden States, Not the Embedding API?</h2>

            <div class="grid md:grid-cols-2 gap-6">
                <div class="bg-white rounded-2xl shadow-lg p-6">
                    <div class="flex items-center gap-3 mb-4">
                        <span class="text-red-500 text-2xl">✗</span>
                        <h3 class="text-lg font-semibold">Standard Embedding APIs</h3>
                    </div>
                    <ul class="space-y-3 text-gray-600">
                        <li class="flex items-start gap-2">
                            <span class="text-gray-400">•</span>
                            <span>Use specialized embedding models (BERT, E5, etc.)</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-gray-400">•</span>
                            <span>Different architecture than LLaMA</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-gray-400">•</span>
                            <span>Typically 384-1024 dimensions</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-gray-400">•</span>
                            <span>Cannot replicate LinkedIn's exact methodology</span>
                        </li>
                    </ul>
                </div>

                <div class="bg-white rounded-2xl shadow-lg p-6">
                    <div class="flex items-center gap-3 mb-4">
                        <span class="text-green-500 text-2xl">✓</span>
                        <h3 class="text-lg font-semibold">Hidden State Extraction</h3>
                    </div>
                    <ul class="space-y-3 text-gray-600">
                        <li class="flex items-start gap-2">
                            <span class="text-green-500">✓</span>
                            <span>Uses the exact same LLaMA architecture</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-green-500">✓</span>
                            <span>Matches LinkedIn's paper methodology</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-green-500">✓</span>
                            <span>3072 dimensions (same as production)</span>
                        </li>
                        <li class="flex items-start gap-2">
                            <span class="text-green-500">✓</span>
                            <span>Mean pooling replicates their approach</span>
                        </li>
                    </ul>
                </div>
            </div>
        </section>

        <!-- The Math -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">The Mathematics of Similarity</h2>

            <div class="bg-white rounded-2xl shadow-lg p-8">
                <div class="grid md:grid-cols-2 gap-8">
                    <div>
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Cosine Similarity Formula</h3>
                        <div class="bg-gray-50 rounded-xl p-6 text-center">
                            <p class="text-2xl font-mono mb-4">
                                cos(θ) = <span class="text-blue-600">A · B</span> / (<span class="text-green-600">||A||</span> × <span class="text-green-600">||B||</span>)
                            </p>
                            <div class="text-sm text-gray-500 space-y-1 text-left">
                                <p><span class="text-blue-600 font-semibold">A · B</span> = Dot product of vectors</p>
                                <p><span class="text-green-600 font-semibold">||A||, ||B||</span> = Vector magnitudes</p>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">What the Score Means</h3>
                        <div class="space-y-3">
                            <div class="flex items-center gap-4">
                                <div class="w-20 h-8 bg-green-500 rounded flex items-center justify-center text-white font-semibold">
                                    1.0
                                </div>
                                <span class="text-gray-600">Identical - profiles understood the same way</span>
                            </div>
                            <div class="flex items-center gap-4">
                                <div class="w-20 h-8 bg-green-400 rounded flex items-center justify-center text-white font-semibold">
                                    0.95+
                                </div>
                                <span class="text-gray-600">Very similar - minimal perceived difference</span>
                            </div>
                            <div class="flex items-center gap-4">
                                <div class="w-20 h-8 bg-yellow-400 rounded flex items-center justify-center text-white font-semibold">
                                    0.8-0.95
                                </div>
                                <span class="text-gray-600">Somewhat different - noticeable bias</span>
                            </div>
                            <div class="flex items-center gap-4">
                                <div class="w-20 h-8 bg-red-500 rounded flex items-center justify-center text-white font-semibold">
                                    < 0.8
                                </div>
                                <span class="text-gray-600">Very different - significant bias detected</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Research Paper Reference -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">Based on Published Research</h2>

            <div class="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-8 border border-blue-100">
                <div class="flex flex-col md:flex-row items-start gap-6">
                    <div class="bg-white rounded-xl p-4 shadow-md">
                        <svg class="w-16 h-16 text-blue-600" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm-1 2l5 5h-5V4zM6 20V4h6v6h6v10H6z"/>
                            <path d="M8 12h8v2H8zm0 4h8v2H8z"/>
                        </svg>
                    </div>
                    <div class="flex-1">
                        <h3 class="text-xl font-semibold text-gray-900 mb-2">
                            "Dual-Encoder LLM Approach for Efficient Retrieval at LinkedIn"
                        </h3>
                        <p class="text-gray-600 mb-4">
                            arXiv:2510.14223v1 · Published October 2025
                        </p>
                        <div class="bg-white rounded-lg p-4 text-sm text-gray-600">
                            <p class="mb-2"><strong>Key findings from the paper:</strong></p>
                            <ul class="space-y-1">
                                <li>• LLaMA-3 3B base achieves Recall@10 of 0.2434</li>
                                <li>• With InfoNCE fine-tuning: Recall@10 of 0.4238 (74% improvement)</li>
                                <li>• Production deployment showed +0.8% revenue lift</li>
                                <li>• Hidden state extraction with mean pooling is the core method</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Statistical Rigor -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">Statistical Rigor: Not Just Thresholds</h2>

            <div class="bg-white rounded-2xl shadow-lg p-8">
                <p class="text-gray-600 mb-6">
                    Many bias detection tools use arbitrary thresholds (e.g., "bias if score < 0.95").
                    Our approach uses proper statistical hypothesis testing:
                </p>

                <div class="grid md:grid-cols-3 gap-6">
                    <div class="bg-blue-50 rounded-xl p-6">
                        <h4 class="font-semibold text-blue-900 mb-3">P-values</h4>
                        <p class="text-sm text-blue-800">
                            Tests whether observed differences could be due to random chance.
                            P < 0.05 means results are statistically significant.
                        </p>
                    </div>

                    <div class="bg-green-50 rounded-xl p-6">
                        <h4 class="font-semibold text-green-900 mb-3">Effect Size (Cohen's d)</h4>
                        <p class="text-sm text-green-800">
                            Measures the magnitude of bias, not just its existence.
                            Tells us if the bias is negligible, small, medium, or large.
                        </p>
                    </div>

                    <div class="bg-purple-50 rounded-xl p-6">
                        <h4 class="font-semibold text-purple-900 mb-3">Confidence Intervals</h4>
                        <p class="text-sm text-purple-800">
                            Shows the range where the true bias likely falls.
                            If CI includes 1.0, we can't conclude there's bias.
                        </p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Limitations -->
        <section class="mb-12">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">Honest Limitations</h2>

            <div class="bg-white rounded-2xl shadow-lg p-8">
                <div class="space-y-6">
                    <div class="flex items-start gap-4">
                        <div class="bg-amber-100 rounded-full p-2">
                            <svg class="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                            </svg>
                        </div>
                        <div>
                            <h4 class="font-semibold text-gray-900">No Fine-Tuning Data</h4>
                            <p class="text-gray-600">
                                LinkedIn fine-tuned their model on 5 million member-item pairs. We use the base
                                LLaMA model, which means our results reflect foundational model biases, not
                                production system biases.
                            </p>
                        </div>
                    </div>

                    <div class="flex items-start gap-4">
                        <div class="bg-amber-100 rounded-full p-2">
                            <svg class="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                            </svg>
                        </div>
                        <div>
                            <h4 class="font-semibold text-gray-900">Name-Based Bias Only</h4>
                            <p class="text-gray-600">
                                We only test bias based on name changes. Other potential biases (location,
                                education, profile photo) are not captured in this analysis.
                            </p>
                        </div>
                    </div>

                    <div class="flex items-start gap-4">
                        <div class="bg-amber-100 rounded-full p-2">
                            <svg class="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                            </svg>
                        </div>
                        <div>
                            <h4 class="font-semibold text-gray-900">Simulated Profiles</h4>
                            <p class="text-gray-600">
                                Test data uses synthetic profiles, not real LinkedIn data. Results indicate
                                potential for bias but not its actual manifestation on the platform.
                            </p>
                        </div>
                    </div>

                    <div class="flex items-start gap-4">
                        <div class="bg-amber-100 rounded-full p-2">
                            <svg class="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                            </svg>
                        </div>
                        <div>
                            <h4 class="font-semibold text-gray-900">Downstream Effects Unknown</h4>
                            <p class="text-gray-600">
                                Even if similarity scores differ, we don't know how LinkedIn's ranking
                                algorithms weight these embeddings in final results.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-8">
        <div class="max-w-6xl mx-auto px-6">
            <div class="flex flex-col md:flex-row items-center justify-between gap-4">
                <div class="flex items-center gap-4">
                    <img src="https://www.trustinsights.ai/wp-content/uploads/2017/12/2021-02-23_12-58-20.jpg"
                         alt="TrustInsights.ai" class="h-12">
                    <div>
                        <p class="font-heading text-lg">TrustInsights.ai</p>
                        <p class="text-gray-400 text-sm">AI-Powered Analytics & Insights</p>
                    </div>
                </div>
                <div class="text-center md:text-right text-sm text-gray-400">
                    <p>LinkedIn Bias Auditor - Methodology Explainer</p>
                    <p class="mt-1">Based on LinkedIn's research (arXiv:2510.14223)</p>
                </div>
            </div>
        </div>
    </footer>
</body>
</html>"""


def generate_readme_explainer(output_dir: str | Path) -> str:
    """Generate a comprehensive README.htm system explainer.

    Creates a detailed HTML document explaining the entire system for both
    technical and non-technical audiences. Includes hypothesis, architecture,
    data structure, engineering decisions, and interpretation guidance.

    Args:
        output_dir: Directory where the README will be saved.

    Returns:
        Path to the generated README file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    html_content = _render_readme_explainer()

    readme_path = output_path / "README.htm"
    readme_path.write_text(html_content, encoding="utf-8")

    logger.info(f"README explainer generated: {readme_path}")
    return str(readme_path)


def _render_readme_explainer() -> str:
    """Render the comprehensive README explainer HTML.

    Returns:
        Complete HTML document as string.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LinkedIn Bias Auditor - System Documentation | TrustInsights.ai</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'ti-coral': '#F05662',
                        'ti-pink': '#FF818D',
                        'ti-blue': '#33B0EF',
                    },
                    fontFamily: {
                        'heading': ['"Barlow Condensed"', 'sans-serif'],
                        'body': ['Barlow', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <style>
        body { font-family: 'Barlow', sans-serif; }
        h1, h2, h3, h4, h5, h6 { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }
        .gradient-header { background: linear-gradient(135deg, #F05662 0%, #33B0EF 100%); }
        .diagram-box { border: 2px solid #F05662; }
        .arrow-down::after {
            content: '↓';
            display: block;
            text-align: center;
            font-size: 1.5rem;
            color: #F05662;
        }
        details summary { cursor: pointer; }
        details summary::-webkit-details-marker { display: none; }
        .code-block {
            background: #1e293b;
            color: #e2e8f0;
            font-family: 'Monaco', 'Menlo', monospace;
        }
    </style>
</head>
<body class="bg-gray-50 text-gray-800 font-body">
    <!-- Header -->
    <header class="gradient-header text-white py-12">
        <div class="max-w-6xl mx-auto px-6">
            <div class="flex items-center justify-center gap-4 mb-6">
                <img src="https://www.trustinsights.ai/wp-content/uploads/2017/12/2021-02-23_12-58-20.jpg"
                     alt="TrustInsights.ai" class="h-16">
            </div>
            <div class="text-center">
                <h1 class="text-4xl font-heading mb-4">LinkedIn Bias Auditor</h1>
                <p class="text-xl opacity-90 max-w-3xl mx-auto">
                    A comprehensive system for detecting gender bias in AI embedding models
                    used for professional profile retrieval
                </p>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-6 py-12">
        <!-- Table of Contents -->
        <nav class="bg-white rounded-xl shadow-lg p-6 mb-12">
            <h2 class="text-lg font-bold text-gray-900 mb-4">Contents</h2>
            <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                <a href="#overview" class="text-blue-600 hover:underline">1. Executive Overview</a>
                <a href="#hypothesis" class="text-blue-600 hover:underline">2. The Hypothesis</a>
                <a href="#architecture" class="text-blue-600 hover:underline">3. System Architecture</a>
                <a href="#data" class="text-blue-600 hover:underline">4. The Data</a>
                <a href="#engineering" class="text-blue-600 hover:underline">5. Engineering Details</a>
                <a href="#statistics" class="text-blue-600 hover:underline">6. Statistical Methods</a>
                <a href="#interpretation" class="text-blue-600 hover:underline">7. Interpreting Results</a>
                <a href="#limitations" class="text-blue-600 hover:underline">8. Limitations</a>
                <a href="#running" class="text-blue-600 hover:underline">9. How to Run</a>
                <a href="#outputs" class="text-blue-600 hover:underline">10. Output Files</a>
            </div>
        </nav>

        <!-- 1. Executive Overview -->
        <section id="overview" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">1</span>
                Executive Overview (Non-Technical)
            </h2>

            <div class="space-y-6 text-gray-600">
                <div>
                    <h3 class="font-semibold text-gray-800 mb-2">What is this?</h3>
                    <p>A tool to test AI systems for gender bias. Specifically, we test whether
                    LinkedIn's underlying AI model treats identical professional profiles differently
                    based solely on whether the name appears male or female.</p>
                </div>

                <div>
                    <h3 class="font-semibold text-gray-800 mb-2">Why does it matter?</h3>
                    <p>LinkedIn uses AI to show you profiles, job recommendations, and content.
                    If the AI treats people differently based on their name, that's bias—and it
                    could affect who gets seen for jobs, whose content gets promoted, and who
                    connects with whom.</p>
                </div>

                <div>
                    <h3 class="font-semibold text-gray-800 mb-2">What do we test?</h3>
                    <p>We show the AI identical professional profiles with only the name changed
                    (one male-coded, one female-coded) and measure if it "sees" them differently.
                    If there's no bias, identical content should produce identical AI representations.</p>
                </div>

                <div class="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg">
                    <p class="font-semibold text-blue-800">Key Insight</p>
                    <p class="text-blue-700">If the AI perceives "Bob Smith, Marketing Director"
                    differently from "Barbara Smith, Marketing Director" (with identical job descriptions),
                    that's evidence of gender bias in the model.</p>
                </div>

                <div class="bg-amber-50 border-l-4 border-amber-400 p-4 rounded-r-lg mt-6">
                    <p class="font-semibold text-amber-800">The Pizza Analogy</p>
                    <p class="text-amber-700">Think of it like ordering the same pizza from the same parlor twice.
                    If both pizzas are identical, that's a similarity score of 100%. If one pizza is slightly
                    different (95% similar), you might shrug it off. But if <em>every</em> pizza you order
                    is 5% smaller than advertised, that's not an accident—that's a pattern.</p>
                    <p class="text-amber-700 mt-2">This tool helps detect whether the AI consistently treats
                    female-named profiles slightly differently than male-named ones—a pattern that may seem
                    small individually but adds up to meaningful bias.</p>
                </div>
            </div>
        </section>

        <!-- 2. The Hypothesis -->
        <section id="hypothesis" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">2</span>
                The Hypothesis
            </h2>

            <div class="grid md:grid-cols-2 gap-6 mb-6">
                <div class="bg-green-50 border border-green-200 rounded-xl p-6">
                    <h3 class="font-bold text-green-800 mb-2">Null Hypothesis (H₀)</h3>
                    <p class="text-green-700">The AI model treats male and female names identically
                    when processing professional profiles.</p>
                    <p class="mt-2 text-sm text-green-600">Expected: Cosine similarity = 1.0</p>
                </div>

                <div class="bg-red-50 border border-red-200 rounded-xl p-6">
                    <h3 class="font-bold text-red-800 mb-2">Alternative Hypothesis (H₁)</h3>
                    <p class="text-red-700">The AI model creates systematically different
                    representations based on gendered names.</p>
                    <p class="mt-2 text-sm text-red-600">Evidence: Cosine similarity &lt; 1.0</p>
                </div>
            </div>

            <div class="bg-gray-50 rounded-xl p-6">
                <h3 class="font-semibold text-gray-800 mb-3">Why 1.0 is the target</h3>
                <p class="text-gray-600">If there's no bias, identical profiles (differing only by name)
                should produce identical embeddings. In vector space, identical embeddings have a
                cosine similarity of exactly 1.0. Any value below 1.0 indicates the model perceives
                some difference—and since the only difference is the gendered name, that difference
                is gender-based.</p>
            </div>
        </section>

        <!-- 3. System Architecture -->
        <section id="architecture" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">3</span>
                System Architecture
            </h2>

            <!-- Visual Diagram -->
            <div class="bg-gray-50 rounded-xl p-6 mb-6 overflow-x-auto">
                <div class="min-w-[600px]">
                    <!-- Row 1: Data Sources -->
                    <div class="flex justify-center gap-8 mb-4">
                        <div class="diagram-box bg-white rounded-lg p-4 text-center w-40">
                            <p class="font-bold text-blue-700">Male Data</p>
                            <p class="text-sm text-gray-500">maledata.json</p>
                        </div>
                        <div class="diagram-box bg-white rounded-lg p-4 text-center w-40">
                            <p class="font-bold text-pink-600">Female Data</p>
                            <p class="text-sm text-gray-500">femaledata.json</p>
                        </div>
                    </div>

                    <div class="arrow-down"></div>

                    <!-- Row 2: Validation -->
                    <div class="flex justify-center mb-4">
                        <div class="diagram-box bg-white rounded-lg p-4 text-center w-72">
                            <p class="font-bold text-gray-700">Pair Validation</p>
                            <p class="text-sm text-gray-500">Verify identical content (except name)</p>
                        </div>
                    </div>

                    <div class="arrow-down"></div>

                    <!-- Row 3: Embedding -->
                    <div class="flex justify-center mb-4">
                        <div class="diagram-box bg-blue-100 rounded-lg p-4 text-center w-80">
                            <p class="font-bold text-blue-700">LLaMA-3.2 Model</p>
                            <p class="text-sm text-gray-600">Hidden state extraction + mean pooling</p>
                            <p class="text-xs text-gray-500">Output: 3072-dimensional vectors</p>
                        </div>
                    </div>

                    <div class="arrow-down"></div>

                    <!-- Row 4: Comparison -->
                    <div class="flex justify-center mb-4">
                        <div class="diagram-box bg-white rounded-lg p-4 text-center w-72">
                            <p class="font-bold text-gray-700">Cosine Similarity</p>
                            <p class="text-sm text-gray-500">cos(θ) = (A·B) / (||A|| × ||B||)</p>
                        </div>
                    </div>

                    <div class="arrow-down"></div>

                    <!-- Row 5: Analysis -->
                    <div class="flex justify-center mb-4">
                        <div class="diagram-box bg-green-100 rounded-lg p-4 text-center w-96">
                            <p class="font-bold text-green-700">Statistical Analysis</p>
                            <p class="text-sm text-gray-600">Bootstrap CI • Permutation test • Wilcoxon</p>
                            <p class="text-sm text-gray-600">Cohen's d • WEAT effect size • Power analysis</p>
                        </div>
                    </div>

                    <div class="arrow-down"></div>

                    <!-- Row 6: Outputs -->
                    <div class="flex justify-center gap-4">
                        <div class="diagram-box bg-white rounded-lg p-3 text-center w-36">
                            <p class="font-bold text-gray-700">analysis.htm</p>
                            <p class="text-xs text-gray-500">Results dashboard</p>
                        </div>
                        <div class="diagram-box bg-white rounded-lg p-3 text-center w-36">
                            <p class="font-bold text-gray-700">README.htm</p>
                            <p class="text-xs text-gray-500">This document</p>
                        </div>
                        <div class="diagram-box bg-white rounded-lg p-3 text-center w-36">
                            <p class="font-bold text-gray-700">audit.db</p>
                            <p class="text-xs text-gray-500">SQLite database</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 4. The Data -->
        <section id="data" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">4</span>
                The Data
            </h2>

            <div class="space-y-6">
                <div>
                    <h3 class="font-semibold text-gray-800 mb-2">Source</h3>
                    <p class="text-gray-600">Identical LinkedIn-style posts with name variations.
                    Two JSON files contain matched pairs of profiles.</p>
                </div>

                <div>
                    <h3 class="font-semibold text-gray-800 mb-2">Structure</h3>
                    <p class="text-gray-600">Each pair has matching <code class="bg-gray-100 px-1 rounded">headline</code>
                    and <code class="bg-gray-100 px-1 rounded">text_content</code>—only the
                    <code class="bg-gray-100 px-1 rounded">name</code> differs.</p>
                </div>

                <div class="bg-gray-50 rounded-xl p-6">
                    <h3 class="font-semibold text-gray-800 mb-4">Example Pair</h3>
                    <div class="grid md:grid-cols-2 gap-4">
                        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                            <p class="font-bold text-blue-700 mb-2">Male Version</p>
                            <p class="text-sm"><strong>Name:</strong> Bob Hutchins</p>
                            <p class="text-sm"><strong>Headline:</strong> Marketing Director</p>
                            <p class="text-sm"><strong>Content:</strong> I believe AI will transform...</p>
                        </div>
                        <div class="bg-pink-50 border border-pink-200 rounded-lg p-4">
                            <p class="font-bold text-pink-700 mb-2">Female Version</p>
                            <p class="text-sm"><strong>Name:</strong> Bobbi Hutchins</p>
                            <p class="text-sm"><strong>Headline:</strong> Marketing Director</p>
                            <p class="text-sm"><strong>Content:</strong> I believe AI will transform...</p>
                        </div>
                    </div>
                    <p class="text-sm text-gray-500 mt-4 text-center">
                        ↑ The headline and content are <strong>identical</strong>. Only the name changes.
                    </p>
                </div>
            </div>
        </section>

        <!-- 5. Engineering Details -->
        <section id="engineering" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">5</span>
                Engineering Details (Technical)
            </h2>

            <details class="mb-6">
                <summary class="bg-gray-50 rounded-lg p-4 font-semibold text-gray-800 hover:bg-gray-100">
                    Why Hidden States (not Embedding API)?
                </summary>
                <div class="p-4 border-l-2 border-blue-200 ml-4 mt-2 text-gray-600">
                    <p class="mb-3">LinkedIn's research paper (arXiv:2510.14223) describes extracting
                    embeddings from LLaMA's hidden states, not using a separate embedding API.
                    We replicate this exactly:</p>
                    <div class="code-block rounded-lg p-4 text-sm overflow-x-auto">
                        <pre>outputs = model(**inputs, output_hidden_states=True)
embedding = outputs.hidden_states[-1].mean(dim=1)  # Mean pooling</pre>
                    </div>
                </div>
            </details>

            <details class="mb-6">
                <summary class="bg-gray-50 rounded-lg p-4 font-semibold text-gray-800 hover:bg-gray-100">
                    Why Mean Pooling?
                </summary>
                <div class="p-4 border-l-2 border-blue-200 ml-4 mt-2 text-gray-600">
                    <p>The paper specifies averaging across all tokens to create a single vector
                    representing the entire text. This produces a 3072-dimensional vector that
                    captures the semantic meaning of the complete profile.</p>
                </div>
            </details>

            <details class="mb-6">
                <summary class="bg-gray-50 rounded-lg p-4 font-semibold text-gray-800 hover:bg-gray-100">
                    Why Cosine Similarity?
                </summary>
                <div class="p-4 border-l-2 border-blue-200 ml-4 mt-2 text-gray-600">
                    <p class="mb-3">Cosine similarity is the standard metric for comparing embedding
                    vectors. It measures the angle between two vectors:</p>
                    <p class="font-mono text-center text-lg mb-3">cos(θ) = (A·B) / (||A|| × ||B||)</p>
                    <p>Range: -1 to 1 (we typically see 0 to 1 for same-direction vectors).
                    A value of 1.0 means the vectors are identical.</p>
                </div>
            </details>

            <details>
                <summary class="bg-gray-50 rounded-lg p-4 font-semibold text-gray-800 hover:bg-gray-100">
                    Model Specifications
                </summary>
                <div class="p-4 border-l-2 border-blue-200 ml-4 mt-2">
                    <table class="w-full text-sm">
                        <tr class="border-b">
                            <td class="py-2 text-gray-600">Model</td>
                            <td class="py-2 font-mono">LLaMA-3.2 3B</td>
                        </tr>
                        <tr class="border-b">
                            <td class="py-2 text-gray-600">Precision</td>
                            <td class="py-2 font-mono">bfloat16</td>
                        </tr>
                        <tr class="border-b">
                            <td class="py-2 text-gray-600">Hidden Size</td>
                            <td class="py-2 font-mono">3072</td>
                        </tr>
                        <tr class="border-b">
                            <td class="py-2 text-gray-600">Layers</td>
                            <td class="py-2 font-mono">28</td>
                        </tr>
                        <tr>
                            <td class="py-2 text-gray-600">Extraction</td>
                            <td class="py-2 font-mono">Last layer + mean pool</td>
                        </tr>
                    </table>
                </div>
            </details>
        </section>

        <!-- 6. Statistical Methods -->
        <section id="statistics" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">6</span>
                Statistical Methods (Why Multiple Tests?)
            </h2>

            <p class="text-gray-600 mb-6">We use multiple statistical tests because different tests
            have different assumptions. Convergent evidence from multiple tests is stronger than
            any single test—and reviewers will ask about robustness.</p>

            <div class="overflow-x-auto mb-6">
                <table class="w-full text-sm">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-3 text-left font-semibold">Test</th>
                            <th class="px-4 py-3 text-left font-semibold">Purpose</th>
                            <th class="px-4 py-3 text-left font-semibold">Assumption</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y">
                        <tr>
                            <td class="px-4 py-3 font-medium">t-test</td>
                            <td class="px-4 py-3 text-gray-600">Traditional parametric test</td>
                            <td class="px-4 py-3 text-gray-600">Assumes normality</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-medium">Permutation test</td>
                            <td class="px-4 py-3 text-gray-600">Distribution-free significance</td>
                            <td class="px-4 py-3 text-gray-600">None (gold standard for WEAT)</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-medium">Wilcoxon</td>
                            <td class="px-4 py-3 text-gray-600">Non-parametric backup</td>
                            <td class="px-4 py-3 text-gray-600">Symmetric distribution</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-medium">Bootstrap CI</td>
                            <td class="px-4 py-3 text-gray-600">Robust confidence interval</td>
                            <td class="px-4 py-3 text-gray-600">None</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-medium">Shapiro-Wilk</td>
                            <td class="px-4 py-3 text-gray-600">Tests normality assumption</td>
                            <td class="px-4 py-3 text-gray-600">Validates t-test use</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="grid md:grid-cols-2 gap-6">
                <div class="bg-blue-50 rounded-xl p-6">
                    <h3 class="font-semibold text-blue-800 mb-2">Cohen's d</h3>
                    <p class="text-blue-700 text-sm">Measures effect size (magnitude of bias).
                    Unlike p-values, tells us if the bias is practically significant.</p>
                    <p class="mt-2 text-xs text-blue-600">
                        &lt;0.2: negligible | 0.2-0.5: small | 0.5-0.8: medium | &gt;0.8: large
                    </p>
                </div>
                <div class="bg-green-50 rounded-xl p-6">
                    <h3 class="font-semibold text-green-800 mb-2">WEAT Effect Size</h3>
                    <p class="text-green-700 text-sm">Standard metric in embedding bias literature.
                    Makes results directly comparable to published research on AI bias.</p>
                </div>
            </div>
        </section>

        <!-- 7. Interpreting Results -->
        <section id="interpretation" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">7</span>
                Interpreting Results
            </h2>

            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-3 text-left font-semibold">Metric</th>
                            <th class="px-4 py-3 text-center font-semibold text-green-700">Good (No Bias)</th>
                            <th class="px-4 py-3 text-center font-semibold text-yellow-700">Concerning</th>
                            <th class="px-4 py-3 text-center font-semibold text-red-700">Severe</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y">
                        <tr>
                            <td class="px-4 py-3 font-medium">Mean Similarity</td>
                            <td class="px-4 py-3 text-center bg-green-50">&gt; 0.99</td>
                            <td class="px-4 py-3 text-center bg-yellow-50">0.90 - 0.99</td>
                            <td class="px-4 py-3 text-center bg-red-50">&lt; 0.90</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-medium">Cohen's d</td>
                            <td class="px-4 py-3 text-center bg-green-50">&lt; 0.2</td>
                            <td class="px-4 py-3 text-center bg-yellow-50">0.2 - 0.8</td>
                            <td class="px-4 py-3 text-center bg-red-50">&gt; 0.8</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-medium">P-value</td>
                            <td class="px-4 py-3 text-center bg-green-50">&gt; 0.05</td>
                            <td class="px-4 py-3 text-center bg-yellow-50">&lt; 0.05</td>
                            <td class="px-4 py-3 text-center bg-red-50">&lt; 0.001</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Individual vs Aggregate Explanation -->
            <div class="mt-6 p-6 bg-amber-50 border border-amber-200 rounded-xl">
                <h3 class="font-bold text-amber-800 mb-3">Why Individual Results Look Fine But Overall Shows Bias</h3>
                <p class="text-amber-700 mb-4">
                    You may notice that <strong>individual pairs</strong> show "Small Deviation" or "Minimal Deviation"
                    while the <strong>overall verdict</strong> warns of "Large Bias Detected." This isn't a contradiction—it's
                    statistics revealing a pattern.
                </p>
                <div class="bg-white rounded-lg p-4 border border-amber-200">
                    <p class="text-amber-800 font-medium mb-2">The Pizza Parlor Analogy:</p>
                    <p class="text-gray-700 mb-2">
                        Imagine ordering 76 pizzas from a parlor. Each pizza is 95% the size of what you ordered.
                        Individually, you might think "close enough" for each one.
                    </p>
                    <p class="text-gray-700 mb-2">
                        But collectively? That 5% shortfall on 76 pizzas adds up to almost <strong>4 entire pizzas</strong>
                        worth of missing pizza. One undersized pizza is an accident. Seventy-six consistently
                        undersized pizzas is a policy.
                    </p>
                    <p class="text-amber-700 mt-3 italic">
                        Cohen's d measures this consistency. A "Large" effect size means the pattern is consistent
                        and meaningful—even if each individual deviation seemed minor.
                    </p>
                </div>
            </div>
        </section>

        <!-- 8. Limitations -->
        <section id="limitations" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">8</span>
                Limitations
            </h2>

            <div class="space-y-4">
                <div class="flex items-start gap-4 p-4 bg-amber-50 rounded-lg">
                    <span class="text-amber-600 text-xl">⚠️</span>
                    <div>
                        <h4 class="font-semibold text-amber-800">Base model only</h4>
                        <p class="text-amber-700 text-sm">We test LLaMA-3.2 base, not LinkedIn's
                        fine-tuned production model. Results indicate foundational model bias,
                        not production system behavior.</p>
                    </div>
                </div>

                <div class="flex items-start gap-4 p-4 bg-amber-50 rounded-lg">
                    <span class="text-amber-600 text-xl">⚠️</span>
                    <div>
                        <h4 class="font-semibold text-amber-800">Name bias only</h4>
                        <p class="text-amber-700 text-sm">Other biases (photos, locations, education)
                        are not tested. Name is the only variable we manipulate.</p>
                    </div>
                </div>

                <div class="flex items-start gap-4 p-4 bg-amber-50 rounded-lg">
                    <span class="text-amber-600 text-xl">⚠️</span>
                    <div>
                        <h4 class="font-semibold text-amber-800">Synthetic data</h4>
                        <p class="text-amber-700 text-sm">Test profiles are synthetic, not real
                        LinkedIn profiles. Real-world behavior may differ.</p>
                    </div>
                </div>

                <div class="flex items-start gap-4 p-4 bg-amber-50 rounded-lg">
                    <span class="text-amber-600 text-xl">⚠️</span>
                    <div>
                        <h4 class="font-semibold text-amber-800">Downstream effects unknown</h4>
                        <p class="text-amber-700 text-sm">LinkedIn's ranking layers may amplify or
                        reduce embedding bias. We test embeddings, not final user outcomes.</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- 9. How to Run -->
        <section id="running" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">9</span>
                How to Run
            </h2>

            <div class="code-block rounded-xl p-6 space-y-4">
                <div>
                    <p class="text-gray-400 text-sm mb-1"># Install dependencies</p>
                    <p class="text-green-400">poetry install</p>
                </div>

                <div>
                    <p class="text-gray-400 text-sm mb-1"># Run full audit</p>
                    <p class="text-green-400">poetry run python -m src.main audit</p>
                </div>

                <div>
                    <p class="text-gray-400 text-sm mb-1"># Run validation only (no model inference)</p>
                    <p class="text-green-400">poetry run python -m src.main audit --simulate</p>
                </div>
            </div>
        </section>

        <!-- 10. Output Files -->
        <section id="outputs" class="bg-white rounded-xl shadow-lg p-8 mb-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                <span class="bg-blue-100 text-blue-700 rounded-full w-10 h-10 flex items-center justify-center font-bold">10</span>
                Output Files
            </h2>

            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-3 text-left font-semibold">File</th>
                            <th class="px-4 py-3 text-left font-semibold">Description</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y">
                        <tr>
                            <td class="px-4 py-3 font-mono text-blue-600">output/analysis/analysis.htm</td>
                            <td class="px-4 py-3 text-gray-600">Interactive dashboard with charts, tables, and toggle views</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-mono text-blue-600">output/analysis/methodology_explainer.htm</td>
                            <td class="px-4 py-3 text-gray-600">Technical deep-dive on LinkedIn replication methodology</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-mono text-blue-600">output/README.htm</td>
                            <td class="px-4 py-3 text-gray-600">This comprehensive system explainer</td>
                        </tr>
                        <tr>
                            <td class="px-4 py-3 font-mono text-blue-600">output/audit.db</td>
                            <td class="px-4 py-3 text-gray-600">SQLite database with all raw results</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-8 mt-12">
        <div class="max-w-6xl mx-auto px-6">
            <div class="flex flex-col md:flex-row items-center justify-between gap-4">
                <div class="flex items-center gap-4">
                    <img src="https://www.trustinsights.ai/wp-content/uploads/2017/12/2021-02-23_12-58-20.jpg"
                         alt="TrustInsights.ai" class="h-12">
                    <div>
                        <p class="font-heading text-lg">TrustInsights.ai</p>
                        <p class="text-gray-400 text-sm">AI-Powered Analytics & Insights</p>
                    </div>
                </div>
                <div class="text-center md:text-right text-sm text-gray-400">
                    <p>LinkedIn Bias Auditor - System Documentation</p>
                    <p class="mt-1">Built with Python, Hugging Face Transformers, and statistical rigor.</p>
                </div>
            </div>
        </div>
    </footer>
</body>
</html>"""


# end src/report.py
