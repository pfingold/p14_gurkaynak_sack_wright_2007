"""
Build aggregate HTML pages for chartbook replication figures.

Outputs (in docs/charts):
- replication_method_curves.html
- replication_gsw_overlays.html
- replication_correlation_heatmaps.html
"""

import base64
from pathlib import Path

from settings import config

BASE_DIR = Path(config("BASE_DIR"))
OUTPUT_DIR = BASE_DIR / "_output"
DOCS_CHARTS_DIR = BASE_DIR / "docs" / "charts"
DOCS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def _inline_png_data_uri(filename):
    """Read a PNG from _output and return a base64 data URI."""
    png_path = OUTPUT_DIR / filename
    if not png_path.exists():
        raise FileNotFoundError(f"Missing required chart PNG: {png_path}")
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _page_html(title, subtitle, sections):
    """Create a simple standalone HTML page that embeds chart PNGs."""
    parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'/>",
        f"  <title>{title}</title>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; line-height: 1.5; }",
        "    h1 { margin-bottom: 0.25rem; }",
        "    .subtitle { color: #555; margin-top: 0; margin-bottom: 1.25rem; }",
        "    .section { margin: 1.5rem 0 2rem; }",
        "    .section h2 { margin-bottom: 0.5rem; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }",
        "    .card { border: 1px solid #ddd; border-radius: 10px; padding: 10px; background: #fff; }",
        "    .card h3 { font-size: 0.96rem; margin: 0 0 8px; }",
        "    .chart-img { width: 100%; height: auto; border: 0; background: #fff; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
        f"  <p class='subtitle'>{subtitle}</p>",
    ]

    for section in sections:
        parts.extend(
            [
                "  <div class='section'>",
                f"    <h2>{section['title']}</h2>",
                "    <div class='grid'>",
            ]
        )
        for item in section["items"]:
            src = _inline_png_data_uri(item["path"])
            parts.extend(
                [
                    "      <div class='card'>",
                    f"        <h3>{item['title']}</h3>",
                    f"        <img class='chart-img' src='{src}' alt='{item['title']}' />",
                    "      </div>",
                ]
            )
        parts.extend(["    </div>", "  </div>"])

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def _write_page(filename, title, subtitle, sections):
    """Write one aggregate chart page to docs/charts."""
    html = _page_html(title=title, subtitle=subtitle, sections=sections)
    out_path = DOCS_CHARTS_DIR / filename
    out_path.write_text(html, encoding="utf-8")
    return out_path


def build_method_curves_page():
    """Build the page with method-specific discount/spot/forward selected-date charts."""
    sections = [
        {
            "title": "McCulloch Curves",
            "items": [
                {"title": "Discount Curves (Selected Dates)", "path": "mcc_discount_selected_dates.png"},
                {"title": "Spot Curves (Selected Dates)", "path": "mcc_spot_cc_selected_dates.png"},
                {"title": "Forward Curves (Selected Dates)", "path": "mcc_fwd_instant_cc_selected_dates.png"},
            ],
        },
        {
            "title": "Fisher Curves",
            "items": [
                {"title": "Discount Curves (Selected Dates)", "path": "fisher_discount_selected_dates.png"},
                {"title": "Spot Curves (Selected Dates)", "path": "fisher_spot_cc_selected_dates.png"},
                {"title": "Forward Curves (Selected Dates)", "path": "fisher_fwd_instant_cc_selected_dates.png"},
            ],
        },
        {
            "title": "Waggoner Curves",
            "items": [
                {"title": "Discount Curves (Selected Dates)", "path": "waggoner_discount_selected_dates.png"},
                {"title": "Spot Curves (Selected Dates)", "path": "waggoner_spot_cc_selected_dates.png"},
                {"title": "Forward Curves (Selected Dates)", "path": "waggoner_fwd_instant_cc_selected_dates.png"},
            ],
        },
    ]
    return _write_page(
        filename="replication_method_curves.html",
        title="Replication Method Curves",
        subtitle="Discount, spot, and forward curve views for each spline method.",
        sections=sections,
    )


def build_gsw_overlay_page():
    """Build the page with low/median/high-correlation overlays (GSW + spline methods)."""
    sections = [
        {
            "title": "Low Correlation Date",
            "items": [
                {"title": "Discount Overlay", "path": "methods_vs_gsw_low_corr_discount.png"},
                {"title": "Spot Overlay", "path": "methods_vs_gsw_low_corr_spot_cc.png"},
                {"title": "Forward Overlay", "path": "methods_vs_gsw_low_corr_fwd_instant_cc.png"},
            ],
        },
        {
            "title": "Median Correlation Date",
            "items": [
                {"title": "Discount Overlay", "path": "methods_vs_gsw_median_corr_discount.png"},
                {"title": "Spot Overlay", "path": "methods_vs_gsw_median_corr_spot_cc.png"},
                {"title": "Forward Overlay", "path": "methods_vs_gsw_median_corr_fwd_instant_cc.png"},
            ],
        },
        {
            "title": "High Correlation Date",
            "items": [
                {"title": "Discount Overlay", "path": "methods_vs_gsw_high_corr_discount.png"},
                {"title": "Spot Overlay", "path": "methods_vs_gsw_high_corr_spot_cc.png"},
                {"title": "Forward Overlay", "path": "methods_vs_gsw_high_corr_fwd_instant_cc.png"},
            ],
        },
    ]
    return _write_page(
        filename="replication_gsw_overlays.html",
        title="GSW vs Spline Method Overlays",
        subtitle="Low, median, and high correlation dates for visual model comparison.",
        sections=sections,
    )


def build_correlation_heatmaps_page():
    """Build the page with pairwise method correlation heatmaps."""
    sections = [
        {
            "title": "Pairwise Correlation Heatmaps",
            "items": [
                {"title": "Spot Curve Correlations", "path": "method_corr_heatmap_spot_cc.png"},
                {"title": "Forward Curve Correlations", "path": "method_corr_heatmap_forward_instant_cc.png"},
            ],
        }
    ]
    return _write_page(
        filename="replication_correlation_heatmaps.html",
        title="Spline Method Correlation Heatmaps",
        subtitle="Pairwise method-vs-method correlation diagnostics across curve representations.",
        sections=sections,
    )


def main():
    """Run all aggregate chartbook page builders."""
    outputs = [
        build_method_curves_page(),
        build_gsw_overlay_page(),
        build_correlation_heatmaps_page(),
    ]
    for p in outputs:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
