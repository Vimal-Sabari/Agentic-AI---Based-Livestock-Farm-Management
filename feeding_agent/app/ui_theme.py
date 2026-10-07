"""
Self-contained UI styling and color constants for Feeding & Nutrition Agent dashboard.
"""
from __future__ import annotations

COLOR_PRIMARY = "#d97706"  # Amber accent
COLOR_PRIMARY_HOVER = "#b45309"
COLOR_PRIMARY_LIGHT = "#fef3c7"

COLOR_GREEN = "#16a34a"  # does_not_support / normal
COLOR_AMBER = "#f59e0b"  # inconclusive / mild
COLOR_RED = "#dc2626"    # supports / strong
COLOR_GREY = "#9ca3af"   # no data / neutral

CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .agent-header {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .verdict-card {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 6px solid #d97706;
        background-color: #fffbf0;
    }

    .verdict-supports {
        border-left-color: #dc2626;
        background-color: #fef2f2;
    }

    .verdict-does-not-support {
        border-left-color: #16a34a;
        background-color: #f0fdf4;
    }

    .verdict-inconclusive {
        border-left-color: #f59e0b;
        background-color: #fffbeb;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-supports {
        background-color: #fee2e2;
        color: #991b1b;
    }

    .badge-does-not-support {
        background-color: #dcfce7;
        color: #166534;
    }

    .badge-inconclusive {
        background-color: #fef3c7;
        color: #92400e;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
</style>
"""
