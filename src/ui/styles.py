"""
Minimalist UI Styling.
"""

THEME_CONFIG = {
    "environment_welfare": {"name": "Environment Agent", "icon": "E", "accent": "#000000"},
    "feeding_nutrition": {"name": "Feeding Agent", "icon": "F", "accent": "#000000"},
    "production": {"name": "Production Agent", "icon": "P", "accent": "#000000"},
    "health_behavior": {"name": "Health Agent", "icon": "H", "accent": "#000000"},
    "multi_agent": {"name": "Multi-Agent System", "icon": "M", "accent": "#000000"}
}

STATUS_COLORS = {
    "supports": {"bg": "#ffffff", "border": "#000000", "text": "#000000", "label": "Supports Hypothesis"},
    "does_not_support": {"bg": "#ffffff", "border": "#000000", "text": "#000000", "label": "Does Not Support"},
    "inconclusive": {"bg": "#ffffff", "border": "#000000", "text": "#000000", "label": "Inconclusive"},
    "no_data": {"bg": "#ffffff", "border": "#000000", "text": "#000000", "label": "No Data"},
}

def get_custom_css(agent_key: str = "environment_welfare") -> str:
    return """
    <style>
    /* Absolute minimal light theme */
    html, body, [class*="css"] {
        font-family: sans-serif !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Remove padding and margins where possible */
    .block-container {
        padding: 1rem !important;
        max-width: 1000px;
    }

    .agent-header {
        border-bottom: 1px solid #000000;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
    }

    .agent-title {
        font-size: 1.2rem;
        font-weight: normal;
    }

    .status-pill {
        border: 1px solid #000;
        padding: 2px 5px;
        font-size: 0.8rem;
    }

    .verdict-card {
        border: 1px solid #000000 !important;
        background-color: #ffffff !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }

    .verdict-headline {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .verdict-badges {
        margin-top: 0.5rem;
    }

    .badge {
        border: 1px solid #000;
        padding: 2px 5px;
        font-size: 0.8rem;
        margin-right: 5px;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Plain buttons */
    div.stButton > button:first-child {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }

    div.stButton > button:first-child:hover {
        background-color: #f0f0f0 !important;
    }

    /* Plain Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #000000 !important;
    }
    
    /* Remove headers background */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
    """
