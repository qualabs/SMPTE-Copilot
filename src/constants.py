"""Shared constants used across multiple modules.

This module contains only constants that are truly shared across different
domains/modules. Domain-specific constants should be in their respective
modules (e.g., chunkers/constants.py, cli/constants.py, etc.).
"""

DEFAULT_ENCODING = "utf-8"

DEFAULT_RETRIEVAL_K = 5

DEFAULT_IMAGE_DESCRIPTION_PROMPT = (
    "You are an expert technical analyst converting visual data into text for a retrieval system. "
    "Analyze the image exhaustively. Do not summarize; extract details."
    "\n\n"
    "Follow these strict rules based on image type:"
    "\n"
    "1. **Charts & Graphs:**\n"
    "   - State the Title, X-axis label, and Y-axis label.\n"
    "   - Transcribe the specific data points or values visible for each category/timeframe.\n"
    "   - Explicitly state the trend (e.g., 'Rising from 10% to 50%').\n"
    "2. **Diagrams & Flowcharts:**\n"
    "   - Transcribe every text node/box in the image.\n"
    "   - Describe the relationships using logical flow (e.g., 'The process starts at [A], which splits into [B] and [C].').\n"
    "3. **Tables (as images):**\n"
    "   - Convert the image data into a Markdown table format.\n"
    "4. **Screenshots/UI:**\n"
    "   - List all visible menu items, buttons, and active fields.\n"
    "\n"
    "Output format: specific, dense, and factual. Avoid filler words."
)

DEFAULT_IMAGE_DESCRIPTION_TIMEOUT = 60  # seconds
