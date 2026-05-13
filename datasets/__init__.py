"""
Make the local `datasets` directory a proper Python package.

This avoids import collisions with third-party packages that also use the name
`datasets` (e.g., HuggingFace `datasets`).
"""

