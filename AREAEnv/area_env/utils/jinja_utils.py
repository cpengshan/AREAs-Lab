"""Jinja2 template rendering utilities."""

import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def render_template(template_path: str, **kwargs) -> str:
    """Render a Jinja2 template from an absolute file path.

    Args:
        template_path: Absolute path to the .jinja template file.
        **kwargs: Variables passed to the template.

    Returns:
        Rendered string.
    """
    template_dir = os.path.dirname(template_path)
    template_name = os.path.basename(template_path)
    env = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
    )
    template = env.get_template(template_name)
    return template.render(**kwargs)
