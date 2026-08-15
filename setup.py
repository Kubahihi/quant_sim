from pathlib import Path

from setuptools import find_packages, setup


def _locked_requirements() -> list[str]:
    """Use the same exact dependency set as the deployed Streamlit app."""
    manifest = Path(__file__).with_name("requirements.txt")
    return [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

setup(
    name="quant_platform",
    version="0.2.0",
    packages=find_packages(where=".", include=["src", "src.*"]),
    install_requires=_locked_requirements(),
    python_requires=">=3.12,<3.13",
)
