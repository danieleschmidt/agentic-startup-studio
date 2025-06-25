from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import xml.etree.ElementTree as ET

import coverage
import pytest


def run_pytest_with_coverage(
    test_dirs: Iterable[str | Path] | str | Path,
) -> Tuple[int, Path]:
    """Run pytest with coverage and return exit code and coverage xml path."""
    if isinstance(test_dirs, (str, Path)):
        test_args = [str(Path(test_dirs))]
    else:
        test_args = [str(Path(d)) for d in test_dirs]
    cov_config = Path(".coveragerc") if Path(".coveragerc").exists() else None
    cov = coverage.Coverage(config_file=str(cov_config) if cov_config else False)
    cov.start()
    exit_code = pytest.main(test_args)
    cov.stop()
    cov.save()
    xml_path = Path("coverage.xml")
    try:
        cov.xml_report(outfile=str(xml_path))
        cov.html_report(directory="htmlcov")
    except coverage.exceptions.NoDataError:
        pass
    return exit_code, xml_path


def check_coverage_threshold(xml_path: Path, min_percentage: float = 80.0) -> float:
    """Verify coverage meets the given threshold.

    Parameters
    ----------
    xml_path : Path
        Path to a coverage XML report.
    min_percentage : float
        Minimum acceptable coverage percentage.

    Returns
    -------
    float
        The measured coverage percentage.

    Raises
    ------
    ValueError
        If the coverage is below the required threshold.
    """
    root = ET.parse(xml_path).getroot()
    line_rate = float(root.attrib.get("line-rate", 0))
    percentage = line_rate * 100
    if percentage < min_percentage:
        raise ValueError(
            f"Coverage {percentage:.2f}% is below required {min_percentage}%"
        )
    return percentage
