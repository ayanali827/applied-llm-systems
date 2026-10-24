"""Generate a small sample PDF for trying out Task 2.

Writes a fictional 3-page manual for the "RoboMover X2" delivery robot
into this package's data folder. Each page holds different facts, so
source citations (file + page) can be checked after ingesting it.
Nothing else in the project imports reportlab; this script is the
only reason reportlab is in requirements.txt.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "sample_robomover_x2.pdf"

PAGES: list[tuple[str, list[str]]] = [
    (
        "RoboMover X2 - Overview",
        [
            "The RoboMover X2 is an autonomous delivery robot built by Datheon Robotics.",
            "It carries payloads of up to 25 kilograms in a locked cargo box.",
            "The robot is designed for hospitals and office buildings, where it "
            "delivers medicines, documents, and small parcels between floors.",
        ],
    ),
    (
        "RoboMover X2 - Battery and Charging",
        [
            "The RoboMover X2 runs on a 1.2 kilowatt-hour battery pack.",
            "A full charge gives up to 6 hours of continuous operation.",
            "Charging happens at the DockStation D2 and takes 90 minutes "
            "from empty to full.",
        ],
    ),
    (
        "RoboMover X2 - Safety",
        [
            "Lidar and ultrasonic sensors let the RoboMover X2 detect obstacles "
            "and stop within 0.4 metres of them.",
            "In pedestrian areas the robot never drives faster than 4 kilometres per hour.",
            "A red emergency stop button on the top panel cuts power to the wheels immediately.",
        ],
    ),
]


def build_sample_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    """Write the 3-page sample PDF and return its path."""
    styles = getSampleStyleSheet()
    story: list = []
    for page_number, (title, paragraphs) in enumerate(PAGES):
        if page_number > 0:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 12))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyText"]))
            story.append(Spacer(1, 6))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    doc.build(story)
    return output_path


def main() -> None:
    """Generate the sample PDF and print where it was written."""
    path = build_sample_pdf()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
