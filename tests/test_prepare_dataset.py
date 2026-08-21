from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.prepare_dataset import convert_annotation, sequence_group


def test_rdd_xml_keeps_only_d40_and_normalizes_box(tmp_path: Path) -> None:
    image_path = tmp_path / "India_000001.jpg"
    Image.new("RGB", (200, 100)).save(image_path)
    annotation_path = tmp_path / "India_000001.xml"
    annotation_path.write_text(
        """
        <annotation>
          <size><width>200</width><height>100</height></size>
          <object><name>D00</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object>
          <object><name>D40</name><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>80</ymax></bndbox></object>
        </annotation>
        """,
        encoding="utf-8",
    )
    assert convert_annotation(annotation_path, image_path) == [
        "0 0.50000000 0.50000000 0.50000000 0.60000000"
    ]


def test_non_pothole_image_becomes_intentional_negative(tmp_path: Path) -> None:
    image_path = tmp_path / "India_000002.jpg"
    Image.new("RGB", (20, 10)).save(image_path)
    annotation_path = tmp_path / "India_000002.xml"
    annotation_path.write_text(
        "<annotation><size><width>20</width><height>10</height></size>"
        "<object><name>D10</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
        "<xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    assert convert_annotation(annotation_path, image_path) == []


def test_nearby_numbered_images_share_a_group() -> None:
    assert sequence_group("India", "India_000101", 50) == sequence_group(
        "India", "India_000149", 50
    )
    assert sequence_group("India", "India_000149", 50) != sequence_group(
        "India", "India_000150", 50
    )
