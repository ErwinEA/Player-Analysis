from backend.app.pipeline.heatmap import build_heatmap_from_metre_points


def test_build_heatmap_from_metre_points():
    positions = [(52.5, 34.0), (60.0, 40.0), (70.0, 45.0)]
    result = build_heatmap_from_metre_points(positions)
    assert result.sample_count == 3
    assert len(result.image_png_base64) > 100
    assert result.zone_summary is not None
    assert result.zone_summary.middle_third_pct > 0.0
