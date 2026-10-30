from datetime import UTC, datetime, timedelta
from xml.etree import ElementTree as ET

import pytest

from api.services import summary_images as images

SVG = '{http://www.w3.org/2000/svg}'
NOW = datetime(2026, 10, 29, 12, tzinfo=UTC)


def inputs(count=3, values=None, levels=None):
    values = values or [300., 400., 450., 500., 550., 600., 650.]
    points = [dict(time=NOW+timedelta(hours=i-3), value=value) for i, value in enumerate(values)]
    return dict(before=points[3-count:3], current=points[3], after=points[4:4+count],
                critical_levels=levels or [450., 500., 600.])


@pytest.mark.parametrize('count', [2, 3])
def test_svg_is_transparent_and_contains_only_chart_and_current_marker(count):
    content = images.ForecastImageGenerator().render(**inputs(count))
    svg = ET.fromstring(content)
    assert svg.attrib == dict(width='480', height='240', viewBox='0 0 480 240')
    assert len(svg.findall(f'{SVG}circle')) == 2*count+1
    assert len(svg.findall(f'{SVG}polyline')[0].attrib['points'].split()) == 2*count+1
    assert {node.tag for node in svg.iter()} == {
        SVG+name for name in ('svg', 'defs', 'linearGradient', 'stop', 'line', 'polyline', 'circle')}
    for circle in svg.findall(f'{SVG}circle'):
        assert 0 < float(circle.attrib['cx']) < 480
        assert 0 < float(circle.attrib['cy']) < 240
    assert len(content) < 3000


@pytest.mark.parametrize('levels', [[500], [450, 600], [450, 500, 600], [-30, -50, -100],
                                   [450, 500, 600, 700], [-30, -50, -100, -150]])
def test_gradient_follows_critical_levels_in_danger_order(levels):
    data = inputs(levels=levels)
    if levels[0] < 0:
        for point in [*data['before'], data['current'], *data['after']]:
            point['value'] = -(point['value']-250)/4
    svg = ET.fromstring(images.ForecastImageGenerator().render(**data))
    gradient = svg.find(f'{SVG}defs/{SVG}linearGradient')
    stops = gradient.findall(f'{SVG}stop')
    assert stops[0].attrib == {'offset': '0.000000', 'stop-color': '#22c55e'}
    assert stops[-1].attrib == {'offset': '1.000000', 'stop-color': images.ForecastImageGenerator.colors[len(levels)]}
    offsets = [float(stop.attrib['offset']) for stop in stops]
    assert offsets == sorted(set(offsets))
    assert [stop.attrib['stop-color'] for stop in stops] == list(images.ForecastImageGenerator.colors[:len(levels)+1])
    y1, y2 = float(gradient.attrib['y1']), float(gradient.attrib['y2'])
    assert (y1 > y2) == (levels[0] > 0)


def test_constant_series_and_actual_time_spacing():
    data = inputs(values=[400.]*7)
    data['after'][-1]['time'] += timedelta(hours=3)
    svg = ET.fromstring(images.ForecastImageGenerator().render(**data))
    points = [tuple(map(float, pair.split(','))) for pair in svg.find(f'{SVG}polyline').attrib['points'].split()]
    assert len({point[1] for point in points}) == 1
    assert points[-1][0]-points[-2][0] == pytest.approx(4*(points[1][0]-points[0][0]), abs=.01)
    marker = svg.find(f'{SVG}line')
    assert float(marker.attrib['x1']) == float(marker.attrib['x2']) == points[len(data['before'])][0]


@pytest.mark.parametrize('invalid', ['short_before', 'long_after', 'order', 'duplicate', 'naive',
                                    'nonfinite', 'empty_levels', 'unordered_levels', 'duplicate_levels'])
def test_invalid_inputs_are_rejected(invalid):
    data = inputs()
    if invalid == 'short_before':
        data['before'] = data['before'][:1]
    elif invalid == 'long_after':
        data['after'] *= 2
    elif invalid == 'order':
        data['before'].reverse()
    elif invalid == 'duplicate':
        data['after'][0]['time'] = data['current']['time']
    elif invalid == 'naive':
        data['current']['time'] = NOW.replace(tzinfo=None)
    elif invalid == 'nonfinite':
        data['current']['value'] = float('nan')
    elif invalid == 'empty_levels':
        data['critical_levels'] = []
    elif invalid == 'unordered_levels':
        data['critical_levels'] = [4, 6, 5]
    else:
        data['critical_levels'] = [4, 4]
    with pytest.raises(ValueError):
        images.ForecastImageGenerator().render(**data)

