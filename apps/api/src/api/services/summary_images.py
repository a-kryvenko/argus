"""Transparent SVG charts for the cached homepage summary."""
from api.schemas.summary import ForecastImageData


class ForecastImageGenerator:
    width = 480
    height = 240
    padding = 12
    colors = ('#22c55e', '#eab308', '#f97316', '#ef4444', '#dc143c')

    def render(self, before, current, after, critical_levels) -> bytes:
        data = ForecastImageData(before=before, current=current, after=after, critical_levels=critical_levels)
        points = [*data.before, data.current, *data.after]
        values = [point.value for point in points]
        low, high = min(values), max(values)
        margin = (high-low) * .1 if high > low else max(abs(low) * .05, 1.)
        low, high = low-margin, high+margin

        def y(value):
            return self.height-self.padding - (value-low)/(high-low)*(self.height-2*self.padding)

        coordinates = [(
            self.padding + index/(len(points)-1)*(self.width-2*self.padding),
            y(point.value),
        ) for index, point in enumerate(points)]
        levels = data.critical_levels
        direction = 1 if (levels[-1] > levels[0] if len(levels) > 1 else levels[0] >= 0) else -1
        levels = [value*direction for value in levels]
        # Extend the first danger band down to green; a lone level uses zero as its safe baseline.
        safe = levels[0]-(levels[1]-levels[0] if len(levels) > 1 else max(abs(levels[0]), 1.))
        anchors = [(safe, self.colors[0]),
                   *((level, self.colors[i+1]) for i, level in enumerate(levels))]
        stops = ''.join(f'<stop offset="{(value-safe)/(levels[-1]-safe):.6f}" stop-color="{color}"/>'
                        for value, color in anchors)
        line = ' '.join(f'{x:.3f},{v:.3f}' for x, v in coordinates)
        current_x = coordinates[len(data.before)][0]
        dots = ''.join(f'<circle cx="{x:.3f}" cy="{v:.3f}" r="{5 if i == len(data.before) else 3.5}" '
                       'fill="url(#risk)"/>' for i, (x, v) in enumerate(coordinates))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="240" viewBox="0 0 480 240">'
                f'<defs><linearGradient id="risk" gradientUnits="userSpaceOnUse" x1="0" x2="0" '
                f'y1="{y(safe*direction):.3f}" y2="{y(levels[-1]*direction):.3f}">{stops}'
                '</linearGradient></defs>'
                f'<line x1="{current_x:.3f}" x2="{current_x:.3f}" '
                f'y1="{self.padding}" y2="{self.height-self.padding}" '
                'stroke="white" stroke-width="1.5" vector-effect="non-scaling-stroke" '
                'stroke-dasharray="4 4" stroke-opacity="0.7"/>'
                f'<polyline points="{line}" fill="none" stroke="url(#risk)" stroke-width="2" '
                'vector-effect="non-scaling-stroke" '
                f'stroke-linecap="round" stroke-linejoin="round"/>{dots}</svg>').encode()

