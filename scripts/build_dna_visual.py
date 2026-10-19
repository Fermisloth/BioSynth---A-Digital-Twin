"""Generate the decorative, locally served molecular helix (no runtime dependencies)."""
from math import sin, cos, pi
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'ui' / 'assets'


def build(animated):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 640">',
             '<defs><radialGradient id="violet" cx="30%" cy="25%"><stop stop-color="#ffffff"/><stop offset=".3" stop-color="#dab7ff"/><stop offset=".7" stop-color="#9761cf"/><stop offset="1" stop-color="#502785"/></radialGradient>',
             '<radialGradient id="rose" cx="30%" cy="25%"><stop stop-color="#fff"/><stop offset=".35" stop-color="#edc9e7"/><stop offset="1" stop-color="#9e71b6"/></radialGradient></defs>']
    frames = 33 if animated else 1

    def point(t, side, phase, offset=0):
        a = t * pi * 4.2 + phase + side * pi
        z = cos(a)
        return (260 + 110 * sin(a) + 85 * (t - .5) + offset,
                -12 + t * 665 + 10 * z, z)

    # Fine connecting bonds and spherical atoms make the helix legible in silhouette.
    for step in range(38):
        t = step / 37
        for bead in range(11):
            fraction = bead / 10
            coords = []
            for frame in range(frames):
                phase = frame / 32 * 2 * pi if animated else .6
                x1, y1, z1 = point(t, 0, phase)
                x2, y2, z2 = point(t, 1, phase)
                coords.append((x1+(x2-x1)*fraction, y1+(y2-y1)*fraction, 2.2, .3+.3*(1-abs(fraction-.5))))
            parts.append(circle(coords, 'rose', animated))
    for side in range(2):
        for step in range(110):
            t = step / 109
            for cluster in range(3):
                coords = []
                for frame in range(frames):
                    phase = frame / 32 * 2 * pi if animated else .6
                    x, y, z = point(t, side, phase)
                    x += sin(step * 2.3 + cluster * 2.1) * (5 + cluster*3)
                    y += cos(step * 1.7 + cluster) * 6
                    coords.append((x, y, (3.1 + (z+1)*1.5) * (1 if cluster==0 else .7), .45 + (z+1)*.25))
                parts.append(circle(coords, 'violet' if side==0 else 'rose', animated))
    parts.append('</svg>')
    return ''.join(parts)


def circle(coords, fill, animated):
    x, y, radius, opacity = coords[0]
    result = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" opacity="{opacity:.2f}" fill="url(#{fill})">'
    if animated:
        for index, name in enumerate(('cx', 'cy', 'r', 'opacity')):
            values = ';'.join(f'{p[index]:.2f}' for p in coords)
            result += f'<animate attributeName="{name}" values="{values}" dur="18s" repeatCount="indefinite"/>'
    return result + '</circle>'


if __name__ == '__main__':
    for animated, name in ((True, 'dna-helix.svg'), (False, 'dna-helix-still.svg')):
        target = OUT / name
        target.write_text(build(animated), encoding='utf-8')
        print(f'{target.name}: {target.stat().st_size:,} bytes')
