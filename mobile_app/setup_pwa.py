"""
One-time PWA icon generator.
Run this once before starting the server:

    python setup_pwa.py

Generates:
    icon-192.png   (Android home screen, PWA manifest)
    icon-512.png   (Splash screen, high-DPI)
    apple-touch-icon.png  (iOS Add to Home Screen)
"""

import os
import sys
from pathlib import Path

OUT = Path(__file__).parent

def make_icon(size: int, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor='#0D0D0D')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.axis('off')

    # Background
    ax.set_facecolor('#0D0D0D')

    # Lime rounded square base
    sq = mpatches.FancyBboxPatch(
        (10, 10), 80, 80,
        boxstyle='round,pad=0',
        facecolor='#161616',
        edgecolor='#C9F31D',
        linewidth=size / 64,
        zorder=1,
    )
    ax.add_patch(sq)

    # Activity bar chart (7 bars, activity colours)
    colours = ['#a78bfa', '#818cf8', '#60a5fa', '#34d399', '#f87171', '#fbbf24', '#fb923c']
    heights = [0.35, 0.50, 0.60, 1.00, 0.75, 0.85, 0.55]   # normalised
    bar_w   = 7.5
    gap     = 2.5
    total_w = len(colours) * bar_w + (len(colours) - 1) * gap
    x0      = (100 - total_w) / 2
    max_h   = 50
    base_y  = 22

    for i, (c, h) in enumerate(zip(colours, heights)):
        x = x0 + i * (bar_w + gap)
        rect = mpatches.Rectangle(
            (x, base_y), bar_w, h * max_h,
            facecolor=c, edgecolor='none', zorder=2,
        )
        ax.add_patch(rect)

    # Lime underline
    ax.plot([15, 85], [18, 18], color='#C9F31D', linewidth=size / 96, zorder=3)

    # "HAR" text
    ax.text(
        50, 83, 'HAR',
        ha='center', va='center',
        fontsize=size / 24,
        fontweight='black',
        color='#C9F31D',
        fontfamily='monospace',
        zorder=3,
    )

    fig.savefig(str(path), dpi=100, facecolor='#0D0D0D', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'  Created {path.name}  ({size}x{size})')


def main():
    print('\n  Generating PWA icons...\n')
    try:
        make_icon(192, OUT / 'icon-192.png')
        make_icon(512, OUT / 'icon-512.png')

        # Apple touch icon = 180x180 (same image, different name)
        import shutil
        shutil.copy(OUT / 'icon-192.png', OUT / 'apple-touch-icon.png')
        print('  Created apple-touch-icon.png')

        print('\n  Done. Now run:  python serve.py\n')

    except Exception as exc:
        print(f'\n  ERROR: {exc}')
        sys.exit(1)


if __name__ == '__main__':
    main()
