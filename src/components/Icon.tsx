import type { CSSProperties } from 'react'

const paths = {
    arrow: 'M5 12h14m-6-6 6 6-6 6',
    upRight: 'M7 17 17 7M7 7h10v10',
    back: 'M19 12H5m6-6-6 6 6 6',
    search: 'm21 21-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
    sun: 'M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
    moon: 'M20.5 13A9 9 0 0 1 11 3.5 9 9 0 1 0 20.5 13Z',
    music: 'M9 18V5l11-2v13M9 18c0 1.7-1.6 3-3.5 3S2 20 2 18.5 3.6 16 5.5 16 9 16.5 9 18Zm11-2c0 1.7-1.6 3-3.5 3s-3.5-1-3.5-2.5 1.6-2.5 3.5-2.5 3.5.5 3.5 2Z',
    play: 'm8 5 11 7-11 7Z',
    pause: 'M8 5v14M16 5v14',
    next: 'm5 5 10 7-10 7ZM19 5v14',
    previous: 'm19 5-10 7 10 7ZM5 5v14',
    close: 'm6 6 12 12M6 18 18 6',
    copy: 'M9 9h11v12H9zM5 15H3V3h12v2',
    check: 'm5 12 4 4L19 6',
    list: 'M9 6h12M9 12h12M9 18h12M3 6h.01M3 12h.01M3 18h.01',
    chevron: 'm9 5 7 7-7 7',
    expand: 'M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5',
} as const

export function Icon({
    name,
    size = 18,
    style,
}: {
    name: keyof typeof paths
    size?: number
    style?: CSSProperties
}) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            style={style}
        >
            <path d={paths[name]} />
        </svg>
    )
}
