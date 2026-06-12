export type MarkdownHeading = {
    id: string
    level: 1 | 2 | 3 | 4 | 5 | 6
    line: number
    title: string
}

function slugify(value: string) {
    return (
        value
            .trim()
            .toLowerCase()
            .replace(/[`*_~[\]()]/g, '')
            .replace(/[^\p{Letter}\p{Number}\s-]/gu, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .replace(/^-|-$/g, '') || 'section'
    )
}

export function createHeadingSlugger() {
    const counts = new Map<string, number>()

    return (title: string) => {
        const base = slugify(title)
        const count = counts.get(base) ?? 0
        counts.set(base, count + 1)

        return count === 0 ? base : `${base}-${count + 1}`
    }
}

type MarkdownHeadingOptions = {
    maxLevel?: MarkdownHeading['level']
    minLevel?: MarkdownHeading['level']
}

export function getMarkdownHeadings(content: string, options: MarkdownHeadingOptions = {}) {
    const headings: MarkdownHeading[] = []
    const slug = createHeadingSlugger()
    let inCodeBlock = false
    const minLevel = options.minLevel ?? 1
    const maxLevel = options.maxLevel ?? 6

    content.split('\n').forEach((line, index) => {
        if (line.trim().startsWith('```')) {
            inCodeBlock = !inCodeBlock
            return
        }

        if (inCodeBlock) {
            return
        }

        const match = /^(#{1,6})\s+(.+?)\s*#*$/.exec(line)

        if (!match) {
            return
        }

        const level = match[1].length as MarkdownHeading['level']
        const title = match[2].trim()
        const heading = {
            id: slug(title),
            level,
            line: index + 1,
            title,
        }

        if (level >= minLevel && level <= maxLevel) {
            headings.push(heading)
        }
    })

    return headings
}
