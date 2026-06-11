import frontMatter from 'front-matter'

type BlogFrontMatter = {
    title?: string
    date?: string
    tags?: string[]
    summary?: string
}

export type BlogPost = {
    slug: string
    title: string
    date: string
    tags: string[]
    summary: string
    content: string
    readingTime: string
}

const modules = import.meta.glob<string>('../content/blog/*.md', {
    eager: true,
    import: 'default',
    query: '?raw',
})

function slugFromPath(path: string) {
    return path.split('/').at(-1)?.replace(/\.md$/, '') ?? path
}

function estimateReadingTime(content: string) {
    const words = content.trim().split(/\s+/).filter(Boolean).length
    const minutes = Math.max(1, Math.ceil(words / 220))
    return `${minutes} min read`
}

function normalizePost(path: string, raw: string): BlogPost {
    const parsed = frontMatter<BlogFrontMatter>(raw)
    const slug = slugFromPath(path)

    return {
        slug,
        title: parsed.attributes.title ?? slug,
        date: parsed.attributes.date ?? '1970-01-01',
        tags: parsed.attributes.tags ?? [],
        summary: parsed.attributes.summary ?? '',
        content: parsed.body.trim(),
        readingTime: estimateReadingTime(parsed.body),
    }
}

const posts = Object.entries(modules)
    .map(([path, raw]) => normalizePost(path, raw))
    .sort((a, b) => b.date.localeCompare(a.date))

export function getBlogPosts() {
    return posts
}

export function getBlogPost(slug: string) {
    return posts.find((post) => post.slug === slug)
}
