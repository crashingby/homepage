type BlogFrontMatter = {
    title?: string
    date?: string
    tags?: string[]
    summary?: string
}

export type BlogTopic = {
    slug: string
    label: string
    description: string
}

export type BlogPost = {
    slug: string
    title: string
    date: string
    tags: string[]
    summary: string
    content: string
    readingTime: string
    topic: BlogTopic
    path: string
}

const topicConfig: Record<string, Omit<BlogTopic, 'slug'>> = {
    'cpp-knowledge': {
        label: 'CPP 知识',
        description: 'C++ language notes, STL, memory model, templates, and engineering practice.',
    },
    'gpu-programming': {
        label: 'GPU 编程',
        description: 'CUDA, Triton, kernel optimization, GPU architecture, and performance notes.',
    },
    'llm-inference': {
        label: 'LLM 推理',
        description: 'Inference serving, KV cache, batching, scheduling, and system design.',
    },
    leetcode: {
        label: '力扣题',
        description: 'LeetCode problems, algorithm notes, and solution write-ups.',
    },
}

const fallbackTopic: BlogTopic = {
    slug: 'notes',
    label: '随笔',
    description: 'General technical notes.',
}

const modules = import.meta.glob<string>('../content/blog/**/*.md', {
    eager: true,
    import: 'default',
    query: '?raw',
})

function parseFrontMatter(raw: string) {
    if (!raw.startsWith('---')) {
        return {
            attributes: {},
            body: raw,
        }
    }

    const end = raw.indexOf('\n---', 3)

    if (end === -1) {
        return {
            attributes: {},
            body: raw,
        }
    }

    const frontMatterText = raw.slice(3, end).trim()
    const body = raw.slice(end + 4).trim()
    const attributes: BlogFrontMatter = {}

    for (const line of frontMatterText.split('\n')) {
        const separator = line.indexOf(':')

        if (separator === -1) {
            continue
        }

        const key = line.slice(0, separator).trim()
        const value = line.slice(separator + 1).trim()

        if (key === 'title') {
            attributes.title = value
        }

        if (key === 'date') {
            attributes.date = value
        }

        if (key === 'summary') {
            attributes.summary = value
        }

        if (key === 'tags') {
            attributes.tags = value
                .replace(/^\[/, '')
                .replace(/\]$/, '')
                .split(',')
                .map((tag) => tag.trim())
                .filter(Boolean)
        }
    }

    return {
        attributes,
        body,
    }
}

function slugFromPath(path: string) {
    return path.split('/').at(-1)?.replace(/\.md$/, '') ?? path
}

function topicSlugFromPath(path: string) {
    const parts = path.split('/')
    const blogIndex = parts.indexOf('blog')
    return parts.at(blogIndex + 1) ?? fallbackTopic.slug
}

function topicFromPath(path: string): BlogTopic {
    const slug = topicSlugFromPath(path)
    const config = topicConfig[slug]

    if (!config) {
        return {
            ...fallbackTopic,
            slug,
            label: slug
                .split('-')
                .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
                .join(' '),
        }
    }

    return {
        slug,
        ...config,
    }
}

function estimateReadingTime(content: string) {
    const words = content.trim().split(/\s+/).filter(Boolean).length
    const minutes = Math.max(1, Math.ceil(words / 220))
    return `${minutes} min read`
}

function normalizePost(path: string, raw: string): BlogPost {
    const parsed = parseFrontMatter(raw)
    const slug = slugFromPath(path)
    const topic = topicFromPath(path)

    return {
        slug,
        title: parsed.attributes.title ?? slug,
        date: parsed.attributes.date ?? '1970-01-01',
        tags: parsed.attributes.tags ?? [],
        summary: parsed.attributes.summary ?? '',
        content: parsed.body.trim(),
        readingTime: estimateReadingTime(parsed.body),
        topic,
        path: `/blog/${topic.slug}/${slug}`,
    }
}

const posts = Object.entries(modules)
    .map(([path, raw]) => normalizePost(path, raw))
    .sort((a, b) => b.date.localeCompare(a.date))

export function getBlogPosts() {
    return posts
}

export function getBlogTopics() {
    return Object.entries(topicConfig).map(([slug, config]) => ({
        slug,
        ...config,
    }))
}

export function getBlogPostsByTopic(topicSlug: string) {
    return posts.filter((post) => post.topic.slug === topicSlug)
}

export function getBlogTopic(topicSlug: string) {
    return getBlogTopics().find((topic) => topic.slug === topicSlug)
}

export function getBlogPost(topicSlug: string | undefined, slug: string) {
    return posts.find((post) => {
        if (topicSlug) {
            return post.topic.slug === topicSlug && post.slug === slug
        }

        return post.slug === slug
    })
}
