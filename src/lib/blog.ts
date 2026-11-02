import { getMarkdownHeadings } from './markdown'

type BlogFrontMatter = {
    title?: string
    date?: string
    tags?: string[]
    summary?: string
    series?: string
    seriesOrder?: number
}

export type BlogTopic = {
    slug: string
    label: string
    description: string
}

export type BlogCategory = {
    slug: string
    label: string
    description: string
    series?: string
}

export type BlogSeries = {
    slug: string
    label: string
    order: number
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
    category?: BlogCategory
    series?: BlogSeries
    path: string
}

const topicConfig: Record<string, Omit<BlogTopic, 'slug'>> = {
    'cpp-knowledge': {
        label: 'CPP 知识',
        description: 'C++ language notes, STL, memory model, templates, and engineering practice.',
    },
    'gpu-programming': {
        label: 'GPU 编程',
        description: '从 GPU 基础、CUDA 与算子编写，到 CUTLASS、CuTe、Triton 和性能分析。',
    },
    'llm-inference': {
        label: 'LLM 推理',
        description: 'Inference serving, KV cache, batching, scheduling, and system design.',
    },
    leetcode: {
        label: '力扣题',
        description: 'LeetCode problems, algorithm notes, and solution write-ups.',
    },
    'Fault-Tolerance-system': {
        label: '容错系统复习',
        description: '有关容错系统的知识和题目',
    },
    'Software-Construction': {
        label: '软件构造复习',
        description: '有关软件构造的知识笔记',
    },
    'Paper-Reading': {
        label: '论文阅读笔记',
        description: '简单总结读过的论文',
    },
    'Tips': {
        label: '小知识',
        description: '一些实用的小知识',
    },
    'training-system': {
        label: '训练系统',
        description: '有关深度学习训练的理论，系统实现',
    },
    'gpu-communication': {
        label: 'GPU通信系统',
        description: '有关GPU通信的理论和实践知识',
    },
    'system-software-design': {
        label: '系统软件设计',
        description: '学习如何设计一个合理的系统',
    },
}

const categoryConfig: Record<string, BlogCategory[]> = {
    'gpu-programming': [
        {
            slug: 'fundamentals',
            label: 'GPU 基础与数值表示',
            description: '矩阵布局、精度类型与量化基础。',
        },
        {
            slug: 'cuda',
            label: 'CUDA 编程',
            description: '内存、同步、异步搬运、运行时与算子实现。',
        },
        {
            slug: 'cutlass-cute',
            label: 'CUTLASS 与 CuTe',
            description: '布局代数、线程分区、数据搬运与矩阵计算。',
        },
        {
            slug: 'triton',
            label: 'Triton 编程',
            description: '编程模型、自动调优与算子实践。',
        },
        {
            slug: 'flash-attention',
            label: 'FlashAttention 源码阅读',
            description: '从 Python 接口逐步追踪到 CUDA 前向计算。',
            series: 'flash-attention-2',
        },
        {
            slug: 'performance',
            label: '性能分析与优化',
            description: '性能模型、瓶颈判断与 Nsight Compute 指标。',
        },
    ],
}

const seriesLabels: Record<string, string> = {
    'flash-attention-2': 'FlashAttention-2 源码学习',
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

        if (key === 'series') {
            attributes.series = value
        }

        if (key === 'seriesOrder' && Number.isInteger(Number(value)) && Number(value) > 0) {
            attributes.seriesOrder = Number(value)
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
    const parts = path.split('/')
    const categorySlug = parts[parts.indexOf('blog') + 2]
    const category = getBlogCategories(topic.slug).find((item) => item.slug === categorySlug)
    const { series, seriesOrder } = parsed.attributes

    return {
        slug,
        title: parsed.attributes.title ?? slug,
        date: parsed.attributes.date ?? '1970-01-01',
        tags: parsed.attributes.tags ?? [],
        summary: parsed.attributes.summary ?? '',
        content: parsed.body.trim(),
        readingTime: estimateReadingTime(parsed.body),
        topic,
        category,
        series: series && seriesOrder
            ? { slug: series, label: seriesLabels[series] ?? series, order: seriesOrder }
            : undefined,
        path: `/blog/${topic.slug}/${slug}`,
    }
}

const posts = Object.entries(modules)
    .map(([path, raw]) => normalizePost(path, raw))
    .sort((a, b) => b.date.localeCompare(a.date))

// 二级目录用于分类，文章路径仍使用专题和文件名；在加载时检查路径唯一性。
const postPaths = new Set<string>()
for (const post of posts) {
    if (postPaths.has(post.path)) {
        throw new Error(`文章路径重复：${post.path}，请使用专题内唯一的 Markdown 文件名。`)
    }
    postPaths.add(post.path)
}

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

export function getBlogCategories(topicSlug: string) {
    return categoryConfig[topicSlug] ?? []
}

export function getBlogCategoryPath(topicSlug: string, categorySlug?: string) {
    const search = categorySlug ? `?${new URLSearchParams({ category: categorySlug })}` : ''
    return `/blog/topic/${topicSlug}${search}`
}

export function getBlogPostsByCategory(topicSlug: string, categorySlug: string) {
    const category = getBlogCategories(topicSlug).find((item) => item.slug === categorySlug)
    const categoryPosts = posts.filter(
        (post) => post.topic.slug === topicSlug && post.category?.slug === categorySlug,
    )

    if (!category?.series) return categoryPosts

    const seriesSlug = category.series
    return categoryPosts.toSorted((a, b) => {
        const aOrder = a.series?.slug === seriesSlug ? a.series.order : Infinity
        const bOrder = b.series?.slug === seriesSlug ? b.series.order : Infinity
        return (aOrder === bOrder ? 0 : aOrder < bOrder ? -1 : 1) || b.date.localeCompare(a.date)
    })
}

export function getBlogSeriesPosts(sourcePost: BlogPost) {
    if (!sourcePost.series) return []
    return posts
        .filter((post) => post.topic.slug === sourcePost.topic.slug &&
            post.series?.slug === sourcePost.series?.slug)
        .toSorted((a, b) => a.series!.order - b.series!.order)
}

export function getRelatedBlogPosts(sourcePost: BlogPost, limit = 2) {
    return getBlogPostsByTopic(sourcePost.topic.slug)
        .filter((post) => post.path !== sourcePost.path)
        .toSorted((a, b) =>
            Number(b.category?.slug === sourcePost.category?.slug) -
            Number(a.category?.slug === sourcePost.category?.slug),
        )
        .slice(0, limit)
}

export function getBlogPost(topicSlug: string | undefined, slug: string) {
    return posts.find((post) => {
        if (topicSlug) {
            return post.topic.slug === topicSlug && post.slug === slug
        }

        return post.slug === slug
    })
}

function findWikiTargetPost(sourcePost: BlogPost, pageName: string) {
    const normalizedPageName = pageName.trim()

    return (
        posts.find(
            (post) =>
                post.topic.slug === sourcePost.topic.slug &&
                (post.slug === normalizedPageName || post.title === normalizedPageName),
        ) ??
        posts.find((post) => post.slug === normalizedPageName || post.title === normalizedPageName)
    )
}

function findWikiHeadingId(post: BlogPost, headingTitle: string) {
    const normalizedHeadingTitle = headingTitle.trim()
    const heading = getMarkdownHeadings(post.content).find(
        (item) => item.title === normalizedHeadingTitle,
    )

    return heading?.id
}

function escapeMarkdownLinkText(value: string) {
    return value.replace(/]/g, '\\]')
}

function wikiLinkToMarkdown(sourcePost: BlogPost, rawTarget: string) {
    const [targetPart, aliasPart] = rawTarget.split('|')
    const [rawPageName, rawHeadingTitle] = targetPart.split('#')
    const pageName = rawPageName.trim() || sourcePost.slug
    const headingTitle = rawHeadingTitle?.trim()
    const targetPost = findWikiTargetPost(sourcePost, pageName)

    if (!targetPost) {
        return `[[${rawTarget}]]`
    }

    const headingId = headingTitle ? findWikiHeadingId(targetPost, headingTitle) : undefined
    const label =
        aliasPart?.trim() ||
        (headingTitle ? `${targetPost.title} > ${headingTitle}` : targetPost.title)
    const href = `#${targetPost.path}${headingId ? `#${headingId}` : ''}`

    return `[${escapeMarkdownLinkText(label)}](${href})`
}

export function resolveWikiLinks(content: string, sourcePost: BlogPost) {
    let inCodeBlock = false

    return content
        .split('\n')
        .map((line) => {
            if (line.trim().startsWith('```')) {
                inCodeBlock = !inCodeBlock
                return line
            }

            if (inCodeBlock) {
                return line
            }

            return line.replace(/\[\[([^\]\n]+)]]/g, (_, rawTarget: string) =>
                wikiLinkToMarkdown(sourcePost, rawTarget),
            )
        })
        .join('\n')
}
