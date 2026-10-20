import frontMatter from 'front-matter'
import { getMarkdownHeadings } from './markdown'

type InterviewFrontMatter = {
    title?: string
    summary?: string
    order?: number
}

export type InterviewTrack = {
    slug: string
    label: string
    caption: string
    description: string
    order: number
}

export type InterviewChapter = {
    slug: string
    title: string
    summary: string
    content: string
    order: number
    track: InterviewTrack
    path: string
}

const trackConfig: Record<string, Omit<InterviewTrack, 'slug'>> = {
    cpp: {
        label: 'C++ 编程',
        caption: 'CODE & REASONING',
        description: '从代码实现出发，理解并发、复杂度与边界条件。',
        order: 1,
    },
    infra: {
        label: 'AI Infra',
        caption: 'SYSTEMS & SERVING',
        description: '围绕部署、推理框架与性能优化，梳理系统问题。',
        order: 2,
    },
}

const modules = import.meta.glob<string>('../content/interview/*/*.md', {
    eager: true,
    import: 'default',
    query: '?raw',
})

const chapters: InterviewChapter[] = Object.entries(modules)
    .map(([sourcePath, raw]) => {
        const parts = sourcePath.split('/')
        const trackSlug = parts.at(-2)!
        const slug = parts.at(-1)!.replace(/\.md$/, '')
        const { attributes, body } = frontMatter<InterviewFrontMatter>(raw)
        const titleHeading = getMarkdownHeadings(body).find(
            (heading) => heading.level === 1,
        )
        const track: InterviewTrack = {
            slug: trackSlug,
            ...(trackConfig[trackSlug] ?? {
                label: trackSlug,
                caption: 'INTERVIEW TOPIC',
                description: '',
                order: 99,
            }),
        }

        return {
            slug,
            title: attributes.title ?? titleHeading?.title ?? slug,
            summary: attributes.summary ?? '',
            content: body.trim(),
            order: attributes.order ?? 99,
            track,
            path: `/interview/${encodeURIComponent(trackSlug)}/${encodeURIComponent(slug)}`,
        }
    })
    .sort(
        (a, b) =>
            a.track.order - b.track.order ||
            a.order - b.order ||
            a.title.localeCompare(b.title, 'zh-CN'),
    )

export function getInterviewChapters() {
    return chapters
}

export function getInterviewTracks() {
    return Array.from(
        new Map(chapters.map((chapter) => [chapter.track.slug, chapter.track])).values(),
    )
}

export function getInterviewChapter(trackSlug: string, slug: string) {
    return chapters.find(
        (chapter) => chapter.track.slug === trackSlug && chapter.slug === slug,
    )
}

const legacyPaths = new Map([
    ['cpp手撕题', '/interview/cpp/cpp-hands-on'],
    ['Infra八股', '/interview/infra/infra-questions'],
])

export function getLegacyInterviewPath(slug: string) {
    return legacyPaths.get(slug)
}
