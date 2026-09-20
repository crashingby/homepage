import { useEffect, useMemo, useState } from 'react'
import { Link, Navigate, useLocation, useParams } from 'react-router-dom'
import { MarkdownPost } from '../components/MarkdownPost'
import { ArticleTOC } from '../components/ArticleTOC'
import { Icon } from '../components/Icon'
import {
    getBlogPost,
    getBlogPostsByTopic,
    resolveWikiLinks,
    type BlogPost,
} from '../lib/blog'
import { getMarkdownHeadings } from '../lib/markdown'
import { useReadingProgress } from '../lib/useReadingProgress'

function Article({ post }: { post: BlogPost }) {
    const location = useLocation()
    const [wide, setWide] = useState(false)
    const [largeText, setLargeText] = useState(false)
    const content = useMemo(() => resolveWikiLinks(post.content, post), [post])
    const allHeadings = useMemo(() => getMarkdownHeadings(content), [content])
    const duplicateTitle =
        allHeadings[0]?.level === 1 &&
        allHeadings[0].title.replace(/[*`_]/g, '').trim() === post.title.trim()
            ? allHeadings[0]
            : undefined
    const headings = useMemo(
        () =>
            allHeadings.filter(
                (heading) =>
                    heading.id !== duplicateTitle?.id && heading.level <= 4,
            ),
        [allHeadings, duplicateTitle?.id],
    )
    const { activeId, progress } = useReadingProgress(headings)
    const related = getBlogPostsByTopic(post.topic.slug)
        .filter((item) => item.path !== post.path)
        .slice(0, 2)

    useEffect(() => {
        document.title = `${post.title} — Crashing By`
    }, [post.title])
    useEffect(() => {
        if (!location.hash) return
        let id = location.hash.slice(1)
        try {
            id = decodeURIComponent(id)
        } catch {
            /* Keep malformed fragments readable. */
        }
        const timer = window.setTimeout(
            () =>
                document
                    .getElementById(id)
                    ?.scrollIntoView({ block: 'start', behavior: 'instant' }),
            100,
        )
        return () => window.clearTimeout(timer)
    }, [location.hash])

    return (
        <article
            className={`blog-post${wide ? ' is-wide' : ''}${largeText ? ' large-text' : ''}`}
        >
            <div
                className="reading-progress"
                style={{ width: `${progress}%` }}
                aria-hidden="true"
            />
            <div className="article-breadcrumb">
                <Link to="/blog">笔记</Link>
                <Icon name="chevron" size={13} />
                <Link to={`/blog/topic/${post.topic.slug}`}>
                    {post.topic.label}
                </Link>
                <span className="breadcrumb-end">/ 阅读</span>
            </div>
            <div className="article-layout">
                <div className="article-main">
                    <header className="post-header">
                        <p className="eyebrow">
                            {post.topic.label} <span>/ FIELD NOTES</span>
                        </p>
                        <h1 id={duplicateTitle?.id}>{post.title}</h1>
                        {post.summary && (
                            <p className="post-summary">{post.summary}</p>
                        )}
                        <div className="post-meta">
                            {post.date !== '1970-01-01' && (
                                <time dateTime={post.date}>{post.date}</time>
                            )}
                            <span>
                                {post.readingTime.replace(
                                    ' min read',
                                    ' 分钟阅读',
                                )}
                            </span>
                            <span>Huang Xinying</span>
                        </div>
                        <div className="article-header-bottom">
                            <div className="tag-list" aria-label="文章标签">
                                {post.tags.map((tag) => (
                                    <Link
                                        key={tag}
                                        to={`/blog?q=${encodeURIComponent(tag)}`}
                                    >
                                        {tag}
                                    </Link>
                                ))}
                            </div>
                            <div
                                className="reading-controls"
                                aria-label="阅读设置"
                            >
                                <button
                                    type="button"
                                    aria-label="放大正文字号"
                                    aria-pressed={largeText}
                                    title="放大正文字号"
                                    onClick={() => setLargeText(!largeText)}
                                >
                                    Aa
                                </button>
                                <button
                                    type="button"
                                    className="width-control"
                                    aria-label="加宽阅读区域"
                                    aria-pressed={wide}
                                    title="加宽阅读区域"
                                    onClick={() => setWide(!wide)}
                                >
                                    <Icon name="expand" size={16} />
                                </button>
                            </div>
                        </div>
                    </header>
                    <div className="article-content">
                        <MarkdownPost
                            content={content}
                            hiddenHeadingId={duplicateTitle?.id}
                        />
                    </div>
                    <footer className="article-end">
                        <span className="end-mark">∎</span>
                        <p>读到这里，谢谢你的时间。</p>
                        <Link
                            to={`/blog/topic/${post.topic.slug}`}
                            className="text-link"
                        >
                            <Icon name="back" size={16} /> 回到
                            {post.topic.label}
                        </Link>
                    </footer>
                    {related.length > 0 && (
                        <section className="related-notes">
                            <p className="eyebrow">KEEP EXPLORING</p>
                            <h2>继续翻一翻</h2>
                            <div>
                                {related.map((item) => (
                                    <Link to={item.path} key={item.path}>
                                        <span>{item.title}</span>
                                        <Icon name="arrow" size={18} />
                                    </Link>
                                ))}
                            </div>
                        </section>
                    )}
                </div>
                {headings.length > 0 && (
                    <ArticleTOC
                        headings={headings}
                        activeId={activeId}
                        progress={progress}
                    />
                )}
            </div>
        </article>
    )
}

export function BlogPostPage() {
    const { slug, topicSlug } = useParams()
    const post = slug ? getBlogPost(topicSlug, slug) : undefined
    return post ? (
        <Article post={post} key={post.path} />
    ) : (
        <Navigate to="/blog" replace />
    )
}
