import { useEffect } from 'react'
import { Link, Navigate, useLocation, useParams } from 'react-router-dom'
import { MarkdownPost } from '../components/MarkdownPost'
import { getBlogPost, resolveWikiLinks } from '../lib/blog'
import { getMarkdownHeadings } from '../lib/markdown'

function scrollToHeading(id: string) {
    document.getElementById(id)?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
    })
}

export function BlogPostPage() {
    const location = useLocation()
    const { slug, topicSlug } = useParams()
    const post = slug ? getBlogPost(topicSlug, slug) : undefined

    useEffect(() => {
        if (!location.hash) {
            return
        }

        const id = decodeURIComponent(location.hash.slice(1))
        window.setTimeout(() => scrollToHeading(id), 0)
    }, [location.hash, post?.path])

    if (!post) {
        return <Navigate to="/blog" replace />
    }

    const content = resolveWikiLinks(post.content, post)
    const headings = getMarkdownHeadings(post.content, {
        maxLevel: 4,
        minLevel: 1,
    })

    return (
        <article className="page blog-post">
            <Link to={`/blog/topic/${post.topic.slug}`} className="back-link">
                Back to {post.topic.label}
            </Link>

            <header className="post-header">
                <div className="post-meta">
                    <time dateTime={post.date}>{post.date}</time>
                    {post.readingTime}
                    <Link to={`/blog/topic/${post.topic.slug}`}>{post.topic.label}</Link>
                </div>

                <h1>{post.title}</h1>
                <p>{post.summary}</p>

                <div className="tag-list" aria-label="Tags">
                    {post.tags.map((tag) => (
                        <span key={tag}>{tag}</span>
                    ))}
                </div>
            </header>

            <div className="blog-post-layout">
                <MarkdownPost content={content} />

                {headings.length > 0 && (
                    <aside className="post-toc" aria-label="Table of contents">
                        <p>目录</p>
                        <nav>
                            {headings.map((heading) => (
                                <a
                                    className={`toc-level-${heading.level}`}
                                    href={`#${heading.id}`}
                                    key={heading.id}
                                    onClick={(event) => {
                                        event.preventDefault()
                                        scrollToHeading(heading.id)
                                    }}
                                >
                                    {heading.title}
                                </a>
                            ))}
                        </nav>
                    </aside>
                )}
            </div>
        </article>
    )
}
