import { Link, Navigate, useParams } from 'react-router-dom'
import { MarkdownPost } from '../components/MarkdownPost'
import { getBlogPost } from '../lib/blog'

export function BlogPostPage() {
    const { slug } = useParams()
    const post = slug ? getBlogPost(slug) : undefined

    if (!post) {
        return <Navigate to="/blog" replace />
    }

    return (
        <article className="page blog-post">
            <Link to="/blog" className="back-link">
                Back to blog
            </Link>

            <header className="post-header">
                <div className="post-meta">
                    <time dateTime={post.date}>{post.date}</time>
                    {post.readingTime}
                </div>

                <h1>{post.title}</h1>
                <p>{post.summary}</p>

                <div className="tag-list" aria-label="Tags">
                    {post.tags.map((tag) => (
                        <span key={tag}>{tag}</span>
                    ))}
                </div>
            </header>

            <MarkdownPost content={post.content} />
        </article>
    )
}
