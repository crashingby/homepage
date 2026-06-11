import { Link } from 'react-router-dom'
import { getBlogPosts } from '../lib/blog'

const posts = getBlogPosts()

export function BlogIndexPage() {
    return (
        <section className="page blog-index">
            <p className="eyebrow">Notes · Systems · GPU Computing</p>
            <h1>Blog</h1>
            <p className="page-intro">
                Notes on GPU programming, high-performance computing, and LLM systems.
            </p>

            <div className="post-list">
                {posts.map((post) => (
                    <article className="post-card" key={post.slug}>
                        <div className="post-meta">
                            <time dateTime={post.date}>{post.date}</time>
                            {post.readingTime}
                        </div>

                        <h2>
                            <Link to={`/blog/${post.slug}`}>{post.title}</Link>
                        </h2>

                        <p>{post.summary}</p>

                        <div className="tag-list" aria-label="Tags">
                            {post.tags.map((tag) => (
                                <span key={tag}>{tag}</span>
                            ))}
                        </div>
                    </article>
                ))}
            </div>
        </section>
    )
}
