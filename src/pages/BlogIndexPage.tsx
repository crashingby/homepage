import { Link, NavLink, Navigate, useParams } from 'react-router-dom'
import { getBlogPosts, getBlogPostsByTopic, getBlogTopics, getBlogTopic } from '../lib/blog'

const topics = getBlogTopics()

export function BlogIndexPage() {
    const { topicSlug } = useParams()
    const activeTopic = topicSlug ? getBlogTopic(topicSlug) : undefined
    const posts = topicSlug ? getBlogPostsByTopic(topicSlug) : getBlogPosts()

    if (topicSlug && !activeTopic) {
        return <Navigate to="/blog" replace />
    }

    return (
        <section className="page blog-index">
            <p className="eyebrow">Notes · Systems · GPU Computing</p>
            <h1>{activeTopic?.label ?? 'Blog'}</h1>
            <p className="page-intro">
                {activeTopic?.description ??
                    'Notes on GPU programming, high-performance computing, and LLM systems.'}
            </p>

            <nav className="topic-nav" aria-label="Blog topics">
                <NavLink to="/blog" end>
                    All
                    <span>{getBlogPosts().length}</span>
                </NavLink>

                {topics.map((topic) => (
                    <NavLink to={`/blog/topic/${topic.slug}`} key={topic.slug}>
                        {topic.label}
                        <span>{getBlogPostsByTopic(topic.slug).length}</span>
                    </NavLink>
                ))}
            </nav>

            <div className="post-list">
                {posts.map((post) => (
                    <article className="post-card" key={`${post.topic.slug}/${post.slug}`}>
                        <div className="post-meta">
                            <time dateTime={post.date}>{post.date}</time>
                            {post.readingTime}
                            <Link to={`/blog/topic/${post.topic.slug}`}>{post.topic.label}</Link>
                        </div>

                        <h2>
                            <Link to={post.path}>{post.title}</Link>
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
