import { Link } from 'react-router-dom'
import type { BlogPost } from '../lib/blog'
import { Icon } from './Icon'

export function PostList({
    posts,
    numbered = false,
}: {
    posts: BlogPost[]
    numbered?: boolean
}) {
    return (
        <div className={`post-list${numbered ? ' post-list-numbered' : ''}`}>
            {posts.map((post, index) => (
                <article className="post-row" key={post.path}>
                    {numbered && (
                        <span className="post-number">
                            {String(index + 1).padStart(2, '0')}
                        </span>
                    )}
                    <div className="post-row-content">
                        <div className="post-meta">
                            <Link to={`/blog/topic/${post.topic.slug}`}>
                                {post.topic.label}
                            </Link>
                            <span className="meta-dot">·</span>
                            <time dateTime={post.date}>
                                {post.date === '1970-01-01'
                                    ? '未标注日期'
                                    : post.date}
                            </time>
                            <span className="reading-time">
                                {post.readingTime.replace(
                                    ' min read',
                                    ' 分钟阅读',
                                )}
                            </span>
                        </div>
                        <h2>
                            <Link to={post.path}>{post.title}</Link>
                        </h2>
                        {post.summary && <p>{post.summary}</p>}
                        <div className="row-tags">
                            {post.tags.slice(0, 3).map((tag) => (
                                <span key={tag}>{tag}</span>
                            ))}
                        </div>
                    </div>
                    <Link
                        className="post-row-arrow"
                        to={post.path}
                        aria-label={`阅读 ${post.title}`}
                    >
                        <Icon name="upRight" size={22} />
                    </Link>
                </article>
            ))}
        </div>
    )
}
