import { useEffect, useMemo } from 'react'
import { NavLink, Navigate, useParams, useSearchParams } from 'react-router-dom'
import { Icon } from '../components/Icon'
import { PostList } from '../components/PostList'
import {
    getBlogPosts,
    getBlogPostsByTopic,
    getBlogTopics,
    getBlogTopic,
} from '../lib/blog'

const topics = getBlogTopics()
export function BlogIndexPage() {
    const { topicSlug } = useParams()
    const [searchParams, setSearchParams] = useSearchParams()
    const query = searchParams.get('q') ?? ''
    const activeTopic = topicSlug ? getBlogTopic(topicSlug) : undefined
    const posts = topicSlug ? getBlogPostsByTopic(topicSlug) : getBlogPosts()
    const filtered = useMemo(() => {
        const terms = query
            .toLocaleLowerCase()
            .trim()
            .split(/\s+/)
            .filter(Boolean)
        return posts.filter((post) => {
            const searchable =
                `${post.title} ${post.summary} ${post.tags.join(' ')} ${post.content}`.toLocaleLowerCase()
            return terms.every((term) => searchable.includes(term))
        })
    }, [posts, query])
    useEffect(() => {
        document.title = `${activeTopic?.label ?? '所有笔记'} — Crashing By`
    }, [activeTopic?.label])
    if (topicSlug && !activeTopic) return <Navigate to="/blog" replace />
    return (
        <section className="page blog-index">
            <header className="archive-header">
                <p className="eyebrow">THE ARCHIVE</p>
                <h1>
                    写下来，才更清楚<span>。</span>
                </h1>
                <p>
                    关于 GPU、C++ 和系统的学习笔记。也是一份持续生长的知识地图。
                </p>
            </header>
            <div className="archive-layout">
                <aside className="archive-sidebar">
                    <div className="sidebar-label">
                        探索专题 <span>{topics.length}</span>
                    </div>
                    <nav className="topic-nav" aria-label="笔记专题">
                        <NavLink
                            to={{
                                pathname: '/blog',
                                search: searchParams.toString(),
                            }}
                            end
                        >
                            全部笔记 <span>{getBlogPosts().length}</span>
                        </NavLink>
                        {topics.map((topic) => (
                            <NavLink
                                to={{
                                    pathname: `/blog/topic/${topic.slug}`,
                                    search: searchParams.toString(),
                                }}
                                key={topic.slug}
                            >
                                {topic.label}
                                <span>
                                    {getBlogPostsByTopic(topic.slug).length}
                                </span>
                            </NavLink>
                        ))}
                    </nav>
                    <p className="sidebar-note">
                        一点一滴，
                        <br />
                        把知识连成系统。
                    </p>
                </aside>
                <div className="archive-main">
                    <label className="search-field">
                        <Icon name="search" size={19} />
                        <input
                            type="search"
                            aria-label="搜索笔记"
                            placeholder="搜索标题、关键词或正文…"
                            value={query}
                            onChange={(event) =>
                                setSearchParams(
                                    event.target.value
                                        ? { q: event.target.value }
                                        : {},
                                    { replace: true },
                                )
                            }
                        />
                        {query && (
                            <button
                                type="button"
                                className="icon-button"
                                onClick={() =>
                                    setSearchParams({}, { replace: true })
                                }
                                aria-label="清空搜索"
                            >
                                <Icon name="close" size={16} />
                            </button>
                        )}
                    </label>
                    <div className="archive-results">
                        <h2>{activeTopic?.label ?? '全部笔记'}</h2>
                        <span>
                            {filtered.length} 篇{query ? '匹配的笔记' : '笔记'}{' '}
                            <span className="results-order">/ 按时间排序</span>
                        </span>
                    </div>
                    {filtered.length ? (
                        <PostList posts={filtered} />
                    ) : (
                        <div className="empty-state">
                            <Icon name="search" size={32} />
                            <h3>还没找到这篇笔记</h3>
                            <p>试试更短的关键词，或者切换到其他专题。</p>
                            <button
                                className="secondary-button"
                                onClick={() =>
                                    setSearchParams({}, { replace: true })
                                }
                            >
                                清空搜索
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </section>
    )
}
