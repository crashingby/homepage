import { useEffect } from 'react'
import { Link, Navigate, useParams, useSearchParams } from 'react-router-dom'
import { Icon } from '../components/Icon'
import { PostList } from '../components/PostList'
import {
    getBlogCategories,
    getBlogPosts,
    getBlogPostsByCategory,
    getBlogPostsByTopic,
    getBlogTopics,
    getBlogTopic,
} from '../lib/blog'

const topics = getBlogTopics()

export function BlogIndexPage() {
    const { topicSlug } = useParams()
    const [searchParams, setSearchParams] = useSearchParams()
    const query = searchParams.get('q') ?? ''
    const categorySlug = searchParams.get('category') ?? ''
    const activeTopic = topicSlug ? getBlogTopic(topicSlug) : undefined
    const categories = topicSlug ? getBlogCategories(topicSlug) : []
    const activeCategory = categories.find((category) => category.slug === categorySlug)
    const posts = topicSlug
        ? activeCategory
            ? getBlogPostsByCategory(topicSlug, activeCategory.slug)
            : getBlogPostsByTopic(topicSlug)
        : getBlogPosts()
    const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean)
    const filtered = terms.length ? posts.filter((post) => {
        const searchable =
            `${post.title} ${post.summary} ${post.tags.join(' ')} ${post.category?.label ?? ''} ${post.content}`.toLocaleLowerCase()
        return terms.every((term) => searchable.includes(term))
    }) : posts

    useEffect(() => {
        document.title = `${activeCategory?.label ?? activeTopic?.label ?? '所有笔记'} — Crashing By`
    }, [activeCategory?.label, activeTopic?.label])

    useEffect(() => {
        window.scrollTo({ top: 0, behavior: 'instant' })
    }, [topicSlug, categorySlug])

    function categorySearch(category?: string) {
        const next = new URLSearchParams(searchParams)
        next.delete('category')
        if (category) next.set('category', category)
        return next.toString()
    }

    function updateQuery(value: string) {
        setSearchParams((previous) => {
            const next = new URLSearchParams(previous)
            if (value) next.set('q', value)
            else next.delete('q')
            return next
        }, { replace: true })
    }

    if (topicSlug && !activeTopic) return <Navigate to="/blog" replace />
    if (categorySlug && !activeCategory) {
        return <Navigate to={{
            pathname: topicSlug ? `/blog/topic/${topicSlug}` : '/blog',
            search: categorySearch(),
        }} replace />
    }

    const showGroups = categories.length > 0 && !activeCategory
    const filteredPaths = new Set(filtered.map((post) => post.path))
    const groups = showGroups ? categories.map((category) => ({
        category,
        posts: getBlogPostsByCategory(topicSlug!, category.slug)
            .filter((post) => filteredPaths.has(post.path)),
    })).filter((group) => group.posts.length > 0) : []
    const uncategorized = showGroups ? filtered.filter((post) => !post.category) : []
    const sortLabel = showGroups ? '按分类浏览'
        : activeCategory?.series ? '按系列顺序' : '按时间排序'

    return (
        <section className="page blog-index">
            <header className="archive-header">
                <p className="eyebrow">THE ARCHIVE</p>
                <h1>
                    {categories.length ? activeTopic?.label : '写下来，才更清楚'}<span>。</span>
                </h1>
                <p>
                    {categories.length ? activeTopic?.description
                        : '关于 GPU、C++ 和系统的学习笔记。也是一份持续生长的知识地图。'}
                </p>
            </header>
            <div className="archive-layout">
                <aside className="archive-sidebar">
                    <div className="sidebar-label">
                        探索专题 <span>{topics.length}</span>
                    </div>
                    <nav className="topic-nav" aria-label="笔记专题">
                        <Link
                            to={{ pathname: '/blog', search: categorySearch() }}
                            className={!topicSlug ? 'active' : undefined}
                            aria-current={!topicSlug ? 'page' : undefined}
                        >
                            全部笔记 <span>{getBlogPosts().length}</span>
                        </Link>
                        {topics.map((topic) => (
                            <div className="topic-nav-group" key={topic.slug}>
                                <Link
                                    to={{ pathname: `/blog/topic/${topic.slug}`, search: categorySearch() }}
                                    className={topicSlug === topic.slug
                                        ? activeCategory ? 'is-parent-active' : 'active'
                                        : undefined}
                                    aria-current={topicSlug === topic.slug && !activeCategory ? 'page' : undefined}
                                >
                                    {topic.label}
                                    <span>{getBlogPostsByTopic(topic.slug).length}</span>
                                </Link>
                                {topicSlug === topic.slug && categories.length > 0 && (
                                    <div className="topic-subnav" role="group" aria-label={`${topic.label}分类`}>
                                        {categories.map((category) => (
                                            <Link
                                                key={category.slug}
                                                to={{ pathname: `/blog/topic/${topic.slug}`, search: categorySearch(category.slug) }}
                                                className={activeCategory?.slug === category.slug ? 'active' : undefined}
                                                aria-current={activeCategory?.slug === category.slug ? 'page' : undefined}
                                            >
                                                {category.label}
                                                <span>{getBlogPostsByCategory(topic.slug, category.slug).length}</span>
                                            </Link>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </nav>
                    <p className="sidebar-note">一点一滴，<br />把知识连成系统。</p>
                </aside>
                <div className="archive-main">
                    <label className="search-field">
                        <Icon name="search" size={19} />
                        <input
                            type="search"
                            aria-label="搜索笔记"
                            placeholder={activeCategory ? `搜索${activeCategory.label}…` : '搜索标题、关键词或正文…'}
                            value={query}
                            onChange={(event) => updateQuery(event.target.value)}
                        />
                        {query && (
                            <button type="button" className="icon-button" onClick={() => updateQuery('')} aria-label="清空搜索">
                                <Icon name="close" size={16} />
                            </button>
                        )}
                    </label>
                    {categories.length > 0 && (
                        <nav className="category-navigation" aria-label={`${activeTopic?.label}文章分类`}>
                            <div className="category-navigation-header">
                                <span>按方向浏览</span>
                                <Link
                                    to={{ pathname: `/blog/topic/${topicSlug}`, search: categorySearch() }}
                                    aria-current={!activeCategory ? 'page' : undefined}
                                >
                                    全部 {getBlogPostsByTopic(topicSlug!).length} 篇
                                    <Icon name="arrow" size={14} />
                                </Link>
                            </div>
                            <div className="category-grid">
                                {categories.map((category) => (
                                    <Link
                                        key={category.slug}
                                        to={{ pathname: `/blog/topic/${topicSlug}`, search: categorySearch(category.slug) }}
                                        className={`category-card${activeCategory?.slug === category.slug ? ' active' : ''}`}
                                        aria-current={activeCategory?.slug === category.slug ? 'page' : undefined}
                                    >
                                        <span className="category-card-title">
                                            {category.label}
                                            <span>{getBlogPostsByCategory(topicSlug!, category.slug).length}</span>
                                        </span>
                                        <span className="category-card-description">{category.description}</span>
                                    </Link>
                                ))}
                            </div>
                        </nav>
                    )}
                    <div className="archive-results">
                        <h2>{activeCategory?.label ?? activeTopic?.label ?? '全部笔记'}</h2>
                        <span>
                            {filtered.length} 篇{query ? '匹配的笔记' : '笔记'}{' '}
                            <span className="results-order">/ {sortLabel}</span>
                        </span>
                    </div>
                    {activeCategory && <p className="category-description">{activeCategory.description}</p>}
                    {filtered.length ? showGroups ? (
                        <div className="category-sections">
                            {groups.map(({ category, posts: categoryPosts }) => (
                                <section key={category.slug} className="category-section" aria-labelledby={`category-${category.slug}`}>
                                    <header className="category-section-header">
                                        <div>
                                            <h3 id={`category-${category.slug}`}>
                                                <Link to={{ pathname: `/blog/topic/${topicSlug}`, search: categorySearch(category.slug) }}>{category.label}</Link>
                                            </h3>
                                            <p>{category.description}{category.series ? ' 按系列顺序阅读。' : ''}</p>
                                        </div>
                                        <Link to={{ pathname: `/blog/topic/${topicSlug}`, search: categorySearch(category.slug) }} className="category-section-count" aria-label={`查看${category.label}，${categoryPosts.length}篇`}>
                                            {categoryPosts.length} 篇 <Icon name="arrow" size={15} />
                                        </Link>
                                    </header>
                                    <PostList posts={categoryPosts} headingLevel={4} />
                                </section>
                            ))}
                            {uncategorized.length > 0 && (
                                <section className="category-section" aria-labelledby="category-other">
                                    <header className="category-section-header"><h3 id="category-other">其他笔记</h3></header>
                                    <PostList posts={uncategorized} headingLevel={4} />
                                </section>
                            )}
                        </div>
                    ) : <PostList posts={filtered} headingLevel={3} /> : (
                        <div className="empty-state">
                            <Icon name="search" size={32} />
                            <h3>还没找到这篇笔记</h3>
                            <p>试试更短的关键词，或者切换到其他分类。</p>
                            {query ? (
                                <button className="secondary-button" onClick={() => updateQuery('')}>清空搜索</button>
                            ) : (
                                <Link className="secondary-button" to={{ pathname: topicSlug ? `/blog/topic/${topicSlug}` : '/blog', search: categorySearch() }}>查看全部笔记</Link>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </section>
    )
}
