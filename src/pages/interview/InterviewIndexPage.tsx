import { useEffect } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Icon } from '../../components/Icon'
import { getInterviewChapters, getInterviewTracks } from '../../lib/interview'

const chapters = getInterviewChapters()
const tracks = getInterviewTracks()

export function InterviewIndexPage() {
    const [searchParams, setSearchParams] = useSearchParams()
    const query = searchParams.get('q') ?? ''
    const selectedTrack = tracks.find(
        (track) => track.slug === searchParams.get('track'),
    )
    const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean)
    const filtered = chapters.filter((chapter) => {
        const text = `${chapter.title} ${chapter.summary} ${chapter.track.label} ${chapter.content}`.toLocaleLowerCase()
        return (
            (!selectedTrack || chapter.track.slug === selectedTrack.slug) &&
            terms.every((term) => text.includes(term))
        )
    })

    useEffect(() => {
        document.title = '面试指南 — Crashing By'
    }, [])

    function updateFilter(key: string, value: string) {
        const next = new URLSearchParams(searchParams)
        if (value) next.set(key, value)
        else next.delete(key)
        setSearchParams(next, { replace: true })
    }

    return (
        <div className="interview-index">
            <header className="interview-intro">
                <p className="eyebrow">
                    <span className="status-dot" /> THE INTERVIEW HANDBOOK
                </p>
                <h1>
                    面试指南<span>。</span>
                </h1>
                <p className="interview-intro-copy">
                    从写出代码，到讲清系统。<br />
                    集中整理 C++ 与 AI Infra 面试中的问题、思路和实现。
                </p>
                <div className="interview-intro-bottom">
                    <div className="interview-stats">
                        <span>
                            <strong>{String(tracks.length).padStart(2, '0')}</strong> 个专题
                        </span>
                        <span>
                            <strong>{String(chapters.length).padStart(2, '0')}</strong> 篇指南
                        </span>
                    </div>
                    {chapters[0] && (
                        <Link to={chapters[0].path} className="text-link">
                            开始阅读 <Icon name="arrow" size={16} />
                        </Link>
                    )}
                </div>
            </header>

            <section
                className="interview-catalog"
                aria-labelledby="interview-catalog-title"
            >
                <div className="interview-catalog-heading">
                    <h2 id="interview-catalog-title">选择一个专题</h2>
                    <p role="status">
                        {query || selectedTrack ? '找到' : '共'} {filtered.length} 篇指南
                    </p>
                </div>
                <label className="search-field">
                    <Icon name="search" size={18} />
                    <input
                        type="search"
                        aria-label="搜索面试指南"
                        placeholder="搜索问题、关键词或代码…"
                        value={query}
                        onChange={(event) => updateFilter('q', event.target.value)}
                    />
                    {query && (
                        <button
                            type="button"
                            className="icon-button"
                            aria-label="清空搜索"
                            onClick={() => updateFilter('q', '')}
                        >
                            <Icon name="close" size={16} />
                        </button>
                    )}
                </label>
                <div
                    className="interview-filters"
                    role="group"
                    aria-label="筛选面试专题"
                >
                    <button
                        type="button"
                        aria-pressed={!selectedTrack}
                        onClick={() => updateFilter('track', '')}
                    >
                        全部专题
                    </button>
                    {tracks.map((track) => (
                        <button
                            key={track.slug}
                            type="button"
                            aria-pressed={selectedTrack?.slug === track.slug}
                            onClick={() => updateFilter('track', track.slug)}
                        >
                            {track.label}
                        </button>
                    ))}
                </div>
                {filtered.length ? (
                    <div className="interview-track-grid">
                        {tracks.map((track, index) => {
                            const trackChapters = filtered.filter(
                                (chapter) => chapter.track.slug === track.slug,
                            )
                            if (!trackChapters.length) return null
                            return (
                                <section
                                    className="interview-track"
                                    key={track.slug}
                                    aria-labelledby={`track-${track.slug}`}
                                >
                                    <div className="interview-track-top">
                                        <span className="interview-track-icon">
                                            <Icon name={track.slug === 'cpp' ? 'code' : 'layers'} size={23} />
                                        </span>
                                        <span className="mono">
                                            {String(index + 1).padStart(2, '0')} /
                                        </span>
                                    </div>
                                    <p className="eyebrow">{track.caption}</p>
                                    <h3 id={`track-${track.slug}`}>{track.label}</h3>
                                    <p className="interview-track-description">
                                        {track.description}
                                    </p>
                                    <div className="interview-chapter-list">
                                        {trackChapters.map((chapter) => (
                                            <Link
                                                to={chapter.path}
                                                className="interview-chapter-card"
                                                key={chapter.path}
                                            >
                                                <h4>
                                                    {chapter.title}
                                                    <Icon name="arrow" size={17} />
                                                </h4>
                                                <p>{chapter.summary}</p>
                                            </Link>
                                        ))}
                                    </div>
                                </section>
                            )
                        })}
                    </div>
                ) : (
                    <div className="empty-state">
                        <Icon name="search" size={30} />
                        <h3>没有找到相关指南</h3>
                        <p>试试其他关键词，或查看全部专题。</p>
                        <button
                            type="button"
                            className="secondary-button"
                            onClick={() => setSearchParams({}, { replace: true })}
                        >
                            重置筛选
                        </button>
                    </div>
                )}
            </section>
        </div>
    )
}
