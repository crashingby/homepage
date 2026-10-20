import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { ArticleTOC } from '../../components/ArticleTOC'
import { Icon } from '../../components/Icon'
import { MarkdownPost } from '../../components/MarkdownPost'
import {
    getInterviewChapter,
    getInterviewChapters,
    type InterviewChapter,
} from '../../lib/interview'
import { getMarkdownHeadings } from '../../lib/markdown'
import { useReadingProgress } from '../../lib/useReadingProgress'

function Chapter({ chapter }: { chapter: InterviewChapter }) {
    const { hash } = useLocation()
    const [largeText, setLargeText] = useState(false)
    const allHeadings = useMemo(
        () => getMarkdownHeadings(chapter.content),
        [chapter.content],
    )
    const duplicateTitle =
        allHeadings[0]?.level === 1 &&
        allHeadings[0].title.replace(/[*`_]/g, '').trim() === chapter.title.trim()
        ? allHeadings[0]
        : undefined
    const headings = useMemo(
        () => allHeadings.filter(
            (heading) => heading.id !== duplicateTitle?.id && heading.level <= 4,
        ),
        [allHeadings, duplicateTitle?.id],
    )
    const { activeId, progress } = useReadingProgress(headings)
    const chapters = getInterviewChapters()
    const index = chapters.findIndex((item) => item.path === chapter.path)
    const previous = chapters[index - 1]
    const next = chapters[index + 1]

    useEffect(() => {
        document.title = `${chapter.title} — 面试指南 — Crashing By`
    }, [chapter.title])

    useEffect(() => {
        if (!hash) return
        let id = hash.slice(1)
        try {
            id = decodeURIComponent(id)
        } catch {
            /* Keep malformed fragments readable. */
        }
        const timer = window.setTimeout(() => {
            document.getElementById(id)?.scrollIntoView({
                block: 'start',
                behavior: 'instant',
            })
        }, 100)
        return () => window.clearTimeout(timer)
    }, [hash])

    return (
        <>
            <div
                className="reading-progress"
                style={{ width: `${progress}%` }}
                aria-hidden="true"
            />
            <nav className="article-breadcrumb interview-breadcrumb" aria-label="面包屑">
                <Link to="/interview">面试指南</Link>
                <Icon name="chevron" size={13} />
                <Link to={`/interview?track=${encodeURIComponent(chapter.track.slug)}`}>
                    {chapter.track.label}
                </Link>
                <Icon name="chevron" size={13} />
                <span aria-current="page">{chapter.title}</span>
            </nav>
            <div className="article-layout interview-reading-layout">
                <article
                    className={`article-main interview-article${largeText ? ' large-text' : ''}`}
                >
                    <header className="interview-chapter-header">
                        <p className="eyebrow">{chapter.track.caption}</p>
                        <h1 id={duplicateTitle?.id}>{chapter.title}</h1>
                        {chapter.summary && (
                            <p className="interview-chapter-summary">{chapter.summary}</p>
                        )}
                        <div className="interview-chapter-tools">
                            <span>
                                第 <span className="mono">{String(index + 1).padStart(2, '0')}</span> 篇 / 共 {chapters.length} 篇指南
                            </span>
                            <div className="reading-controls" aria-label="阅读设置">
                                <button
                                    type="button"
                                    aria-label="放大正文字号"
                                    title="放大正文字号"
                                    aria-pressed={largeText}
                                    onClick={() => setLargeText(!largeText)}
                                >
                                    Aa
                                </button>
                            </div>
                        </div>
                    </header>
                    <div className="article-content">
                        <MarkdownPost
                            content={chapter.content}
                            hiddenHeadingId={duplicateTitle?.id}
                        />
                    </div>
                    <nav className="interview-pagination" aria-label="指南前后篇">
                        {previous ? (
                            <Link to={previous.path}>
                                <span><Icon name="back" size={14} /> 上一篇</span>
                                <strong>{previous.title}</strong>
                            </Link>
                        ) : (
                            <Link to="/interview">
                                <span><Icon name="list" size={14} /> 指南概览</span>
                                <strong>查看全部专题</strong>
                            </Link>
                        )}
                        {next ? (
                            <Link to={next.path} className="interview-next">
                                <span>下一篇 <Icon name="arrow" size={14} /></span>
                                <strong>{next.title}</strong>
                            </Link>
                        ) : (
                            <Link to="/interview" className="interview-next">
                                <span>回到指南 <Icon name="arrow" size={14} /></span>
                                <strong>查看全部专题</strong>
                            </Link>
                        )}
                    </nav>
                </article>
                {headings.length > 0 && (
                    <ArticleTOC headings={headings} activeId={activeId} progress={progress} />
                )}
            </div>
        </>
    )
}

export function InterviewNotFoundPage() {
    useEffect(() => {
        document.title = '未找到指南 — Crashing By'
    }, [])

    return (
        <section className="interview-not-found">
            <p className="eyebrow">404 / INTERVIEW HANDBOOK</p>
            <h1>没有找到这篇指南</h1>
            <p>这篇内容可能已经移动，请从指南概览继续阅读。</p>
            <Link to="/interview" className="primary-button">
                返回面试指南 <Icon name="arrow" size={16} />
            </Link>
        </section>
    )
}

export function InterviewChapterPage() {
    const { trackSlug, slug } = useParams()
    const chapter = trackSlug && slug ? getInterviewChapter(trackSlug, slug) : undefined
    return chapter ? (
        <Chapter chapter={chapter} key={chapter.path} />
    ) : (
        <InterviewNotFoundPage />
    )
}
