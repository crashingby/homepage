import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { MarkdownHeading } from '../lib/markdown'
import { Icon } from './Icon'

type HeadingGroup = { heading: MarkdownHeading; children: MarkdownHeading[] }

export function ArticleTOC({
    headings,
    activeId,
    progress,
}: {
    headings: MarkdownHeading[]
    activeId: string
    progress: number
}) {
    const { pathname } = useLocation()
    const [isOpen, setIsOpen] = useState(false)
    const [expanded, setExpanded] = useState<string[]>([])
    const navRef = useRef<HTMLElement>(null)
    const baseLevel = Math.min(...headings.map((heading) => heading.level))
    const groups: HeadingGroup[] = []
    headings.forEach((heading) => {
        if (heading.level === baseLevel || !groups.length)
            groups.push({ heading, children: [] })
        else groups[groups.length - 1].children.push(heading)
    })

    useEffect(() => {
        const nav = navRef.current
        const current = nav?.querySelector<HTMLElement>(
            '[aria-current="location"]',
        )
        if (!nav || !current) return
        const currentRect = current.getBoundingClientRect()
        const navRect = nav.getBoundingClientRect()
        if (
            currentRect.top < navRect.top ||
            currentRect.bottom > navRect.bottom
        ) {
            nav.scrollTop +=
                currentRect.top - navRect.top - nav.clientHeight / 3
        }
    }, [activeId])

    const headingLink = (heading: MarkdownHeading) => (
        <Link
            to={{ pathname, hash: `#${heading.id}` }}
            aria-current={activeId === heading.id ? 'location' : undefined}
            onClick={() => {
                setIsOpen(false)
                document
                    .getElementById(heading.id)
                    ?.scrollIntoView({
                        behavior: matchMedia('(prefers-reduced-motion: reduce)')
                            .matches
                            ? 'instant'
                            : 'smooth',
                        block: 'start',
                    })
            }}
            style={{
                paddingLeft: `${12 + Math.max(0, heading.level - baseLevel) * 12}px`,
            }}
        >
            <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                    p: ({ children }) => <span>{children}</span>,
                    a: ({ children }) => <span>{children}</span>,
                }}
            >
                {heading.title}
            </ReactMarkdown>
        </Link>
    )

    return (
        <aside
            className={`article-toc${isOpen ? ' is-open' : ''}`}
            aria-label="文章目录"
        >
            <button
                type="button"
                className="mobile-toc-toggle"
                aria-expanded={isOpen}
                aria-controls="article-toc-nav"
                onClick={() => setIsOpen(!isOpen)}
            >
                <span>
                    <Icon name="list" size={17} /> 本文目录
                </span>
                <span>
                    {Math.round(progress)}%{' '}
                    <Icon name={isOpen ? 'close' : 'chevron'} size={16} />
                </span>
            </button>
            <div className="toc-content">
                <div className="toc-heading">
                    <span>本文目录</span>
                    <span className="mono">ON THIS PAGE</span>
                </div>
                <nav id="article-toc-nav" ref={navRef}>
                    {groups.map(({ heading, children }) => {
                        const activeGroup =
                            activeId === heading.id ||
                            children.some((child) => child.id === activeId)
                        const showChildren =
                            activeGroup || expanded.includes(heading.id)
                        return (
                            <div
                                key={heading.id}
                                className={`toc-group${activeGroup ? ' is-active' : ''}`}
                            >
                                <div className="toc-parent">
                                    {headingLink(heading)}
                                    {children.length > 0 && (
                                        <button
                                            type="button"
                                            aria-label={`${showChildren ? '收起' : '展开'} ${heading.title}`}
                                            aria-expanded={showChildren}
                                            disabled={activeGroup}
                                            onClick={() =>
                                                setExpanded((ids) =>
                                                    ids.includes(heading.id)
                                                        ? ids.filter(
                                                              (id) =>
                                                                  id !==
                                                                  heading.id,
                                                          )
                                                        : [...ids, heading.id],
                                                )
                                            }
                                        >
                                            <Icon
                                                name="chevron"
                                                size={13}
                                                style={{
                                                    transform: showChildren
                                                        ? 'rotate(90deg)'
                                                        : undefined,
                                                }}
                                            />
                                        </button>
                                    )}
                                </div>
                                {showChildren &&
                                    children.map((child) => (
                                        <div
                                            className="toc-child"
                                            key={child.id}
                                        >
                                            {headingLink(child)}
                                        </div>
                                    ))}
                            </div>
                        )
                    })}
                </nav>
                <div className="toc-progress">
                    <div>
                        <span>阅读进度</span>
                        <span className="mono">{Math.round(progress)}%</span>
                    </div>
                    <progress
                        value={progress}
                        max={100}
                        aria-label="阅读进度"
                    />
                </div>
                <button
                    className="back-to-top"
                    onClick={() =>
                        window.scrollTo({
                            top: 0,
                            behavior: matchMedia(
                                '(prefers-reduced-motion: reduce)',
                            ).matches
                                ? 'instant'
                                : 'smooth',
                        })
                    }
                >
                    ↑ 回到顶部
                </button>
            </div>
        </aside>
    )
}
