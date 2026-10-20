import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { Icon } from '../components/Icon'
import { Nav } from '../components/Nav'
import { getInterviewChapters, getInterviewTracks } from '../lib/interview'

const chapters = getInterviewChapters()
const tracks = getInterviewTracks()

function InterviewSidebar() {
    const [open, setOpen] = useState(false)

    return (
        <aside className={`interview-sidebar${open ? ' is-open' : ''}`}>
            <div className="interview-sidebar-heading">
                <Link to="/interview" className="interview-brand">
                    <span className="interview-brand-icon">
                        <Icon name="book" size={22} />
                    </span>
                    <span>
                        面试指南<small>THE INTERVIEW HANDBOOK</small>
                    </span>
                </Link>
                <button
                    type="button"
                    className="interview-nav-toggle"
                    aria-expanded={open}
                    aria-controls="interview-navigation"
                    onClick={() => setOpen(!open)}
                >
                    章节导航 <Icon name={open ? 'close' : 'list'} size={17} />
                </button>
            </div>
            <nav
                id="interview-navigation"
                className="interview-navigation"
                aria-label="面试指南章节"
            >
                <NavLink to="/interview" end className="interview-overview-link">
                    <Icon name="list" size={16} /> 指南概览
                </NavLink>
                {tracks.map((track, index) => (
                    <div className="interview-nav-group" key={track.slug}>
                        <p>
                            <span>{String(index + 1).padStart(2, '0')}</span>
                            {track.label}
                        </p>
                        {chapters
                            .filter((chapter) => chapter.track.slug === track.slug)
                            .map((chapter) => (
                                <NavLink key={chapter.path} to={chapter.path}>
                                    {chapter.title}
                                    <Icon name="chevron" size={13} />
                                </NavLink>
                            ))}
                    </div>
                ))}
                <div className="interview-sidebar-note">
                    <p>
                        理解原理，写出实现。<br />
                        也练习把思路讲清楚。
                    </p>
                    <Link to="/">
                        <Icon name="back" size={14} /> 返回个人主页
                    </Link>
                </div>
            </nav>
        </aside>
    )
}

export function InterviewLayout() {
    const { pathname } = useLocation()

    useEffect(() => {
        window.scrollTo({ top: 0, behavior: 'instant' })
    }, [pathname])

    return (
        <div className="interview-site">
            <a
                className="skip-link"
                href="#main-content"
                onClick={(event) => {
                    event.preventDefault()
                    document.getElementById('main-content')?.focus()
                }}
            >
                跳至正文
            </a>
            <Nav />
            <div className="interview-shell">
                <InterviewSidebar key={pathname} />
                <main className="interview-main" id="main-content" tabIndex={-1}>
                    <Outlet />
                </main>
            </div>
            <footer className="interview-footer">
                <Link to="/">Crashing By <span>/ 面试指南</span></Link>
                <span>理解 · 实现 · 表达</span>
            </footer>
        </div>
    )
}
